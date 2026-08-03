#!/usr/bin/env python3
"""
Sample non-eq field ICs from the frozen-AE + skip-distill latent code model.

Preferred over end-to-end field VAE for barred morphology (see
``runs/ml/field_maps/CREATIVE_LATENT_SUMMARY.md``).

    . .venv/bin/activate
    OMP_NUM_THREADS=6 python scripts/sample_field_latent.py \\
        --ckpt runs/ml/field_maps/creative_latent_morph_*/frozen_ae_code_vae.pt \\
        --mode prior --n 2
    OMP_NUM_THREADS=6 python scripts/sample_field_latent.py --mode interp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from galacticsics.ml.conditioning import DEFAULT_THETA_KEYS, theta_from_record
from galacticsics.ml.fields.binning import MultiScaleSliceConfig, bin_multiscale_slice_stacks
from galacticsics.ml.fields.frame import prepare_shared_frame
from galacticsics.ml.fields.latent_code import FrozenAECodeVAE, LatentCodeConfig, load_frozen_teacher
from galacticsics.ml.fields.normalize import FieldNormStats, denormalize_stack, normalize_stack
from galacticsics.ml.fields.resample import resample_particles_from_multiscale
from galacticsics.ml.morton.index import SnapshotRecord, resolve_snapshot_t_gyr
from galacticsics.ml.morton.polygon import _component_ids


MANIFEST = Path("runs/ml/smoke_morton_vae_cpu/manifest_mw_morton_corpus_v2.json")
WORK = Path("runs/mw_morton_corpus_v2")
CRISP = Path("runs/ml/field_maps/crisp_2026-07-24/multitower_slice_ae.pt")
COUNT_FRACTIONS = {"disk": 4 / 7, "halo": 2 / 7, "bulge": 1 / 7}


def _as_stats(raw):
    return {
        k: (v if isinstance(v, FieldNormStats) else FieldNormStats(**(v if isinstance(v, dict) else v.__dict__)))
        for k, v in raw.items()
    }


def _theta(manifest: Path, run_hash: str) -> np.ndarray:
    raw = json.loads(manifest.read_text())
    for r in raw["records"]:
        rec = SnapshotRecord(**r)
        if rec.run_hash == run_hash:
            t = resolve_snapshot_t_gyr(rec.path, rec.t_gyr)
            return theta_from_record(rec.theta, t_gyr=t)
    raise SystemExit(f"no θ for {run_hash}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--teacher", type=Path, default=CRISP)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--mode", choices=("prior", "encode", "interp"), default="prior")
    p.add_argument("--run", type=str, default="54a8faf836a0", help="θ source run hash")
    p.add_argument("--dump", type=str, default="evolution/particles/step_001700.npz")
    p.add_argument("--quiet-run", type=str, default="081ed8af4b2b")
    p.add_argument("--quiet-dump", type=str, default="ic_state.npz")
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--n-particles", type=int, default=80_000)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    t_args = torch.load(args.teacher, map_location="cpu", weights_only=False).get("args", {})
    cfg = MultiScaleSliceConfig.progressive_defaults(
        disk_n_pix=int(t_args.get("disk_n_pix", 128)),
        include_potential=False,
        moment_set=str(t_args.get("moment_set", "disp")),
    )
    teacher, _ = load_frozen_teacher(args.teacher, cfg, device="cpu")
    lcfg = LatentCodeConfig(**{k: v for k, v in ckpt["cfg"].items() if k in LatentCodeConfig.__dataclass_fields__})
    model = FrozenAECodeVAE(teacher, cfg=lcfg, use_flow_prior=bool(ckpt.get("args", {}).get("use_flow", False)))
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    stats = _as_stats(ckpt["norm"])
    out = args.out or (args.ckpt.parent / f"sample_{args.mode}")
    out.mkdir(parents=True, exist_ok=True)

    theta = torch.as_tensor(_theta(args.manifest, args.run)[None], dtype=torch.float32)
    target_shapes = {g.name: (g.n_pix, g.n_pix) for g in cfg.grids}
    n_ch = {g.name: g.n_moment_channels for g in cfg.grids}

    def _save(i, stacks_norm, tag):
        den = {k: denormalize_stack(stacks_norm[k][0].numpy(), stats[k]) for k in stacks_norm}
        parts = resample_particles_from_multiscale(
            den, cfg=cfg, n_particles=args.n_particles, count_fractions=COUNT_FRACTIONS, rng=rng
        )
        path = out / f"{tag}_{i:02d}.npz"
        np.savez_compressed(path, **parts)
        print(f"wrote {path}  n={parts['pos'].shape[0]}")

    with torch.no_grad():
        if args.mode == "prior":
            for i in range(args.n):
                stacks = model.sample(theta, target_shapes=target_shapes, n_channels=n_ch)
                _save(i, stacks, "prior")
        elif args.mode == "encode":
            path = args.work / args.run / args.dump
            with np.load(path, allow_pickle=True) as data:
                pos = np.asarray(data["pos"], dtype=np.float64)
                vel = np.asarray(data["vel"], dtype=np.float64)
                mass = np.asarray(data["mass"], dtype=np.float64)
                tags = data["tags"] if "tags" in data.files else None
                type_id = data["type_id"] if "type_id" in data.files else None
            cid = _component_ids(tags, type_id, pos.shape[0])
            pos, vel, _ = prepare_shared_frame(pos, vel, mass, center=True, rotate=False)
            binned = bin_multiscale_slice_stacks(pos, vel, mass, cid, cfg=cfg)
            batch = {
                k: torch.as_tensor(normalize_stack(v[0], stats[k])[None], dtype=torch.float32)
                for k, v in binned.items()
            }
            mu = model.encode_mu(batch)
            stacks = model.sample(theta, z=mu, target_shapes=target_shapes, n_channels=n_ch)
            _save(0, stacks, "encode_mu")
            np.save(out / "z_mu.npy", mu.numpy())
        else:
            # quiet → bar interp
            def _load(run, dump):
                path = args.work / run / dump
                with np.load(path, allow_pickle=True) as data:
                    pos = np.asarray(data["pos"], dtype=np.float64)
                    vel = np.asarray(data["vel"], dtype=np.float64)
                    mass = np.asarray(data["mass"], dtype=np.float64)
                    tags = data["tags"] if "tags" in data.files else None
                    type_id = data["type_id"] if "type_id" in data.files else None
                cid = _component_ids(tags, type_id, pos.shape[0])
                pos, vel, _ = prepare_shared_frame(pos, vel, mass, center=True, rotate=False)
                binned = bin_multiscale_slice_stacks(pos, vel, mass, cid, cfg=cfg)
                return {
                    k: torch.as_tensor(normalize_stack(v[0], stats[k])[None], dtype=torch.float32)
                    for k, v in binned.items()
                }

            z0 = model.encode_mu(_load(args.quiet_run, args.quiet_dump))
            z1 = model.encode_mu(_load(args.run, args.dump))
            for i, a in enumerate(np.linspace(0, 1, args.n)):
                z = (1 - a) * z0 + a * z1
                stacks = model.sample(theta, z=z, target_shapes=target_shapes, n_channels=n_ch)
                _save(i, stacks, f"interp_a{a:.2f}")
    print(f"done → {out}")


if __name__ == "__main__":
    main()
