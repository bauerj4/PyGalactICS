#!/usr/bin/env python3
"""
Sample non-equilibrium field ICs from a trained MultiTowerSliceVAE checkpoint.

    . .venv/bin/activate
    OMP_NUM_THREADS=6 python scripts/sample_field_vae.py \\
      runs/ml/field_maps/vae_latent_2026-07-24/multitower_slice_vae.pt \\
      --theta-from-run 54a8faf836a0 --dump evolution/particles/step_001700.npz \\
      --n-samples 4 --out runs/ml/field_maps/vae_latent_2026-07-24/samples

Modes:
  * ``z ~ N(0,I)`` (default) — prior sample conditioned on θ
  * ``--z-from-encode PATH`` — encode a snapshot to μ, optionally add noise
  * ``--interp A B`` — interpolate encoded μ between two dumps (α in [0,1])
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from galacticsics.ml.conditioning import theta_from_record
from galacticsics.ml.fields.binning import (
    ComponentSliceGrid,
    MultiScaleSliceConfig,
    bin_multiscale_slice_stacks,
)
from galacticsics.ml.fields.frame import prepare_shared_frame
from galacticsics.ml.fields.normalize import FieldNormStats, denormalize_stack, normalize_stack
from galacticsics.ml.fields.resample import resample_particles_from_multiscale
from galacticsics.ml.fields.vae import FieldVAEConfig, MultiTowerSliceVAE
from galacticsics.ml.morton.index import resolve_snapshot_t_gyr
from galacticsics.ml.morton.tokenize import _component_ids
from ntropy.analysis.disk_density import bin_plane_density, disk_azimuthal_fourier

COUNT_FRACTIONS = {"disk": 4 / 7, "halo": 2 / 7, "bulge": 1 / 7}
MANIFEST = Path("runs/ml/smoke_morton_vae_cpu/manifest_mw_morton_corpus_v2.json")
WORK = Path("runs/mw_morton_corpus_v2")


def _load_ckpt(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    grids = [
        ComponentSliceGrid(
            **{k: v for k, v in g.items() if k in ComponentSliceGrid.__dataclass_fields__}
        )
        for g in ckpt["cfg_grids"]
    ]
    cfg = MultiScaleSliceConfig(grids=tuple(grids), include_potential=False)
    vc = FieldVAEConfig(
        **{k: v for k, v in ckpt["vae_cfg"].items() if k in FieldVAEConfig.__dataclass_fields__}
    )
    model = MultiTowerSliceVAE(cfg, vae_cfg=vc)
    model.load_state_dict(ckpt["model"])
    model.eval()
    stats = {k: FieldNormStats(**v) for k, v in ckpt["norm"].items()}
    return model, cfg, stats, ckpt


def _theta_from_run(manifest: Path, run: str, dump: str) -> np.ndarray:
    raw = json.loads(manifest.read_text())
    for r in raw["records"]:
        if r["run_hash"] == run and dump in r["path"]:
            t = resolve_snapshot_t_gyr(r["path"], r["t_gyr"])
            return theta_from_record(r["theta"], t_gyr=t)
    raise SystemExit(f"no θ for run={run} dump={dump}")


def _a2(pos, mass) -> float:
    return float(
        disk_azimuthal_fourier(
            pos, mass, m=2, r_max=12.0, n_bins=12, z_max=0.5, min_count=10
        )["a_m_over_a0_median"]
    )


def _mass_true(path: Path) -> dict[str, float]:
    with np.load(path, allow_pickle=True) as data:
        mass = np.asarray(data["mass"], dtype=np.float64)
        tags = data["tags"] if "tags" in data.files else None
        tid = data["type_id"] if "type_id" in data.files else None
    cid = _component_ids(tags, tid, mass.shape[0])
    return {
        "disk": float(mass[cid == 0].sum()) if np.any(cid == 0) else 1.0,
        "halo": float(mass[cid == 1].sum()) if np.any(cid == 1) else 1.0,
        "bulge": float(mass[cid == 2].sum()) if np.any(cid == 2) else 1.0,
    }


def _encode_path(model, cfg, stats, path: Path, theta: np.ndarray) -> torch.Tensor:
    with np.load(path, allow_pickle=True) as data:
        pos = np.asarray(data["pos"], dtype=np.float64)
        vel = np.asarray(data["vel"], dtype=np.float64)
        mass = np.asarray(data["mass"], dtype=np.float64)
        tags = data["tags"] if "tags" in data.files else None
        tid = data["type_id"] if "type_id" in data.files else None
    cid = _component_ids(tags, tid, pos.shape[0])
    pos, vel, _ = prepare_shared_frame(pos, vel, mass, center=True, rotate=False)
    maps = {k: v[0] for k, v in bin_multiscale_slice_stacks(pos, vel, mass, cid, cfg=cfg).items()}
    normed = {k: normalize_stack(maps[k], stats[k]) for k in maps}
    batch = {k: torch.as_tensor(v[None], dtype=torch.float32) for k, v in normed.items()}
    theta_t = torch.as_tensor(theta[None], dtype=torch.float32)
    with torch.no_grad():
        out = model(batch, theta_t, sample_posterior=False)
    return out["mu"].detach()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--theta-from-run", type=str, required=True)
    p.add_argument("--dump", type=str, required=True, help="path suffix under the run dir")
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--n-particles", type=int, default=80_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--z-noise", type=float, default=0.0, help="add to encoded μ if --z-from-encode")
    p.add_argument("--z-from-encode", type=Path, default=None, help="encode this NPZ → μ")
    p.add_argument(
        "--interp",
        nargs=2,
        metavar=("NPZ_A", "NPZ_B"),
        default=None,
        help="interpolate μ_A→μ_B at α=i/(n-1)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    model, cfg, stats, _ckpt = _load_ckpt(args.checkpoint)
    theta = _theta_from_run(args.manifest, args.theta_from_run, args.dump)
    theta_t = torch.as_tensor(theta[None], dtype=torch.float32)
    mass_path = args.work / args.theta_from_run / args.dump
    mass_true = _mass_true(mass_path)
    disk_g = cfg.grid_for("disk")
    n_mom = disk_g.n_mom

    zs: list[torch.Tensor] = []
    tags: list[str] = []
    if args.interp is not None:
        za = _encode_path(model, cfg, stats, Path(args.interp[0]), theta)
        zb = _encode_path(model, cfg, stats, Path(args.interp[1]), theta)
        for i in range(args.n_samples):
            a = i / max(args.n_samples - 1, 1)
            zs.append((1 - a) * za + a * zb)
            tags.append(f"interp_a{a:.2f}")
    elif args.z_from_encode is not None:
        mu = _encode_path(model, cfg, stats, args.z_from_encode, theta)
        for i in range(args.n_samples):
            z = mu + float(args.z_noise) * torch.randn_like(mu)
            zs.append(z)
            tags.append(f"mu+{args.z_noise}eps_{i}")
    else:
        for i in range(args.n_samples):
            zs.append(torch.randn(1, model.cfg.latent_dim))
            tags.append(f"prior_{i}")

    panels = []
    labels = []
    summary = []
    for z, tag in zip(zs, tags):
        with torch.no_grad():
            samp = model.sample(theta_t, z=z)
            pred = {k: v.cpu().numpy()[0] for k, v in samp.items()}
        recon = {k: denormalize_stack(pred[k], stats[k]) for k in pred}
        dens = sum(
            np.maximum(recon["disk"][iz * n_mom], 0.0) for iz in range(disk_g.n_z)
        )
        parts = resample_particles_from_multiscale(
            {g.name: recon[g.name][: g.n_moment_channels] for g in cfg.grids},
            cfg=MultiScaleSliceConfig(grids=cfg.grids, include_potential=False),
            n_particles=args.n_particles,
            mass_total_per_component=mass_true,
            count_fractions=COUNT_FRACTIONS,
            rng=rng,
            sample_dispersion=True,
        )
        np.savez_compressed(
            args.out / f"{tag}_particles.npz",
            pos=parts["pos"],
            vel=parts["vel"],
            mass=parts["mass"],
            eps=parts["eps"],
            component_id=parts["component_id"],
            theta=theta,
            z=z.cpu().numpy()[0],
            tag=tag,
        )
        dmask = parts["component_id"] == 0
        a2 = _a2(parts["pos"][dmask], parts["mass"][dmask])
        pmap = bin_plane_density(
            parts["pos"][dmask],
            parts["mass"][dmask],
            axes=(0, 1),
            n_bins=disk_g.n_pix,
            half_extent=disk_g.r_max,
        ).density
        panels.extend([dens, pmap])
        labels.extend([f"{tag} dens", f"A₂≈{a2:.2f}"])
        summary.append({"tag": tag, "a2_resampled": a2, "z_norm": float(z.norm())})
        print(f"{tag}: A₂≈{a2:.3f} |z|={float(z.norm()):.2f}", flush=True)

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    if n == 1:
        axes = [axes]
    for ax, img, lab in zip(axes, panels, labels):
        pos = img[img > 0]
        vmax = float(np.percentile(pos, 99)) if pos.size else 1.0
        ax.imshow(
            np.log1p(np.maximum(img, 0.0)),
            origin="lower",
            cmap="inferno",
            vmin=0,
            vmax=np.log1p(vmax),
        )
        ax.set_title(lab, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Field VAE samples · run={args.theta_from_run} t_gyr={theta[-1]:.2f}")
    fig.tight_layout()
    fig.savefig(args.out / "sample_panel.png", dpi=140)
    plt.close(fig)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
