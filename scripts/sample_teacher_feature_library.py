#!/usr/bin/env python3
"""
Usable latent morphing via frozen teacher feature library (no skip-synth washout).

Bars live in U-Net skips.  This path:
  1. Encodes corpus snaps with frozen crisp AE → stores bottleneck+skips
  2. Compresses to a single vector ``z`` (pooled bottleneck flatten → PCA/μ)
  3. Samples by retrieving / interpolating library members in ``z``-space
  4. Blends full teacher features and decodes with frozen AE

Quiet↔bar encode–interp–decode is already known to morph A₂; this packages it
as a single-latent sampling API without end-to-end VAE decode-from-z.

    OMP_NUM_THREADS=6 python scripts/sample_teacher_feature_library.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from galacticsics.ml.conditioning import DEFAULT_THETA_KEYS, theta_from_record
from galacticsics.ml.fields.autoencoder import dens_map_azimuthal_fourier_numpy
from galacticsics.ml.fields.binning import MultiScaleSliceConfig, bin_multiscale_slice_stacks
from galacticsics.ml.fields.dataset import MultiScaleFieldDataset
from galacticsics.ml.fields.frame import prepare_shared_frame
from galacticsics.ml.fields.latent_code import load_frozen_teacher
from galacticsics.ml.fields.normalize import FieldNormStats, denormalize_stack, normalize_stack
from galacticsics.ml.fields.resample import resample_particles_from_multiscale
from galacticsics.ml.morton.index import SnapshotRecord, resolve_snapshot_t_gyr
from galacticsics.ml.morton.tokenize import _component_ids
from ntropy.analysis.disk_density import disk_azimuthal_fourier


MANIFEST = Path("runs/ml/smoke_morton_vae_cpu/manifest_mw_morton_corpus_v2.json")
WORK = Path("runs/mw_morton_corpus_v2")
CRISP = Path("runs/ml/field_maps/crisp_2026-07-24/multitower_slice_ae.pt")
COUNT = {"disk": 4 / 7, "halo": 2 / 7, "bulge": 1 / 7}


def _as_stats(raw):
    return {
        k: (v if isinstance(v, FieldNormStats) else FieldNormStats(**v))
        for k, v in raw.items()
    }


def _a2_disk(parts):
    pos, mass, cid = parts["pos"], parts["mass"], parts["component_id"]
    disk = cid == 0
    out = disk_azimuthal_fourier(
        pos[disk], mass[disk], m=2, r_max=12.0, n_bins=12, z_max=0.5, min_count=10
    )
    return float(out["a_m_over_a0_median"])


def _pool_z(features, enc_grid=4):
    flats = []
    for name in sorted(features.keys()):
        b = features[name]["bottleneck"]
        spat = torch.nn.functional.adaptive_avg_pool2d(b, (enc_grid, enc_grid))
        flats.append(spat.flatten(1))
    return torch.cat(flats, dim=-1)


def _blend_features(fa, fb, alpha: float):
    out = {}
    for name in fa:
        ba, sa = fa[name]["bottleneck"], fa[name]["skips"]
        bb, sb = fb[name]["bottleneck"], fb[name]["skips"]
        out[name] = {
            "bottleneck": (1 - alpha) * ba + alpha * bb,
            "skips": tuple((1 - alpha) * a + alpha * b for a, b in zip(sa, sb)),
        }
    return out


def _disk_collapse(stack, n_z, n_mom):
    dens = np.zeros(stack.shape[-2:], dtype=np.float64)
    for iz in range(n_z):
        dens += np.maximum(stack[iz * n_mom], 0.0)
    return dens


def _plot(path, panels, labels, title):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    if n == 1:
        axes = [axes]
    for ax, img, lab in zip(axes, panels, labels):
        pos = img[img > 0]
        vmax = float(np.percentile(pos, 99)) if pos.size else 1.0
        ax.imshow(np.log1p(np.maximum(img, 0)), origin="lower", cmap="inferno", vmin=0, vmax=np.log1p(vmax))
        ax.set_title(lab, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("runs/ml/field_maps/creative_feature_library_2026-07-25"))
    p.add_argument("--teacher", type=Path, default=CRISP)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--max-snap", type=int, default=48)
    p.add_argument("--enc-grid", type=int, default=4)
    p.add_argument("--n-resample", type=int, default=80_000)
    p.add_argument("--n-interp", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    t_probe = torch.load(args.teacher, map_location="cpu", weights_only=False)
    t_args = t_probe.get("args", {})
    cfg = MultiScaleSliceConfig.progressive_defaults(
        disk_n_pix=int(t_args.get("disk_n_pix", 128)),
        include_potential=False,
        moment_set=str(t_args.get("moment_set", "disp")),
    )
    teacher, t_ckpt = load_frozen_teacher(args.teacher, cfg, device="cpu")
    stats = _as_stats(t_ckpt["norm"])
    disk_g = cfg.grid_for("disk")

    ds = MultiScaleFieldDataset(
        args.manifest,
        cfg=cfg,
        split=None,
        max_snapshots=args.max_snap,
        seed=args.seed,
        augment=False,
        include_potential=False,
        theta_keys=DEFAULT_THETA_KEYS,
        norm_stats=stats,
    )
    ds.preload()
    print(f"building feature library n={len(ds)}…", flush=True)

    lib_z = []
    lib_feat = []
    lib_meta = []
    lib_a2 = []
    for i in range(len(ds)):
        item = ds[i]
        stacks = {
            k: torch.as_tensor(v[None], dtype=torch.float32)
            for k, v in item["stacks"].items()
        }
        with torch.no_grad():
            feat = teacher.encode_features(stacks)
            z = _pool_z(feat, enc_grid=args.enc_grid)[0].cpu().numpy()
        # dens-map A₂ proxy for ranking
        den = denormalize_stack(stacks["disk"][0].numpy(), stats["disk"])
        dens = _disk_collapse(den, disk_g.n_z, disk_g.n_mom)
        a2 = float(dens_map_azimuthal_fourier_numpy(dens, m=2, r_max=12.0)["a_m_over_a0_median"])
        # Keep CPU tensors for blend (detach)
        feat_cpu = {
            name: {
                "bottleneck": feat[name]["bottleneck"].cpu(),
                "skips": tuple(s.cpu() for s in feat[name]["skips"]),
            }
            for name in feat
        }
        lib_z.append(z)
        lib_feat.append(feat_cpu)
        lib_meta.append({"idx": i, "path": ds.records[i].path, "a2_map": a2})
        lib_a2.append(a2)
        if (i + 1) % 8 == 0:
            print(f"  {i+1}/{len(ds)}", flush=True)

    Z = np.stack(lib_z, axis=0)
    A2 = np.asarray(lib_a2)
    # PCA to a single compact latent for the sampling API
    Zc = Z - Z.mean(0, keepdims=True)
    u, s, vt = np.linalg.svd(Zc, full_matrices=False)
    n_pc = min(64, Z.shape[1], Z.shape[0] - 1)
    W = vt[:n_pc].T  # (D, n_pc)
    codes = Zc @ W  # (N, n_pc)
    np.savez_compressed(
        args.out / "feature_library_codes.npz",
        codes=codes,
        a2=A2,
        mean=Z.mean(0),
        pca_w=W,
        paths=np.asarray([m["path"] for m in lib_meta]),
    )
    print(f"library codes shape={codes.shape}  A₂ range {A2.min():.3f}–{A2.max():.3f}", flush=True)

    # Pick quiet / barred extremes by map A₂
    i_quiet = int(np.argmin(A2))
    i_bar = int(np.argmax(A2))
    print(f"quiet idx={i_quiet} A₂={A2[i_quiet]:.3f}  bar idx={i_bar} A₂={A2[i_bar]:.3f}", flush=True)

    verdict = {
        "approach": "frozen teacher feature library + z=PCA(pooled bottleneck)",
        "n_library": len(lib_meta),
        "latent_dim": int(n_pc),
        "quiet_idx": i_quiet,
        "bar_idx": i_bar,
        "interp": [],
        "prior_knn": [],
    }

    # Quiet→bar feature interpolation (known to work)
    panels, labels, rows = [], [], []
    for alpha in np.linspace(0, 1, args.n_interp):
        feat = _blend_features(lib_feat[i_quiet], lib_feat[i_bar], float(alpha))
        with torch.no_grad():
            out_s = teacher.decode_features(
                feat,
                target_shapes={g.name: (g.n_pix, g.n_pix) for g in cfg.grids},
                n_channels={g.name: g.n_moment_channels for g in cfg.grids},
            )
        den = {k: denormalize_stack(out_s[k][0].numpy(), stats[k]) for k in out_s}
        dens = _disk_collapse(den["disk"], disk_g.n_z, disk_g.n_mom)
        am = dens_map_azimuthal_fourier_numpy(dens, m=2, r_max=12.0)
        a2_map = float(am["a_m_over_a0_median"])
        a2_part = None
        if alpha in (0.0, 0.5, 1.0) or abs(alpha - 0.5) < 1e-9:
            parts = resample_particles_from_multiscale(
                den, cfg=cfg, n_particles=args.n_resample, count_fractions=COUNT, rng=rng
            )
            a2_part = _a2_disk(parts)
        rows.append({"alpha": float(alpha), "a2_map": a2_map, "a2_part": a2_part})
        panels.append(dens)
        labels.append(f"α={alpha:.2f}\nA₂m={a2_map:.2f}" + (f"\nA₂p={a2_part:.2f}" if a2_part else ""))
    _plot(args.out / "library_interp_quiet_to_bar.png", panels, labels, "Teacher-feature library interp")
    verdict["interp"] = rows
    print("interp part A₂:", [r.get("a2_part") for r in rows], flush=True)

    # Prior-like: sample random barred codes (top quartile A₂), decode nearest library member
    # and a blend toward a second barred neighbor (morphology-preserving prior).
    thr = float(np.quantile(A2, 0.75))
    barred_idx = np.where(A2 >= thr)[0]
    quiet_idx = np.where(A2 <= float(np.quantile(A2, 0.25)))[0]
    for kind, pool in (("barred", barred_idx), ("quiet", quiet_idx)):
        for j in range(3):
            i0 = int(rng.choice(pool))
            # nearest other in code space
            d = np.linalg.norm(codes - codes[i0], axis=1)
            d[i0] = np.inf
            i1 = int(np.argmin(d))
            alpha = float(rng.uniform(0.0, 0.35))
            feat = _blend_features(lib_feat[i0], lib_feat[i1], alpha)
            with torch.no_grad():
                out_s = teacher.decode_features(
                    feat,
                    target_shapes={g.name: (g.n_pix, g.n_pix) for g in cfg.grids},
                    n_channels={g.name: g.n_moment_channels for g in cfg.grids},
                )
            den = {k: denormalize_stack(out_s[k][0].numpy(), stats[k]) for k in out_s}
            dens = _disk_collapse(den["disk"], disk_g.n_z, disk_g.n_mom)
            a2_map = float(
                dens_map_azimuthal_fourier_numpy(dens, m=2, r_max=12.0)["a_m_over_a0_median"]
            )
            parts = resample_particles_from_multiscale(
                den, cfg=cfg, n_particles=args.n_resample, count_fractions=COUNT, rng=rng
            )
            a2_part = _a2_disk(parts)
            verdict["prior_knn"].append(
                {
                    "kind": kind,
                    "i0": i0,
                    "i1": i1,
                    "alpha": alpha,
                    "a2_map": a2_map,
                    "a2_part": a2_part,
                    "z": codes[i0].tolist(),
                }
            )
            print(f"  {kind} knn sample A₂ part={a2_part:.3f} map={a2_map:.3f}", flush=True)

    bar_parts = [x["a2_part"] for x in verdict["prior_knn"] if x["kind"] == "barred"]
    quiet_parts = [x["a2_part"] for x in verdict["prior_knn"] if x["kind"] == "quiet"]
    bar_mean = float(np.mean(bar_parts)) if bar_parts else None
    quiet_mean = float(np.mean(quiet_parts)) if quiet_parts else None
    interp_bar = next((r["a2_part"] for r in rows[::-1] if r["a2_part"] is not None), None)
    better = (
        bar_mean is not None
        and bar_mean > 0.15
        and quiet_mean is not None
        and quiet_mean < 0.08
    )
    verdict["bar_prior_mean"] = bar_mean
    verdict["quiet_prior_mean"] = quiet_mean
    verdict["interp_bar_part"] = interp_bar
    verdict["better_than_overnight_vae"] = better
    verdict["overnight_vae_bar_mu_ref"] = 0.058
    (args.out / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps({"better": better, "bar_prior_mean": bar_mean, "quiet_prior_mean": quiet_mean, "interp_bar": interp_bar}, indent=2))
    if better:
        Path("runs/ml/field_maps/LATEST").write_text(str(args.out.resolve()) + "\n")
    print(f"done → {args.out}")


if __name__ == "__main__":
    main()
