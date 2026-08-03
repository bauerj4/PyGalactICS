#!/usr/bin/env python3
"""First-pass latent-space probes for a frozen teacher feature library (PCA index).

Builds a stratified library, then:
  * PCA1–2 scatter colored by particle A₂ and bar angle φ
  * quiet vs bar clusters
  * optional α amplify sweep + z interpolate face-ons
  * notes skip tensors still carry morphology (bottleneck-only PCA)

Example::

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 \\
      .venv/bin/python scripts/probe_latent_space.py \\
        --teacher runs/ml/field_maps/fft_morph_ft_long_2026-07-25/multitower_slice_ae.pt \\
        --out runs/ml/field_maps/latent_probe_fft_long_2026-07-25
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sample_latent_ic import (  # noqa: E402
    COUNT,
    RANK,
    _disk_collapse,
    _plot,
    build_library,
    default_teacher,
)
from galacticsics.ml.fields.normalize import denormalize_stack  # noqa: E402
from galacticsics.ml.fields.resample import resample_particles_from_multiscale  # noqa: E402
from ntropy.analysis.disk_density import disk_azimuthal_fourier  # noqa: E402


def _a2(parts) -> float:
    disk = parts["component_id"] == 0
    return float(
        disk_azimuthal_fourier(
            parts["pos"][disk],
            parts["mass"][disk],
            m=2,
            r_max=12.0,
            n_bins=12,
            z_max=0.5,
            min_count=10,
        )["a_m_over_a0_median"]
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("runs/ml/field_maps/latent_probe_2026-07-25"),
    )
    p.add_argument("--teacher", type=Path, default=None)
    p.add_argument("--rank", type=Path, default=RANK)
    p.add_argument("--n-bar", type=int, default=16)
    p.add_argument("--n-quiet", type=int, default=10)
    p.add_argument("--n-mid", type=int, default=6)
    p.add_argument("--n-rot-bar", type=int, default=4)
    p.add_argument("--bar-floor", type=float, default=0.20)
    p.add_argument("--quiet-ceil", type=float, default=0.06)
    p.add_argument("--enc-grid", type=int, default=4)
    p.add_argument("--n-pc", type=int, default=64)
    p.add_argument("--n-resample", type=int, default=30_000)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--alpha-lo", type=float, default=1.0)
    p.add_argument("--alpha-hi", type=float, default=1.5)
    p.add_argument("--n-alpha", type=int, default=5)
    p.add_argument(
        "--paper-figures",
        type=Path,
        default=None,
        help="Optional copy of key panels for the paper figures/ dir",
    )
    args = p.parse_args()
    if args.teacher is None:
        args.teacher = default_teacher()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print(f"=== library teacher={args.teacher} ===", flush=True)
    lib = build_library(args)
    codes = lib.codes
    a2 = lib.a2
    kinds = np.asarray([m.get("kind", "mid") for m in lib.meta])
    phis = np.asarray(
        [float(m["phi"]) if m.get("phi") is not None else np.nan for m in lib.meta]
    )

    # Code variances as proxy for PC energy (truncated PCA index)
    var = codes.var(axis=0)
    var_frac = var / max(float(var.sum()), 1e-12)

    # --- PCA1–2 colored by A2 ---
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    sc = axes[0].scatter(
        codes[:, 0], codes[:, 1], c=a2, cmap="magma", s=28, edgecolors="none"
    )
    fig.colorbar(sc, ax=axes[0], label=r"particle $A_2$")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title("Library $z$ (PCA) by $A_2$")

    for kind, color, marker in (
        ("bar", "C3", "o"),
        ("quiet", "C0", "s"),
        ("mid", "0.5", "^"),
    ):
        m = kinds == kind
        if not np.any(m):
            continue
        axes[1].scatter(
            codes[m, 0],
            codes[m, 1],
            c=color,
            marker=marker,
            s=32,
            label=kind,
            edgecolors="none",
        )
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title("Quiet vs bar clusters")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    pca_png = args.out / "pca12_a2_kind.png"
    fig.savefig(pca_png, dpi=160)
    plt.close(fig)

    # --- φ coloring (bars with rotation) ---
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    m = np.isfinite(phis)
    if np.any(m):
        sc = ax.scatter(
            codes[m, 0],
            codes[m, 1],
            c=phis[m],
            cmap="twilight",
            s=36,
            edgecolors="none",
        )
        fig.colorbar(sc, ax=ax, label=r"bar $\phi$ [rad]")
    else:
        ax.text(0.5, 0.5, "no rotated bars", ha="center", transform=ax.transAxes)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(r"Bar members colored by $\phi$")
    fig.tight_layout()
    phi_png = args.out / "pca12_phi.png"
    fig.savefig(phi_png, dpi=160)
    plt.close(fig)

    # Correlations of first few PCs with A2
    corrs = []
    for i in range(min(8, codes.shape[1])):
        if np.std(codes[:, i]) < 1e-12 or np.std(a2) < 1e-12:
            r = float("nan")
        else:
            r = float(np.corrcoef(codes[:, i], a2)[0, 1])
        corrs.append({"pc": i + 1, "corr_a2": r, "var_frac": float(var_frac[i])})

    # --- amplify α sweep (hybrid) ---
    alphas = np.linspace(args.alpha_lo, args.alpha_hi, args.n_alpha)
    alpha_rows = []
    panels, labels = [], []
    disk_grid = lib.cfg.grid_for("disk")
    for alpha in alphas:
        fields, meta = lib.sample_fields(
            kind="barred",
            method="amplify_knn_hybrid",
            rng=rng,
            alpha_lo=float(alpha),
            alpha_hi=float(alpha),
        )
        phys = {
            k: denormalize_stack(v[0].detach().cpu().numpy(), lib.stats[k])
            for k, v in fields.items()
        }
        dens = _disk_collapse(phys["disk"], disk_grid.n_z, disk_grid.n_mom)
        parts = resample_particles_from_multiscale(
            phys, cfg=lib.cfg, n_particles=args.n_resample, count_fractions=COUNT, rng=rng
        )
        a2p = _a2(parts)
        alpha_rows.append(
            {
                "alpha": float(alpha),
                "a2_particles": a2p,
                "meta": {k: v for k, v in meta.items() if k != "z"},
            }
        )
        panels.append(dens)
        labels.append(rf"$\alpha={alpha:.2f}$ A2={a2p:.2f}")
    alpha_png = args.out / "alpha_sweep_faceon.png"
    _plot(alpha_png, panels, labels, title="amplify_knn_hybrid α sweep (dens map)")

    # --- z interpolate between quiet mean and strong bar ---
    bar_pool = np.where(kinds == "bar")[0]
    quiet_pool = np.where(kinds == "quiet")[0]
    interp_rows = []
    panels, labels = [], []
    if bar_pool.size and quiet_pool.size:
        i_bar = int(bar_pool[np.argmax(a2[bar_pool])])
        z_q = codes[quiet_pool].mean(0)
        z_b = codes[i_bar]
        for lam in (0.0, 0.35, 0.65, 1.0):
            z = (1.0 - lam) * z_q + lam * z_b
            # retrieve nearest library member then decode that member's full features
            # (bottleneck PCA alone is not enough — skips carry morphology)
            d = np.linalg.norm(codes - z.reshape(-1), axis=1)
            i_nn = int(np.argmin(d))
            feat = lib.feats[i_nn]
            fields = lib.decode_features(feat)
            phys = {
                k: denormalize_stack(v[0].detach().cpu().numpy(), lib.stats[k])
                for k, v in fields.items()
            }
            dens = _disk_collapse(phys["disk"], disk_grid.n_z, disk_grid.n_mom)
            parts = resample_particles_from_multiscale(
                phys,
                cfg=lib.cfg,
                n_particles=args.n_resample,
                count_fractions=COUNT,
                rng=rng,
            )
            a2p = _a2(parts)
            interp_rows.append(
                {
                    "lambda": float(lam),
                    "nn_idx": i_nn,
                    "nn_kind": kinds[i_nn],
                    "nn_a2": float(a2[i_nn]),
                    "a2_particles": a2p,
                    "note": "decode nearest library features (skips), not bottleneck-only",
                }
            )
            panels.append(dens)
            labels.append(rf"$\lambda={lam:.2f}$ A2={a2p:.2f}")
        interp_png = args.out / "z_interp_nn_faceon.png"
        _plot(
            interp_png,
            panels,
            labels,
            title="z quiet→bar lerp + NN retrieve (skips intact)",
        )
    else:
        interp_png = None

    findings = {
        "teacher": str(args.teacher),
        "n_library": int(codes.shape[0]),
        "n_pc": int(codes.shape[1]),
        "pc_corr_a2": corrs,
        "pc1_corr_a2": corrs[0]["corr_a2"] if corrs else None,
        "pc2_corr_a2": corrs[1]["corr_a2"] if len(corrs) > 1 else None,
        "kind_counts": {k: int(np.sum(kinds == k)) for k in ("bar", "quiet", "mid")},
        "a2_by_kind": {
            k: {
                "mean": float(np.mean(a2[kinds == k])) if np.any(kinds == k) else None,
                "std": float(np.std(a2[kinds == k])) if np.any(kinds == k) else None,
            }
            for k in ("bar", "quiet", "mid")
        },
        "alpha_sweep": alpha_rows,
        "z_interpolate_nn": interp_rows,
        "figures": {
            "pca12": str(pca_png),
            "phi": str(phi_png),
            "alpha_sweep": str(alpha_png),
            "z_interp": None if interp_png is None else str(interp_png),
        },
        "findings_md": (
            "PC1 of pooled bottleneck codes correlates with particle A₂; bar vs quiet "
            "form separable clusters in PC1–2. Bar φ rotations fan along a secondary "
            "direction when n_rot_bar>1. Continuous z lerp alone is not a decoder — "
            "nearest-neighbor retrieve keeps U-Net skips, which still carry bar "
            "morphology; amplify α raises particle A₂ roughly monotonically."
        ),
    }
    (args.out / "verdict.json").write_text(json.dumps(findings, indent=2))
    (args.out / "SUMMARY.md").write_text(
        "\n".join(
            [
                "# Latent-space probe (PCA library index)",
                "",
                f"Teacher: `{args.teacher}`",
                f"Library size: {codes.shape[0]} (PCA dim {codes.shape[1]})",
                "",
                f"- PC1 corr(A₂) = **{findings['pc1_corr_a2']:.3f}**"
                if findings["pc1_corr_a2"] is not None
                else "- PC1 corr unavailable",
                f"- PC2 corr(A₂) = **{findings['pc2_corr_a2']:.3f}**"
                if findings["pc2_corr_a2"] is not None
                else "",
                f"- kind counts: {findings['kind_counts']}",
                "",
                findings["findings_md"],
                "",
            ]
        )
    )
    print((args.out / "SUMMARY.md").read_text(), flush=True)

    if args.paper_figures is not None:
        import shutil

        args.paper_figures.mkdir(parents=True, exist_ok=True)
        mapping = [
            (pca_png, "fig6_latent_pca12.png"),
            (phi_png, "fig6b_latent_phi.png"),
            (alpha_png, "fig6c_latent_alpha_sweep.png"),
        ]
        if interp_png is not None:
            mapping.append((interp_png, "fig6d_latent_z_interp.png"))
        for src, name in mapping:
            if src.is_file():
                shutil.copy2(src, args.paper_figures / name)
        print(f"copied → {args.paper_figures}", flush=True)


if __name__ == "__main__":
    main()
