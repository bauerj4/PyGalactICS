#!/usr/bin/env python3
"""
Sample non-eq ICs from a latent ``z`` (+ morphology kind) via teacher feature library.

Recommended recipe (beats v1 knn ~0.15; hits crisp AE bar A₂ ≈ 0.33):
  1. Rank corpus by particle A₂; build stratified library (strong bars + quiet)
  2. Rotate barred members; PCA-pool teacher bottlenecks → ``z``
  3. Sample with ``amplify_residual`` (default; quiet auto-gated) or ``strong_knn`` /
     continuous ``z`` via ``sample_features_z_amplify`` / ``--also-z-retrieve``

    OMP_NUM_THREADS=6 python scripts/sample_latent_ic.py \\
        --method amplify_residual --n-samples 6

Uses / writes artifacts under ``runs/ml/field_maps/creative_feature_library_v2_*``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from galacticsics.ml.fields.autoencoder import dens_map_azimuthal_fourier_numpy
from galacticsics.ml.fields.feature_library import (
    FeatureLibraryConfig,
    TeacherFeatureLibrary,
    encode_snapshot_features,
    load_frozen_teacher_bundle,
)
from galacticsics.ml.fields.normalize import denormalize_stack
from galacticsics.ml.fields.resample import resample_particles_from_multiscale
from ntropy.analysis.disk_density import disk_azimuthal_fourier


MANIFEST = Path("runs/ml/smoke_morton_vae_cpu/manifest_mw_morton_corpus_v2.json")
CRISP = Path("runs/ml/field_maps/crisp_2026-07-24/multitower_slice_ae.pt")
FFT_LONG = Path("runs/ml/field_maps/fft_morph_ft_long_2026-07-25/multitower_slice_ae.pt")
FFT_CONV = Path("runs/ml/field_maps/fft_morph_ft_converged_2026-07-25/multitower_slice_ae.pt")
RANK = Path("runs/ml/field_maps/corpus_particle_a2_rank.json")
COUNT = {"disk": 4 / 7, "halo": 2 / 7, "bulge": 1 / 7}


def default_teacher() -> Path:
    if FFT_LONG.is_file():
        return FFT_LONG
    if FFT_CONV.is_file():
        return FFT_CONV
    return CRISP


def _a2_disk(parts, r_eval: float | None = None):
    disk = parts["component_id"] == 0
    fout = disk_azimuthal_fourier(
        parts["pos"][disk],
        parts["mass"][disk],
        m=2,
        r_max=12.0,
        n_bins=12,
        z_max=0.5,
        min_count=10,
        r_eval=r_eval,
    )
    if r_eval is not None:
        return float(fout["a_m_over_a0_at_r"])
    return float(fout["a_m_over_a0_median"])


def _disk_collapse(stack, n_z, n_mom):
    dens = np.zeros(stack.shape[-2:], dtype=np.float64)
    for iz in range(n_z):
        dens += np.maximum(stack[iz * n_mom], 0.0)
    return dens


def _stratified(ranked, n_bar, n_quiet, n_mid, bar_floor, quiet_ceil):
    seen, picked = set(), []

    def add(row):
        if row["path"] in seen:
            return
        seen.add(row["path"])
        picked.append(row)

    bars = [r for r in ranked if r["a2"] >= bar_floor]
    quiets = [r for r in ranked if r["a2"] <= quiet_ceil]
    for r in bars[:n_bar]:
        add(r)
    for r in reversed(quiets[-n_quiet:]):
        add(r)
    rest = [r for r in ranked if r["path"] not in seen]
    if rest and n_mid > 0:
        idx = np.linspace(0, len(rest) - 1, num=min(n_mid, len(rest)), dtype=int)
        for i in idx:
            add(rest[int(i)])
    return picked


def _plot(path, panels, labels, title):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    if n == 1:
        axes = [axes]
    for ax, img, lab in zip(axes, panels, labels):
        pos = img[img > 0]
        vmax = float(np.percentile(pos, 99)) if pos.size else 1.0
        ax.imshow(
            np.log1p(np.maximum(img, 0)),
            origin="lower",
            cmap="inferno",
            vmin=0,
            vmax=np.log1p(vmax),
        )
        ax.set_title(lab, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def build_library(args) -> TeacherFeatureLibrary:
    ranked = json.loads(Path(args.rank).read_text())
    ranked.sort(key=lambda r: r["a2"], reverse=True)
    picks = _stratified(
        ranked,
        args.n_bar,
        args.n_quiet,
        args.n_mid,
        args.bar_floor,
        args.quiet_ceil,
    )
    teacher, cfg, stats = load_frozen_teacher_bundle(args.teacher)
    lib_cfg = FeatureLibraryConfig(
        enc_grid=args.enc_grid, bar_floor=args.bar_floor, quiet_ceil=args.quiet_ceil
    )
    feats, zs, a2s, meta = [], [], [], []
    print(f"encoding library snaps={len(picks)}…", flush=True)
    for j, row in enumerate(picks):
        is_bar = float(row["a2"]) >= args.bar_floor
        phis: list[float | None]
        if is_bar and args.n_rot_bar > 1:
            phis = list(np.linspace(0.0, 2.0 * np.pi, args.n_rot_bar, endpoint=False))
        else:
            phis = [None]
        for phi in phis:
            feat, z, a2 = encode_snapshot_features(
                row["path"],
                teacher=teacher,
                cfg=cfg,
                stats=stats,
                phi=None if phi is None else float(phi),
                enc_grid=args.enc_grid,
            )
            feats.append(feat)
            zs.append(z)
            a2s.append(a2)
            meta.append(
                {
                    "path": row["path"],
                    "rank_a2": float(row["a2"]),
                    "data_a2": float(a2),
                    "phi": None if phi is None else float(phi),
                    "kind": (
                        "bar"
                        if is_bar
                        else ("quiet" if row["a2"] <= args.quiet_ceil else "mid")
                    ),
                }
            )
        if (j + 1) % 8 == 0 or j + 1 == len(picks):
            print(f"  {j+1}/{len(picks)} lib={len(feats)}", flush=True)
    Z = np.stack(zs, axis=0)
    A2 = np.asarray(a2s, dtype=np.float64)
    Zc = Z - Z.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Zc, full_matrices=False)
    n_pc = min(args.n_pc, Z.shape[1], max(Z.shape[0] - 1, 1))
    W = vt[:n_pc].T
    codes = Zc @ W
    return TeacherFeatureLibrary(
        teacher=teacher,
        cfg=cfg,
        stats=stats,
        feats=feats,
        codes=codes,
        a2=A2,
        meta=meta,
        pca_mean=Z.mean(0),
        pca_w=W,
        lib_cfg=lib_cfg,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("runs/ml/field_maps/creative_latent_ic_2026-07-25"),
    )
    p.add_argument("--teacher", type=Path, default=None)
    p.add_argument("--rank", type=Path, default=RANK)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--method", type=str, default="amplify_knn_hybrid")
    p.add_argument("--n-samples", type=int, default=6)
    p.add_argument("--n-bar", type=int, default=24)
    p.add_argument("--n-quiet", type=int, default=16)
    p.add_argument("--n-mid", type=int, default=12)
    p.add_argument("--n-rot-bar", type=int, default=4)
    p.add_argument("--bar-floor", type=float, default=0.20)
    p.add_argument("--quiet-ceil", type=float, default=0.06)
    p.add_argument("--enc-grid", type=int, default=4)
    p.add_argument("--n-pc", type=int, default=64)
    p.add_argument("--n-resample", type=int, default=80_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--also-z-retrieve",
        action="store_true",
        help="Also sample continuous z~local-KDE then retrieve",
    )
    p.add_argument(
        "--also-z-amplify",
        action="store_true",
        help="Also sample continuous z → strong retrieve → amplify residual",
    )
    p.add_argument("--z-alpha-lo", type=float, default=1.20)
    p.add_argument("--z-alpha-hi", type=float, default=1.50)
    args = p.parse_args()
    if args.teacher is None:
        args.teacher = default_teacher()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.rank.is_file():
        raise SystemExit(f"missing A₂ rank file {args.rank}; run sample_feature_library_v2 first")

    lib = build_library(args)
    lib.save_codes(args.out / "feature_library_codes.npz")
    disk_g = lib.cfg.grid_for("disk")
    print(
        f"library n={len(lib.feats)} bar_pool={lib.bar_pool.size} "
        f"quiet_pool={lib.quiet_pool.size} A₂∈[{lib.a2.min():.3f},{lib.a2.max():.3f}]",
        flush=True,
    )

    verdict = {
        "approach": "TeacherFeatureLibrary latent IC sampling",
        "method": args.method,
        "n_library": len(lib.feats),
        "latent_dim": lib.latent_dim,
        "samples": [],
        "z_retrieve": [],
        "z_amplify": [],
    }
    panels, labels = [], []
    for kind in ("barred", "quiet"):
        for j in range(args.n_samples):
            out, meta = lib.sample_fields(kind=kind, method=args.method, rng=rng)
            den = {k: denormalize_stack(out[k][0].numpy(), lib.stats[k]) for k in out}
            dens = _disk_collapse(den["disk"], disk_g.n_z, disk_g.n_mom)
            a2_map = float(
                dens_map_azimuthal_fourier_numpy(dens, m=2, r_max=12.0)["a_m_over_a0_median"]
            )
            parts = resample_particles_from_multiscale(
                den, cfg=lib.cfg, n_particles=args.n_resample, count_fractions=COUNT, rng=rng
            )
            a2_part = _a2_disk(parts)
            row = {"kind": kind, "a2_map": a2_map, "a2_part": a2_part, **meta}
            verdict["samples"].append(row)
            print(f"  {args.method} {kind}#{j} A₂p={a2_part:.3f}", flush=True)
            if kind == "barred" and j < 4:
                panels.append(dens)
                labels.append(f"{kind}\nA₂p={a2_part:.2f}")

    if args.also_z_retrieve:
        for kind in ("barred", "quiet"):
            for j in range(args.n_samples):
                z = lib.sample_z_kde(kind=kind, rng=rng)
                out, meta = lib.sample_fields(kind=kind, method=args.method, z=z, rng=rng)
                den = {k: denormalize_stack(out[k][0].numpy(), lib.stats[k]) for k in out}
                dens = _disk_collapse(den["disk"], disk_g.n_z, disk_g.n_mom)
                a2_map = float(
                    dens_map_azimuthal_fourier_numpy(dens, m=2, r_max=12.0)[
                        "a_m_over_a0_median"
                    ]
                )
                parts = resample_particles_from_multiscale(
                    den,
                    cfg=lib.cfg,
                    n_particles=args.n_resample,
                    count_fractions=COUNT,
                    rng=rng,
                )
                a2_part = _a2_disk(parts)
                verdict["z_retrieve"].append(
                    {"kind": kind, "a2_map": a2_map, "a2_part": a2_part, **meta}
                )
                print(f"  z-retrieve {kind}#{j} A₂p={a2_part:.3f}", flush=True)

    if args.also_z_amplify:
        for kind in ("barred", "quiet"):
            for j in range(args.n_samples):
                feat, meta = lib.sample_features_z_amplify(
                    kind=kind,
                    rng=rng,
                    alpha_lo=args.z_alpha_lo,
                    alpha_hi=args.z_alpha_hi,
                )
                out = lib.decode_features(feat)
                den = {k: denormalize_stack(out[k][0].numpy(), lib.stats[k]) for k in out}
                dens = _disk_collapse(den["disk"], disk_g.n_z, disk_g.n_mom)
                a2_map = float(
                    dens_map_azimuthal_fourier_numpy(dens, m=2, r_max=12.0)[
                        "a_m_over_a0_median"
                    ]
                )
                parts = resample_particles_from_multiscale(
                    den,
                    cfg=lib.cfg,
                    n_particles=args.n_resample,
                    count_fractions=COUNT,
                    rng=rng,
                )
                a2_part = _a2_disk(parts)
                verdict["z_amplify"].append(
                    {"kind": kind, "a2_map": a2_map, "a2_part": a2_part, **meta}
                )
                print(f"  z-amplify {kind}#{j} A₂p={a2_part:.3f}", flush=True)

    def _means(rows, kind):
        xs = [r["a2_part"] for r in rows if r["kind"] == kind]
        return float(np.mean(xs)) if xs else None

    verdict["bar_mean"] = _means(verdict["samples"], "barred")
    verdict["quiet_mean"] = _means(verdict["samples"], "quiet")
    if verdict["z_retrieve"]:
        verdict["z_bar_mean"] = _means(verdict["z_retrieve"], "barred")
        verdict["z_quiet_mean"] = _means(verdict["z_retrieve"], "quiet")
    if verdict["z_amplify"]:
        verdict["z_amp_bar_mean"] = _means(verdict["z_amplify"], "barred")
        verdict["z_amp_quiet_mean"] = _means(verdict["z_amplify"], "quiet")
    verdict["beats_v1_0p15"] = bool(
        verdict["bar_mean"] is not None
        and verdict["bar_mean"] > 0.15
        and verdict["quiet_mean"] is not None
        and verdict["quiet_mean"] < 0.08
    )
    verdict["hits_target_0p25"] = bool(
        verdict["bar_mean"] is not None
        and verdict["bar_mean"] >= 0.25
        and verdict["quiet_mean"] is not None
        and verdict["quiet_mean"] <= 0.05
    )
    verdict["hits_crisp_0p33"] = bool(
        verdict["bar_mean"] is not None
        and verdict["bar_mean"] >= 0.33
        and verdict["quiet_mean"] is not None
        and verdict["quiet_mean"] <= 0.05
    )
    if panels:
        _plot(args.out / "bar_samples.png", panels, labels, f"{args.method} barred samples")
    (args.out / "verdict.json").write_text(json.dumps(verdict, indent=2))
    keys = (
        "bar_mean",
        "quiet_mean",
        "z_bar_mean",
        "z_quiet_mean",
        "z_amp_bar_mean",
        "z_amp_quiet_mean",
        "hits_crisp_0p33",
        "hits_target_0p25",
        "beats_v1_0p15",
    )
    print(json.dumps({k: verdict[k] for k in keys if k in verdict}, indent=2))
    if verdict["hits_crisp_0p33"] or verdict["hits_target_0p25"] or verdict["beats_v1_0p15"]:
        Path("runs/ml/field_maps/LATEST").write_text(str(args.out.resolve()) + "\n")
    print(f"done → {args.out}")


if __name__ == "__main__":
    main()
