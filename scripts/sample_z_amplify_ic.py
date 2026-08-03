#!/usr/bin/env python3
"""
Continuous-z amplify path: z~local-KDE → retrieve → amplify residual.

Closes the continuous-control gap (~0.17 global KDE) while preserving quiet.

    OMP_NUM_THREADS=6 python scripts/sample_z_amplify_ic.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
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


CRISP = Path("runs/ml/field_maps/crisp_2026-07-24/multitower_slice_ae.pt")
RANK = Path("runs/ml/field_maps/corpus_particle_a2_rank.json")
LOG = Path("runs/ml/field_maps/MARATHON_6H.md")
COUNT = {"disk": 4 / 7, "halo": 2 / 7, "bulge": 1 / 7}


def _a2_disk(parts):
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
    # force strongest bars
    for r in ranked[:12]:
        add(r)
    return picked


def build_library(args) -> TeacherFeatureLibrary:
    ranked = json.loads(Path(args.rank).read_text())
    ranked.sort(key=lambda r: r["a2"], reverse=True)
    picks = _stratified(
        ranked, args.n_bar, args.n_quiet, args.n_mid, args.bar_floor, args.quiet_ceil
    )
    teacher, cfg, stats = load_frozen_teacher_bundle(args.teacher)
    lib_cfg = FeatureLibraryConfig(
        enc_grid=args.enc_grid, bar_floor=args.bar_floor, quiet_ceil=args.quiet_ceil
    )
    feats, zs, a2s, meta = [], [], [], []
    print(f"encoding library snaps={len(picks)}…", flush=True)
    for j, row in enumerate(picks):
        is_bar = float(row["a2"]) >= args.bar_floor
        phis = (
            list(np.linspace(0.0, 2.0 * np.pi, args.n_rot_bar, endpoint=False))
            if is_bar and args.n_rot_bar > 1
            else [None]
        )
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
                    "kind": "bar" if is_bar else ("quiet" if row["a2"] <= args.quiet_ceil else "mid"),
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
    return TeacherFeatureLibrary(
        teacher=teacher,
        cfg=cfg,
        stats=stats,
        feats=feats,
        codes=Zc @ W,
        a2=A2,
        meta=meta,
        pca_mean=Z.mean(0),
        pca_w=W,
        lib_cfg=lib_cfg,
    )


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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("runs/ml/field_maps/marathon_z_amplify_2026-07-25"))
    p.add_argument("--teacher", type=Path, default=CRISP)
    p.add_argument("--rank", type=Path, default=RANK)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--n-bar", type=int, default=32)
    p.add_argument("--n-quiet", type=int, default=18)
    p.add_argument("--n-mid", type=int, default=12)
    p.add_argument("--n-rot-bar", type=int, default=5)
    p.add_argument("--bar-floor", type=float, default=0.22)
    p.add_argument("--quiet-ceil", type=float, default=0.05)
    p.add_argument("--enc-grid", type=int, default=4)
    p.add_argument("--n-pc", type=int, default=64)
    p.add_argument("--n-resample", type=int, default=80_000)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--alpha-lo", type=float, default=1.08)
    p.add_argument("--alpha-hi", type=float, default=1.32)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    lib = build_library(args)
    lib.save_codes(args.out / "feature_library_codes.npz")
    disk_g = lib.cfg.grid_for("disk")
    print(
        f"library n={len(lib.feats)} bar={lib.bar_pool.size} quiet={lib.quiet_pool.size}",
        flush=True,
    )

    # Baselines on same lib: amplify discrete + z_amplify continuous
    verdict = {"approach": "z_amplify continuous", "n_library": len(lib.feats), "runs": {}}
    panels, labels = [], []

    def samp_amplify(kind):
        return lib.sample_features(kind=kind, method="amplify_residual", rng=rng)

    def samp_z_amp(kind):
        return lib.sample_features_z_amplify(
            kind=kind, rng=rng, alpha_lo=args.alpha_lo, alpha_hi=args.alpha_hi
        )

    def samp_z_ret(kind):
        z = lib.sample_z_kde(kind=kind, rng=rng, jitter=0.28, global_mix=0.08)
        return lib.sample_features(kind=kind, z=z, rng=rng)

    for name, sampler in (
        ("amplify_residual", samp_amplify),
        ("z_amplify", samp_z_amp),
        ("z_retrieve_only", samp_z_ret),
    ):
        rows = []
        for kind in ("barred", "quiet"):
            for j in range(args.n_samples):
                feat, meta = sampler(kind)
                out = lib.decode_features(feat)
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
                rows.append(row)
                print(f"  {name} {kind}#{j} A₂p={a2_part:.3f}", flush=True)
                if name == "z_amplify" and kind == "barred" and j < 4:
                    panels.append(dens)
                    labels.append(f"z_amp\nA₂p={a2_part:.2f}")
        bar = float(np.mean([r["a2_part"] for r in rows if r["kind"] == "barred"]))
        qui = float(np.mean([r["a2_part"] for r in rows if r["kind"] == "quiet"]))
        verdict["runs"][name] = {
            "bar_mean": bar,
            "quiet_mean": qui,
            "hits_target": bar >= 0.33 and qui <= 0.05,
            "samples": rows,
        }
        print(f"  → {name}: bar={bar:.3f} quiet={qui:.3f}", flush=True)

    if panels:
        _plot(args.out / "bar_samples.png", panels, labels, "z_amplify barred")
    (args.out / "verdict.json").write_text(json.dumps(verdict, indent=2))

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    with LOG.open("a") as f:
        for name, r in verdict["runs"].items():
            hits = "YES" if r["hits_target"] else ("near" if r["bar_mean"] >= 0.30 and r["quiet_mean"] <= 0.05 else "no")
            f.write(
                f"| {ts} | {name} (z-amp script) | {r['bar_mean']:.3f} | {r['quiet_mean']:.3f} | "
                f"{hits} | {args.out.name} |\n"
            )
    best = max(verdict["runs"].items(), key=lambda kv: (kv[1]["hits_target"], kv[1]["bar_mean"]))
    Path("runs/ml/field_maps/LATEST").write_text(str(args.out.resolve()) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "samples"} for k, v in verdict["runs"].items()}, indent=2))
    print(f"best={best[0]} → {args.out}")


if __name__ == "__main__":
    main()
