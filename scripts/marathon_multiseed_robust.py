#!/usr/bin/env python3
"""Multi-seed robustness for amplify_residual and z_amplify (marathon)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

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
    for r in ranked[:15]:
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
            meta.append({"path": row["path"], "rank_a2": float(row["a2"]), "data_a2": float(a2)})
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


def eval_seed(lib, sampler, n_samples, n_resample, rng):
    disk_g = lib.cfg.grid_for("disk")
    bars, quiets = [], []
    for kind in ("barred", "quiet"):
        for _ in range(n_samples):
            feat, _meta = sampler(kind, rng)
            out = lib.decode_features(feat)
            den = {k: denormalize_stack(out[k][0].numpy(), lib.stats[k]) for k in out}
            dens = _disk_collapse(den["disk"], disk_g.n_z, disk_g.n_mom)
            parts = resample_particles_from_multiscale(
                den, cfg=lib.cfg, n_particles=n_resample, count_fractions=COUNT, rng=rng
            )
            a2 = _a2_disk(parts)
            (bars if kind == "barred" else quiets).append(a2)
    return float(np.mean(bars)), float(np.std(bars)), float(np.mean(quiets))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("runs/ml/field_maps/marathon_multiseed_robust_2026-07-25"))
    p.add_argument("--teacher", type=Path, default=CRISP)
    p.add_argument("--rank", type=Path, default=RANK)
    p.add_argument("--n-bar", type=int, default=36)
    p.add_argument("--n-quiet", type=int, default=20)
    p.add_argument("--n-mid", type=int, default=12)
    p.add_argument("--n-rot-bar", type=int, default=6)
    p.add_argument("--bar-floor", type=float, default=0.25)
    p.add_argument("--quiet-ceil", type=float, default=0.05)
    p.add_argument("--enc-grid", type=int, default=4)
    p.add_argument("--n-pc", type=int, default=64)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--n-seeds", type=int, default=6)
    p.add_argument("--n-resample", type=int, default=80_000)
    p.add_argument("--seed", type=int, default=101)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    lib = build_library(args)
    lib.save_codes(args.out / "feature_library_codes.npz")

    methods = {
        "amplify_hi": lambda kind, rng: lib.sample_features(
            kind=kind, method="amplify_residual", rng=rng, alpha_lo=1.15, alpha_hi=1.45
        ),
        "amplify_knn_hybrid": lambda kind, rng: lib.sample_features(
            kind=kind,
            method="amplify_knn_hybrid",
            rng=rng,
            alpha_lo=1.12,
            alpha_hi=1.35,
            knn_alpha_max=0.10,
            strong_floor=0.30,
        ),
        "z_amplify": lambda kind, rng: lib.sample_features_z_amplify(
            kind=kind, rng=rng, alpha_lo=1.20, alpha_hi=1.50
        ),
        "strong_knn": lambda kind, rng: lib.sample_features(
            kind=kind, method="strong_knn", rng=rng, strong_floor=0.30, alpha_max=0.12
        ),
    }

    verdict = {"n_library": len(lib.feats), "methods": {}}
    for name, samp in methods.items():
        per = []
        for s in range(args.n_seeds):
            rng = np.random.default_rng(args.seed + 17 * s)
            bm, bs, qm = eval_seed(lib, samp, args.n_samples, args.n_resample, rng)
            per.append({"seed": int(args.seed + 17 * s), "bar_mean": bm, "bar_std": bs, "quiet_mean": qm})
            print(f"  {name} seed={per[-1]['seed']} bar={bm:.3f}±{bs:.3f} quiet={qm:.3f}", flush=True)
        bar_means = [x["bar_mean"] for x in per]
        quiet_means = [x["quiet_mean"] for x in per]
        verdict["methods"][name] = {
            "per_seed": per,
            "bar_mean": float(np.mean(bar_means)),
            "bar_std_across_seeds": float(np.std(bar_means)),
            "quiet_mean": float(np.mean(quiet_means)),
            "frac_seeds_hit_0p33": float(np.mean([b >= 0.33 for b in bar_means])),
            "hits_target": float(np.mean(bar_means)) >= 0.33 and float(np.mean(quiet_means)) <= 0.05,
        }
        print(
            f"→ {name}: bar={verdict['methods'][name]['bar_mean']:.3f}±"
            f"{verdict['methods'][name]['bar_std_across_seeds']:.3f} "
            f"quiet={verdict['methods'][name]['quiet_mean']:.3f} "
            f"frac≥0.33={verdict['methods'][name]['frac_seeds_hit_0p33']:.2f}",
            flush=True,
        )

    (args.out / "verdict.json").write_text(json.dumps(verdict, indent=2))
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    best_name = max(
        verdict["methods"].items(),
        key=lambda kv: (
            kv[1]["hits_target"],
            kv[1]["frac_seeds_hit_0p33"],
            kv[1]["bar_mean"],
        ),
    )[0]
    with LOG.open("a") as f:
        for name, r in verdict["methods"].items():
            hits = "YES" if r["hits_target"] else ("near" if r["bar_mean"] >= 0.30 else "no")
            f.write(
                f"| {ts} | multiseed {name} | {r['bar_mean']:.3f}±{r['bar_std_across_seeds']:.3f} | "
                f"{r['quiet_mean']:.3f} | {hits} frac={r['frac_seeds_hit_0p33']:.2f} | {args.out.name} |\n"
            )
        f.write(f"| {ts} | **ROBUST BEST→{best_name}** | — | — | update | {args.out.name} |\n")
    Path("runs/ml/field_maps/LATEST").write_text(str(args.out.resolve()) + "\n")
    print(f"robust best={best_name} → {args.out}")


if __name__ == "__main__":
    main()
