#!/usr/bin/env python3
"""Disk-only residual amplify: amplify disk tower only; keep bulge/halo from quiet/bar blend.

Hypothesis: bar morphology is disk-dominated; amplifying all towers can inject
spurious non-axisym into bulge/halo. Disk-only amplify may stabilize quiet and
keep bars strong.
"""

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


def amplify_disk_only(base: dict, target: dict, alpha: float) -> dict:
    out = {}
    for name in base:
        if name == "disk":
            bb, bt = base[name]["bottleneck"], target[name]["bottleneck"]
            sb, st = base[name]["skips"], target[name]["skips"]
            out[name] = {
                "bottleneck": bb + float(alpha) * (bt - bb),
                "skips": tuple(b + float(alpha) * (t - b) for b, t in zip(sb, st)),
            }
        else:
            # mild blend toward target for BH, not amplified
            beta = min(1.0, 0.35 * float(alpha))
            bb, bt = base[name]["bottleneck"], target[name]["bottleneck"]
            sb, st = base[name]["skips"], target[name]["skips"]
            out[name] = {
                "bottleneck": (1 - beta) * bb + beta * bt,
                "skips": tuple((1 - beta) * b + beta * t for b, t in zip(sb, st)),
            }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("runs/ml/field_maps/marathon_disk_only_amp_2026-07-25"))
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
    p.add_argument("--n-resample", type=int, default=80_000)
    p.add_argument("--seed", type=int, default=505)
    p.add_argument("--alpha-lo", type=float, default=1.15)
    p.add_argument("--alpha-hi", type=float, default=1.45)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    lib = build_library(args)
    lib.save_codes(args.out / "feature_library_codes.npz")
    disk_g = lib.cfg.grid_for("disk")

    def eval_feat(feat, meta, kind):
        out = lib.decode_features(feat)
        den = {k: denormalize_stack(out[k][0].numpy(), lib.stats[k]) for k in out}
        dens = _disk_collapse(den["disk"], disk_g.n_z, disk_g.n_mom)
        a2_map = float(
            dens_map_azimuthal_fourier_numpy(dens, m=2, r_max=12.0)["a_m_over_a0_median"]
        )
        parts = resample_particles_from_multiscale(
            den, cfg=lib.cfg, n_particles=args.n_resample, count_fractions=COUNT, rng=rng
        )
        return {"kind": kind, "a2_map": a2_map, "a2_part": _a2_disk(parts), **meta}

    verdict = {"runs": {}}
    for name, use_disk_only in (("full_amplify", False), ("disk_only_amplify", True)):
        rows = []
        for kind in ("barred", "quiet"):
            for j in range(args.n_samples):
                if kind == "quiet":
                    feat, meta = lib.sample_features(kind="quiet", method="uniform_knn", rng=rng)
                else:
                    pool = lib.bar_pool
                    w = np.maximum(lib.a2[pool], 1e-6) ** 3
                    w = w / w.sum()
                    i0 = int(rng.choice(pool, p=w))
                    alpha = float(rng.uniform(args.alpha_lo, args.alpha_hi))
                    if use_disk_only:
                        feat = amplify_disk_only(lib._quiet_mean, lib.feats[i0], alpha)
                        meta = {"method": "disk_only_amplify", "i0": i0, "alpha": alpha}
                    else:
                        feat, meta = lib.sample_features(
                            kind="barred",
                            method="amplify_residual",
                            rng=rng,
                            alpha_lo=args.alpha_lo,
                            alpha_hi=args.alpha_hi,
                        )
                row = eval_feat(feat, meta, kind)
                rows.append(row)
                print(f"  {name} {kind}#{j} A₂p={row['a2_part']:.3f}", flush=True)
        bar = float(np.mean([r["a2_part"] for r in rows if r["kind"] == "barred"]))
        qui = float(np.mean([r["a2_part"] for r in rows if r["kind"] == "quiet"]))
        verdict["runs"][name] = {
            "bar_mean": bar,
            "quiet_mean": qui,
            "hits_target": bar >= 0.33 and qui <= 0.05,
            "samples": rows,
        }
        print(f"→ {name}: bar={bar:.3f} quiet={qui:.3f}", flush=True)

    (args.out / "verdict.json").write_text(json.dumps(verdict, indent=2))
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    with LOG.open("a") as f:
        for name, r in verdict["runs"].items():
            hits = "YES" if r["hits_target"] else ("near" if r["bar_mean"] >= 0.30 else "no")
            f.write(
                f"| {ts} | {name} | {r['bar_mean']:.3f} | {r['quiet_mean']:.3f} | {hits} | {args.out.name} |\n"
            )
    print(f"done → {args.out}")


if __name__ == "__main__":
    main()
