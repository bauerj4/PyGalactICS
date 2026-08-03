#!/usr/bin/env python3
"""Head-to-head multiseed: z_amplify vs disk_only_amplify vs amplify_knn_hybrid."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

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


def build_library(n_bar=36, n_quiet=20, n_mid=12, n_rot=6, bar_floor=0.25, quiet_ceil=0.05):
    ranked = json.loads(RANK.read_text())
    ranked.sort(key=lambda r: r["a2"], reverse=True)
    picks, seen = [], set()

    def add(row):
        if row["path"] not in seen:
            seen.add(row["path"])
            picks.append(row)

    for r in [x for x in ranked if x["a2"] >= bar_floor][:n_bar]:
        add(r)
    for r in reversed([x for x in ranked if x["a2"] <= quiet_ceil][-n_quiet:]):
        add(r)
    rest = [r for r in ranked if r["path"] not in seen]
    if rest and n_mid:
        for i in np.linspace(0, len(rest) - 1, num=min(n_mid, len(rest)), dtype=int):
            add(rest[int(i)])
    for r in ranked[:12]:
        add(r)
    teacher, cfg, stats = load_frozen_teacher_bundle(CRISP)
    feats, zs, a2s, meta = [], [], [], []
    print(f"encoding {len(picks)}…", flush=True)
    for j, row in enumerate(picks):
        is_bar = row["a2"] >= bar_floor
        phis = list(np.linspace(0, 2 * np.pi, n_rot, endpoint=False)) if is_bar else [None]
        for phi in phis:
            feat, z, a2 = encode_snapshot_features(
                row["path"],
                teacher=teacher,
                cfg=cfg,
                stats=stats,
                phi=None if phi is None else float(phi),
                enc_grid=4,
            )
            feats.append(feat)
            zs.append(z)
            a2s.append(a2)
            meta.append({})
        if (j + 1) % 8 == 0 or j + 1 == len(picks):
            print(f"  {j+1}/{len(picks)} lib={len(feats)}", flush=True)
    Z = np.stack(zs)
    A2 = np.asarray(a2s)
    Zc = Z - Z.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Zc, full_matrices=False)
    W = vt[: min(64, Z.shape[1], len(Z) - 1)].T
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
        lib_cfg=FeatureLibraryConfig(bar_floor=bar_floor, quiet_ceil=quiet_ceil),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("runs/ml/field_maps/marathon_head2head_2026-07-25"))
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--n-samples", type=int, default=6)
    p.add_argument("--seed", type=int, default=700)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lib = build_library()
    lib.save_codes(args.out / "feature_library_codes.npz")
    methods = {
        "z_amplify": lambda kind, rng: lib.sample_features_z_amplify(
            kind=kind, rng=rng, alpha_lo=1.2, alpha_hi=1.5
        ),
        "disk_only_amplify": lambda kind, rng: lib.sample_features(
            kind=kind, method="disk_only_amplify", rng=rng, alpha_lo=1.15, alpha_hi=1.45
        ),
        "amplify_knn_hybrid": lambda kind, rng: lib.sample_features(
            kind=kind,
            method="amplify_knn_hybrid",
            rng=rng,
            alpha_lo=1.12,
            alpha_hi=1.35,
            strong_floor=0.30,
        ),
    }
    verdict = {"methods": {}}
    for name, samp in methods.items():
        per = []
        for s in range(args.n_seeds):
            rng = np.random.default_rng(args.seed + 17 * s)
            bars, quiets = [], []
            for kind in ("barred", "quiet"):
                for _ in range(args.n_samples):
                    feat, _ = samp(kind, rng)
                    out = lib.decode_features(feat)
                    den = {k: denormalize_stack(out[k][0].numpy(), lib.stats[k]) for k in out}
                    parts = resample_particles_from_multiscale(
                        den, cfg=lib.cfg, n_particles=80_000, count_fractions=COUNT, rng=rng
                    )
                    a2 = _a2_disk(parts)
                    (bars if kind == "barred" else quiets).append(a2)
            per.append(
                {
                    "seed": int(args.seed + 17 * s),
                    "bar_mean": float(np.mean(bars)),
                    "quiet_mean": float(np.mean(quiets)),
                }
            )
            print(
                f"  {name} seed={per[-1]['seed']} bar={per[-1]['bar_mean']:.3f} quiet={per[-1]['quiet_mean']:.3f}",
                flush=True,
            )
        bm = float(np.mean([x["bar_mean"] for x in per]))
        bs = float(np.std([x["bar_mean"] for x in per]))
        qm = float(np.mean([x["quiet_mean"] for x in per]))
        verdict["methods"][name] = {
            "per_seed": per,
            "bar_mean": bm,
            "bar_std": bs,
            "quiet_mean": qm,
            "hits": bm >= 0.33 and qm <= 0.05,
        }
        print(f"→ {name}: {bm:.3f}±{bs:.3f} / {qm:.3f}", flush=True)
    (args.out / "verdict.json").write_text(json.dumps(verdict, indent=2))
    best = max(verdict["methods"].items(), key=lambda kv: (kv[1]["hits"], kv[1]["bar_mean"]))[0]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    with LOG.open("a") as f:
        for name, r in verdict["methods"].items():
            hits = "YES" if r["hits"] else "near"
            f.write(
                f"| {ts} | h2h {name} | {r['bar_mean']:.3f}±{r['bar_std']:.3f} | {r['quiet_mean']:.3f} | "
                f"{hits} | {args.out.name} |\n"
            )
        f.write(f"| {ts} | **H2H BEST→{best}** | — | — | update | {args.out.name} |\n")
    print("H2H best", best, flush=True)


if __name__ == "__main__":
    main()
