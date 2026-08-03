#!/usr/bin/env python3
"""
6h generative marathon: iterate teacher-feature-library hybrids vs crisp AE.

Logs every attempt to runs/ml/field_maps/MARATHON_6H.md.
Preserves frozen crisp AE decode; no end-to-end VAE-from-z.

    OMP_NUM_THREADS=6 python scripts/marathon_field_generative.py --hours 6
"""

from __future__ import annotations

import argparse
import json
import time
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
    scale_residual,
)
from galacticsics.ml.fields.normalize import denormalize_stack
from galacticsics.ml.fields.resample import resample_particles_from_multiscale
from ntropy.analysis.disk_density import disk_azimuthal_fourier


MANIFEST = Path("runs/ml/smoke_morton_vae_cpu/manifest_mw_morton_corpus_v2.json")
CRISP = Path("runs/ml/field_maps/crisp_2026-07-24/multitower_slice_ae.pt")
RANK = Path("runs/ml/field_maps/corpus_particle_a2_rank.json")
LOG = Path("runs/ml/field_maps/MARATHON_6H.md")
COUNT = {"disk": 4 / 7, "halo": 2 / 7, "bulge": 1 / 7}
TARGET_BAR = 0.33
TARGET_QUIET = 0.05


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
    return picked


def build_library(
    *,
    teacher,
    cfg,
    stats,
    ranked,
    n_bar,
    n_quiet,
    n_mid,
    n_rot_bar,
    bar_floor,
    quiet_ceil,
    enc_grid,
    n_pc,
    force_top_n: int = 0,
) -> TeacherFeatureLibrary:
    picks = _stratified(ranked, n_bar, n_quiet, n_mid, bar_floor, quiet_ceil)
    if force_top_n > 0:
        seen = {r["path"] for r in picks}
        for r in ranked[:force_top_n]:
            if r["path"] not in seen:
                picks.append(r)
                seen.add(r["path"])
    lib_cfg = FeatureLibraryConfig(
        enc_grid=enc_grid, bar_floor=bar_floor, quiet_ceil=quiet_ceil
    )
    feats, zs, a2s, meta = [], [], [], []
    print(f"encoding library snaps={len(picks)}…", flush=True)
    for j, row in enumerate(picks):
        is_bar = float(row["a2"]) >= bar_floor
        if is_bar and n_rot_bar > 1:
            phis = list(np.linspace(0.0, 2.0 * np.pi, n_rot_bar, endpoint=False))
        else:
            phis = [None]
        for phi in phis:
            feat, z, a2 = encode_snapshot_features(
                row["path"],
                teacher=teacher,
                cfg=cfg,
                stats=stats,
                phi=None if phi is None else float(phi),
                enc_grid=enc_grid,
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
                        else ("quiet" if row["a2"] <= quiet_ceil else "mid")
                    ),
                }
            )
        if (j + 1) % 8 == 0 or j + 1 == len(picks):
            print(f"  {j+1}/{len(picks)} lib={len(feats)}", flush=True)
    Z = np.stack(zs, axis=0)
    A2 = np.asarray(a2s, dtype=np.float64)
    Zc = Z - Z.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Zc, full_matrices=False)
    n_keep = min(n_pc, Z.shape[1], max(Z.shape[0] - 1, 1))
    W = vt[:n_keep].T
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


def eval_method(
    lib: TeacherFeatureLibrary,
    *,
    method: str,
    n_samples: int,
    n_resample: int,
    rng: np.random.Generator,
    also_z: bool = False,
    z_jitter: float = 0.35,
    method_kwargs: dict | None = None,
    custom_bar_sampler=None,
) -> dict:
    method_kwargs = method_kwargs or {}
    disk_g = lib.cfg.grid_for("disk")
    samples = []
    z_rows = []

    def _one(kind, feat_meta_fn):
        feat, meta = feat_meta_fn()
        out = lib.decode_features(feat)
        den = {k: denormalize_stack(out[k][0].numpy(), lib.stats[k]) for k in out}
        dens = _disk_collapse(den["disk"], disk_g.n_z, disk_g.n_mom)
        a2_map = float(
            dens_map_azimuthal_fourier_numpy(dens, m=2, r_max=12.0)["a_m_over_a0_median"]
        )
        parts = resample_particles_from_multiscale(
            den, cfg=lib.cfg, n_particles=n_resample, count_fractions=COUNT, rng=rng
        )
        a2_part = _a2_disk(parts)
        return {
            "kind": kind,
            "a2_map": a2_map,
            "a2_part": a2_part,
            **meta,
        }, dens

    panels, labels = [], []
    for kind in ("barred", "quiet"):
        for j in range(n_samples):
            if kind == "barred" and custom_bar_sampler is not None:
                row, dens = _one(kind, lambda: custom_bar_sampler(lib, rng))
            else:
                row, dens = _one(
                    kind,
                    lambda k=kind: lib.sample_features(
                        kind=k, method=method, rng=rng, **method_kwargs
                    ),
                )
            samples.append(row)
            print(f"  {method} {kind}#{j} A₂p={row['a2_part']:.3f}", flush=True)
            if kind == "barred" and j < 4:
                panels.append(dens)
                labels.append(f"{kind}\nA₂p={row['a2_part']:.2f}")

    if also_z:
        for kind in ("barred", "quiet"):
            for j in range(n_samples):
                z = lib.sample_z_kde(kind=kind, rng=rng, jitter=z_jitter)
                # Do not forward method-specific kwargs into z-retrieve.
                row, _ = _one(
                    kind,
                    lambda k=kind, zz=z: lib.sample_features(
                        kind=k, method="uniform_knn", z=zz, rng=rng
                    ),
                )
                z_rows.append(row)
                print(f"  z-{method} {kind}#{j} A₂p={row['a2_part']:.3f}", flush=True)

    def means(rows, kind):
        xs = [r["a2_part"] for r in rows if r["kind"] == kind]
        return float(np.mean(xs)) if xs else None

    verdict = {
        "method": method,
        "n_library": len(lib.feats),
        "samples": samples,
        "z_retrieve": z_rows,
        "bar_mean": means(samples, "barred"),
        "quiet_mean": means(samples, "quiet"),
        "method_kwargs": method_kwargs,
    }
    if z_rows:
        verdict["z_bar_mean"] = means(z_rows, "barred")
        verdict["z_quiet_mean"] = means(z_rows, "quiet")
    verdict["hits_target"] = bool(
        verdict["bar_mean"] is not None
        and verdict["bar_mean"] >= TARGET_BAR
        and verdict["quiet_mean"] is not None
        and verdict["quiet_mean"] <= TARGET_QUIET
    )
    verdict["near_target"] = bool(
        verdict["bar_mean"] is not None
        and verdict["bar_mean"] >= 0.30
        and verdict["quiet_mean"] is not None
        and verdict["quiet_mean"] <= TARGET_QUIET
    )
    return verdict, panels, labels


def opt_alpha_amplify(lib, rng, alpha_lo=0.7, alpha_hi=1.55, n_grid=9):
    """Pick α so decoded map A₂ is near TARGET_BAR (teacher decode, no particles)."""
    pool = lib.bar_pool
    w = np.maximum(lib.a2[pool], 1e-6) ** 3
    w = w / w.sum()
    i0 = int(rng.choice(pool, p=w))
    base = lib._quiet_mean if lib._quiet_mean is not None else lib.feats[int(pool[0])]
    target = lib.feats[i0]
    disk_g = lib.cfg.grid_for("disk")
    best_a, best_err, best_feat = 1.0, 1e9, target
    for a in np.linspace(alpha_lo, alpha_hi, n_grid):
        feat = scale_residual(base, target, float(a))
        out = lib.decode_features(feat)
        den = denormalize_stack(out["disk"][0].numpy(), lib.stats["disk"])
        dens = _disk_collapse(den, disk_g.n_z, disk_g.n_mom)
        a2_map = float(
            dens_map_azimuthal_fourier_numpy(dens, m=2, r_max=12.0)["a_m_over_a0_median"]
        )
        err = abs(a2_map - TARGET_BAR)
        if err < best_err:
            best_err, best_a, best_feat = err, float(a), feat
    return best_feat, {"method": "opt_alpha_amplify", "i0": i0, "alpha": best_a, "err": best_err}


def log_verdict(out_dir: Path, name: str, verdict: dict, note: str = "") -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    bar = verdict.get("bar_mean")
    qui = verdict.get("quiet_mean")
    zbar = verdict.get("z_bar_mean")
    hits = "YES" if verdict.get("hits_target") else ("near" if verdict.get("near_target") else "no")
    znote = f" zbar={zbar:.3f}" if zbar is not None else ""
    line = (
        f"| {ts} | {name} | {bar:.3f} | {qui:.3f} | {hits}{znote} | "
        f"{out_dir.name} {note}|\n"
    )
    with LOG.open("a") as f:
        f.write(line)
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps({k: verdict[k] for k in ("bar_mean", "quiet_mean", "z_bar_mean", "z_quiet_mean", "hits_target", "near_target") if k in verdict}, indent=2), flush=True)


def _plot(path, panels, labels, title):
    if not panels:
        return
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


def experiment_configs(wave: int) -> list[dict]:
    """Return library+method configs for a marathon wave."""
    base = dict(
        n_quiet=16,
        quiet_ceil=0.05,
        enc_grid=4,
        n_pc=64,
        n_samples=6,
        n_resample=80_000,
    )
    waves = [
        # Wave 0: new methods on denser library
        [
            {
                **base,
                "tag": "dense_strong_knn",
                "n_bar": 36,
                "n_mid": 16,
                "n_rot_bar": 5,
                "bar_floor": 0.20,
                "force_top_n": 12,
                "method": "strong_knn",
                "also_z": True,
                "z_jitter": 0.30,
            },
            {
                **base,
                "tag": "dense_amplify_knn",
                "n_bar": 36,
                "n_mid": 16,
                "n_rot_bar": 5,
                "bar_floor": 0.20,
                "force_top_n": 12,
                "method": "amplify_knn_hybrid",
                "also_z": False,
            },
            {
                **base,
                "tag": "dense_topk_exact",
                "n_bar": 36,
                "n_mid": 12,
                "n_rot_bar": 4,
                "bar_floor": 0.22,
                "force_top_n": 15,
                "method": "topk_exact",
                "also_z": False,
            },
            {
                **base,
                "tag": "dense_amplify_mild",
                "n_bar": 36,
                "n_mid": 16,
                "n_rot_bar": 5,
                "bar_floor": 0.20,
                "force_top_n": 12,
                "method": "amplify_residual",
                "method_kwargs": {"alpha_lo": 0.95, "alpha_hi": 1.20},
                "also_z": False,
            },
            {
                **base,
                "tag": "opt_alpha_amp",
                "n_bar": 36,
                "n_mid": 16,
                "n_rot_bar": 5,
                "bar_floor": 0.20,
                "force_top_n": 12,
                "method": "uniform_knn",  # quiet path
                "custom_bar": "opt_alpha_amplify",
                "also_z": False,
            },
        ],
        # Wave 1: push continuous z + tighter bar bias
        [
            {
                **base,
                "tag": "ultra_strong_knn",
                "n_bar": 40,
                "n_mid": 10,
                "n_rot_bar": 6,
                "bar_floor": 0.25,
                "force_top_n": 15,
                "method": "strong_knn",
                "method_kwargs": {"strong_floor": 0.30, "alpha_max": 0.15, "a2_power": 4.0},
                "also_z": True,
                "z_jitter": 0.25,
            },
            {
                **base,
                "tag": "uniform_knn_dense_zjit",
                "n_bar": 40,
                "n_mid": 20,
                "n_rot_bar": 6,
                "bar_floor": 0.20,
                "force_top_n": 15,
                "method": "uniform_knn",
                "also_z": True,
                "z_jitter": 0.22,
            },
            {
                **base,
                "tag": "amplify_hi_gated",
                "n_bar": 40,
                "n_mid": 12,
                "n_rot_bar": 6,
                "bar_floor": 0.22,
                "force_top_n": 15,
                "method": "amplify_residual",
                "method_kwargs": {"alpha_lo": 1.10, "alpha_hi": 1.40},
                "also_z": False,
            },
            {
                **base,
                "tag": "a2_weighted_dense",
                "n_bar": 40,
                "n_mid": 16,
                "n_rot_bar": 5,
                "bar_floor": 0.20,
                "force_top_n": 12,
                "method": "a2_weighted_knn",
                "method_kwargs": {"k": 4, "temp": 5.0, "a2_power": 4.0},
                "also_z": True,
                "z_jitter": 0.28,
            },
        ],
        # Wave 2: quieter ceil + more quiets; hybrid refine
        [
            {
                **base,
                "tag": "quiet_strict_strong",
                "n_bar": 36,
                "n_quiet": 24,
                "n_mid": 12,
                "n_rot_bar": 5,
                "bar_floor": 0.22,
                "quiet_ceil": 0.04,
                "force_top_n": 12,
                "method": "strong_knn",
                "also_z": True,
                "z_jitter": 0.28,
            },
            {
                **base,
                "tag": "hybrid_amp_1p15",
                "n_bar": 36,
                "n_quiet": 20,
                "n_mid": 12,
                "n_rot_bar": 5,
                "bar_floor": 0.22,
                "quiet_ceil": 0.045,
                "force_top_n": 12,
                "method": "amplify_knn_hybrid",
                "method_kwargs": {
                    "alpha_lo": 1.08,
                    "alpha_hi": 1.22,
                    "knn_alpha_max": 0.12,
                    "strong_floor": 0.28,
                },
                "also_z": False,
            },
            {
                **base,
                "tag": "topk_exact_strict",
                "n_bar": 30,
                "n_quiet": 24,
                "n_mid": 8,
                "n_rot_bar": 4,
                "bar_floor": 0.28,
                "quiet_ceil": 0.04,
                "force_top_n": 15,
                "method": "topk_exact",
                "method_kwargs": {"top_frac": 0.5, "a2_power": 5.0},
                "also_z": False,
            },
            {
                **base,
                "tag": "opt_alpha_v2",
                "n_bar": 40,
                "n_quiet": 20,
                "n_mid": 12,
                "n_rot_bar": 6,
                "bar_floor": 0.22,
                "quiet_ceil": 0.045,
                "force_top_n": 15,
                "method": "uniform_knn",
                "custom_bar": "opt_alpha_amplify",
                "also_z": False,
            },
        ],
        # Wave 3: amplify variants + strong continuous control
        [
            {
                **base,
                "tag": "amplify_sweet",
                "n_bar": 40,
                "n_quiet": 22,
                "n_mid": 14,
                "n_rot_bar": 6,
                "bar_floor": 0.22,
                "quiet_ceil": 0.045,
                "force_top_n": 15,
                "method": "amplify_residual",
                "method_kwargs": {"alpha_lo": 1.12, "alpha_hi": 1.28},
                "also_z": False,
                "n_samples": 10,
            },
            {
                **base,
                "tag": "amplify_knn_sweet",
                "n_bar": 40,
                "n_quiet": 22,
                "n_mid": 14,
                "n_rot_bar": 6,
                "bar_floor": 0.22,
                "quiet_ceil": 0.045,
                "force_top_n": 15,
                "method": "amplify_knn_hybrid",
                "method_kwargs": {
                    "alpha_lo": 1.10,
                    "alpha_hi": 1.30,
                    "knn_alpha_max": 0.10,
                    "strong_floor": 0.30,
                },
                "also_z": False,
                "n_samples": 10,
            },
            {
                **base,
                "tag": "strong_knn_rot8",
                "n_bar": 36,
                "n_quiet": 20,
                "n_mid": 10,
                "n_rot_bar": 8,
                "bar_floor": 0.25,
                "quiet_ceil": 0.045,
                "force_top_n": 15,
                "method": "strong_knn",
                "method_kwargs": {"strong_floor": 0.30, "alpha_max": 0.12, "a2_power": 4.0},
                "also_z": True,
                "z_jitter": 0.20,
                "n_samples": 8,
            },
            {
                **base,
                "tag": "topk_plus_amp",
                "n_bar": 36,
                "n_quiet": 20,
                "n_mid": 10,
                "n_rot_bar": 5,
                "bar_floor": 0.25,
                "quiet_ceil": 0.045,
                "force_top_n": 15,
                "method": "amplify_residual",
                "method_kwargs": {"alpha_lo": 1.05, "alpha_hi": 1.20},
                "also_z": False,
                "n_samples": 8,
            },
            {
                **base,
                "tag": "hier_morph_gated",
                "n_bar": 36,
                "n_quiet": 20,
                "n_mid": 12,
                "n_rot_bar": 5,
                "bar_floor": 0.22,
                "quiet_ceil": 0.045,
                "force_top_n": 12,
                "method": "hier_morph",
                "method_kwargs": {"alpha_lo": 1.05, "alpha_hi": 1.35},
                "also_z": False,
                "n_samples": 8,
            },
        ],
    ]
    return waves[wave % len(waves)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hours", type=float, default=6.0)
    p.add_argument("--teacher", type=Path, default=CRISP)
    p.add_argument("--rank", type=Path, default=RANK)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--start-wave", type=int, default=0)
    p.add_argument("--out-root", type=Path, default=Path("runs/ml/field_maps"))
    p.add_argument("--omp-threads", type=int, default=6)
    args = p.parse_args()

    t0 = time.time()
    deadline = t0 + args.hours * 3600.0
    torch.set_num_threads(max(1, args.omp_threads))
    ranked = json.loads(args.rank.read_text())
    ranked.sort(key=lambda r: r["a2"], reverse=True)
    teacher, cfg, stats = load_frozen_teacher_bundle(args.teacher)

    wave = args.start_wave
    run_i = 0
    best = {"bar_mean": -1.0, "quiet_mean": 99.0, "tag": None, "path": None}

    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(
            f"\n### Marathon runner started {datetime.now(timezone.utc).isoformat()} "
            f"hours={args.hours}\n\n"
        )

    while time.time() < deadline:
        configs = experiment_configs(wave)
        # Rebuild densest library once per wave (reuse across methods with same lib params)
        lib_cache: dict[tuple, TeacherFeatureLibrary] = {}
        for cfg_e in configs:
            if time.time() >= deadline:
                break
            run_i += 1
            tag = cfg_e["tag"]
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out = args.out_root / f"marathon_{tag}_{stamp}"
            out.mkdir(parents=True, exist_ok=True)
            print(f"\n=== RUN {run_i} wave={wave} {tag} → {out} ===", flush=True)

            lib_key = (
                cfg_e["n_bar"],
                cfg_e["n_quiet"],
                cfg_e["n_mid"],
                cfg_e["n_rot_bar"],
                cfg_e["bar_floor"],
                cfg_e["quiet_ceil"],
                cfg_e.get("force_top_n", 0),
                cfg_e["enc_grid"],
                cfg_e["n_pc"],
            )
            if lib_key not in lib_cache:
                lib_cache[lib_key] = build_library(
                    teacher=teacher,
                    cfg=cfg,
                    stats=stats,
                    ranked=ranked,
                    n_bar=cfg_e["n_bar"],
                    n_quiet=cfg_e["n_quiet"],
                    n_mid=cfg_e["n_mid"],
                    n_rot_bar=cfg_e["n_rot_bar"],
                    bar_floor=cfg_e["bar_floor"],
                    quiet_ceil=cfg_e["quiet_ceil"],
                    enc_grid=cfg_e["enc_grid"],
                    n_pc=cfg_e["n_pc"],
                    force_top_n=int(cfg_e.get("force_top_n", 0)),
                )
            lib = lib_cache[lib_key]
            lib.save_codes(out / "feature_library_codes.npz")

            rng = np.random.default_rng(args.seed + run_i * 17)
            custom = None
            if cfg_e.get("custom_bar") == "opt_alpha_amplify":
                custom = opt_alpha_amplify

            verdict, panels, labels = eval_method(
                lib,
                method=cfg_e["method"],
                n_samples=cfg_e["n_samples"],
                n_resample=cfg_e["n_resample"],
                rng=rng,
                also_z=bool(cfg_e.get("also_z", False)),
                z_jitter=float(cfg_e.get("z_jitter", 0.35)),
                method_kwargs=cfg_e.get("method_kwargs") or {},
                custom_bar_sampler=custom,
            )
            verdict["tag"] = tag
            verdict["wave"] = wave
            verdict["lib_n"] = len(lib.feats)
            _plot(out / "bar_samples.png", panels, labels, f"{tag}")
            log_verdict(out, tag, verdict)

            # Track best: prioritize hits, then near, then bar with quiet OK
            score = (
                (10.0 if verdict.get("hits_target") else 0.0)
                + (3.0 if verdict.get("near_target") else 0.0)
                + float(verdict["bar_mean"] or 0.0)
                - 2.0 * max(0.0, float(verdict["quiet_mean"] or 0.0) - TARGET_QUIET)
            )
            best_score = (
                (10.0 if best["bar_mean"] >= TARGET_BAR and best["quiet_mean"] <= TARGET_QUIET else 0.0)
                + (3.0 if best["bar_mean"] >= 0.30 and best["quiet_mean"] <= TARGET_QUIET else 0.0)
                + float(best["bar_mean"])
                - 2.0 * max(0.0, float(best["quiet_mean"]) - TARGET_QUIET)
            )
            if score > best_score:
                best = {
                    "bar_mean": verdict["bar_mean"],
                    "quiet_mean": verdict["quiet_mean"],
                    "tag": tag,
                    "path": str(out),
                    "z_bar_mean": verdict.get("z_bar_mean"),
                }
                Path(args.out_root / "LATEST").write_text(str(out.resolve()) + "\n")
                with LOG.open("a") as f:
                    f.write(
                        f"| {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} | "
                        f"**BEST→{tag}** | {verdict['bar_mean']:.3f} | {verdict['quiet_mean']:.3f} | "
                        f"update | {out.name} |\n"
                    )

            remain = (deadline - time.time()) / 3600.0
            print(f"remaining≈{remain:.2f}h best={best}", flush=True)

        wave += 1
        # free lib cache between waves to limit RAM
        lib_cache.clear()

    with LOG.open("a") as f:
        f.write(
            f"\n### Marathon runner finished {datetime.now(timezone.utc).isoformat()}\n"
            f"Best: {json.dumps(best)}\n"
        )
    print("MARATHON DONE", best, flush=True)


if __name__ == "__main__":
    main()
