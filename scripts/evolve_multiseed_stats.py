#!/usr/bin/env python3
"""Multi-seed evolve statistics for amplify_knn_hybrid (+ data reference).

Runs ≥5 independent seeds. Count convention: **disk alone = N_disk**, with
disk:halo:bulge ≈ 4:2:1 ⇒ total ≈ (7/4)·N_disk. Default ``--n-disk 1000000``
⇒ disk=1e6, halo≈5e5, bulge≈2.5e5, total≈1.75e6.

Evolves ~0.5 Gyr with dt=0.01; reports mean±std of A₂(t0), A₂(tend), ΔA₂,
COM drift; plots A₂(t) with mean ±1σ bands. Methods: data + amplify_knn_hybrid.

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=3 \\
    .venv/bin/python scripts/evolve_multiseed_stats.py \\
      --out runs/ml/field_maps/evolve_multiseed_disk1e6_2026-07-25 \\
      --n-seeds 5 --n-disk 1000000 \\
      --paper-figures papers/mnras_noneq_ics/figures
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evolve_resample_compare import (  # noqa: E402
    _evolve_tracked,
    _full_com,
    _metrics,
    _sample_gen,
    _stratified_down,
    _subsample_dump,
    _vcom_only,
)
from sample_latent_ic import RANK, _a2_disk, build_library, default_teacher  # noqa: E402
from galacticsics.ml.fields.feature_library import load_frozen_teacher_bundle  # noqa: E402


def _interp_a2(t_ref: np.ndarray, t: np.ndarray, a2: np.ndarray) -> np.ndarray:
    return np.interp(t_ref, t, a2)


def _plot_bands(path: Path, series: dict[str, dict], t_ref: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    colors = {"data": "#1f4e79", "amplify_knn_hybrid": "#c44e52"}
    labels = {"data": "data dump", "amplify_knn_hybrid": r"amplify\_knn\_hybrid"}
    for tag, pack in series.items():
        mean = pack["a2_mean"]
        std = pack["a2_std"]
        c = colors.get(tag, None)
        ax.plot(t_ref, mean, lw=2.0, color=c, label=labels.get(tag, tag))
        ax.fill_between(t_ref, mean - std, mean + std, color=c, alpha=0.22, linewidth=0)
    ax.set_xlabel(r"$t$ [Gyr]")
    ax.set_ylabel(r"disk $A_2$")
    ax.set_title(r"$A_2(t)$ mean $\pm 1\sigma$ across seeds")
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("runs/ml/field_maps/evolve_multiseed_disk1e6_2026-07-25"),
    )
    p.add_argument("--teacher", type=Path, default=None)
    p.add_argument("--rank", type=Path, default=RANK)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260725)
    p.add_argument("--n-bar", type=int, default=24)
    p.add_argument("--n-quiet", type=int, default=14)
    p.add_argument("--n-mid", type=int, default=8)
    p.add_argument("--n-rot-bar", type=int, default=4)
    p.add_argument("--bar-floor", type=float, default=0.25)
    p.add_argument("--quiet-ceil", type=float, default=0.05)
    p.add_argument("--enc-grid", type=int, default=4)
    p.add_argument("--n-pc", type=int, default=64)
    p.add_argument(
        "--n-disk",
        type=int,
        default=1_000_000,
        help="Disk particle count alone (mix 4:2:1 ⇒ total = 7/4 · n_disk).",
    )
    p.add_argument(
        "--n-evolve",
        type=int,
        default=None,
        help="Total evolve N (overrides --n-disk). Prefer --n-disk.",
    )
    p.add_argument(
        "--n-resample",
        type=int,
        default=None,
        help="Total resample N (default = n_evolve).",
    )
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--evolve-gyr", type=float, default=0.50)
    p.add_argument("--omp", type=int, default=3)
    p.add_argument("--timeout-s", type=float, default=3600.0)
    p.add_argument("--ic-a2-min", type=float, default=0.30)
    p.add_argument("--ic-a2-tries", type=int, default=6)
    p.add_argument("--n-track", type=int, default=11)
    p.add_argument("--data-path", type=Path, default=None)
    p.add_argument(
        "--methods",
        type=str,
        default="amplify_knn_hybrid",
        help="Comma list of generative methods (data always included).",
    )
    p.add_argument("--paper-figures", type=Path, default=None)
    p.add_argument("--paper-prefix", type=str, default="fig7_multiseed")
    args = p.parse_args()
    if args.teacher is None:
        args.teacher = default_teacher()

    # Count mix disk:halo:bulge ≈ 4:2:1 → disk alone = n_disk, total = 7/4 n_disk.
    if args.n_evolve is None:
        args.n_evolve = int(round(args.n_disk * 7 / 4))
    if args.n_resample is None:
        args.n_resample = int(args.n_evolve)
    n_disk_target = int(round(min(args.n_evolve, args.n_resample) * 4 / 7))

    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(max(1, int(args.omp)))
    torch.set_num_threads(max(1, int(args.omp)))

    ranked = json.loads(Path(args.rank).read_text())
    ranked.sort(key=lambda r: r["a2"], reverse=True)
    data_path = args.data_path or Path(ranked[0]["path"])
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    # Prefer working methods only; drop z_amplify unless explicitly requested.
    n_ev = min(int(args.n_evolve), int(args.n_resample))

    print(f"=== multi-seed evolve teacher={args.teacher} ===", flush=True)
    print(
        f"n_disk≈{n_disk_target:,}  total N={n_ev:,} (mix 4:2:1)  "
        f"seeds={args.n_seeds}  evolve={args.evolve_gyr} Gyr  "
        f"dt={args.dt}  OMP={args.omp}  methods={['data'] + methods}",
        flush=True,
    )
    _teacher, _cfg, _stats = load_frozen_teacher_bundle(args.teacher)
    print("=== build stratified library (once) ===", flush=True)
    lib = build_library(args)
    print(
        f"library n={len(lib.feats)} bar={lib.bar_pool.size} quiet={lib.quiet_pool.size}",
        flush=True,
    )

    per_seed: list[dict] = []
    series_raw: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "data": [],
        **{m: [] for m in methods},
    }

    for si in range(int(args.n_seeds)):
        seed_i = int(args.seed) + 17 * si
        rng = np.random.default_rng(seed_i)
        torch.manual_seed(seed_i)
        print(f"\n=== seed {si + 1}/{args.n_seeds}  seed={seed_i} ===", flush=True)
        seed_row: dict = {"seed": seed_i, "arms": {}}

        # data reference (independent stratified subsample per seed)
        data_ev = _full_com(_subsample_dump(data_path, n_ev, rng))
        arms = {"data": {"parts": data_ev, "meta": {"path": str(data_path)}}}

        for method in methods:
            best = None
            for attempt in range(max(1, args.ic_a2_tries)):
                gen, meta = _sample_gen(lib, method, rng, args.n_resample)
                gen_ev = _vcom_only(_stratified_down(gen, n_ev, rng))
                a2 = float(_a2_disk(gen_ev))
                print(f"  {method} try#{attempt} IC A2={a2:.3f}", flush=True)
                if best is None or a2 > best[0]:
                    best = (a2, gen_ev, meta)
                if a2 >= args.ic_a2_min:
                    break
            assert best is not None
            arms[method] = {
                "parts": best[1],
                "meta": {k: v for k, v in best[2].items() if k != "z"},
            }

        for tag, pack in arms.items():
            parts = pack["parts"]
            pre = _metrics(parts)
            print(f"  evolve {tag} A2={pre['a2']:.3f} N={parts['pos'].shape[0]}", flush=True)
            evo = _evolve_tracked(
                parts,
                end_gyr=args.evolve_gyr,
                dt=args.dt,
                omp=args.omp,
                timeout_s=args.timeout_s,
                n_track=args.n_track,
                faceon_times=None,
            )
            if not evo.get("ok"):
                seed_row["arms"][tag] = {
                    "ok": False,
                    "pre": pre,
                    "error": evo.get("error"),
                }
                print(f"    FAIL {evo.get('error')}", flush=True)
                continue
            pos_f = evo.pop("pos_final")
            evo.pop("faceon_maps", None)
            evo.pop("faceon_t_gyr", None)
            post_parts = {**parts, "pos": pos_f}
            post = _metrics(post_parts)
            row = {
                "ok": True,
                "a2_pre": pre["a2"],
                "a2_post": post["a2"],
                "da2": post["a2"] - pre["a2"],
                "com_drift_kpc": evo["com_drift_kpc"],
                "wall_s": evo["wall_s"],
                "t_gyr": evo["t_gyr"],
                "a2_t": evo["a2_t"],
                "meta": pack.get("meta", {}),
            }
            seed_row["arms"][tag] = row
            series_raw[tag].append(
                (np.asarray(evo["t_gyr"], dtype=float), np.asarray(evo["a2_t"], dtype=float))
            )
            print(
                f"    A2 {pre['a2']:.3f}→{post['a2']:.3f}  Δ={row['da2']:+.3f}  "
                f"COM={row['com_drift_kpc']:.4f}  wall={row['wall_s']:.1f}s",
                flush=True,
            )
        per_seed.append(seed_row)
        # checkpoint after each seed
        (args.out / "checkpoint_per_seed.json").write_text(json.dumps(per_seed, indent=2))

    # Aggregate on common time grid from first successful data run
    t_ref = None
    for s in per_seed:
        arm = s["arms"].get("data")
        if arm and arm.get("ok"):
            t_ref = np.asarray(arm["t_gyr"], dtype=float)
            break
    if t_ref is None:
        t_ref = np.linspace(0.0, float(args.evolve_gyr), args.n_track)

    summary_arms: dict[str, dict] = {}
    plot_series: dict[str, dict] = {}
    for tag in ["data", *methods]:
        ok_rows = [
            s["arms"][tag]
            for s in per_seed
            if tag in s["arms"] and s["arms"][tag].get("ok")
        ]
        if not ok_rows:
            summary_arms[tag] = {"ok": False, "n_ok": 0}
            continue
        a2_pre = np.array([r["a2_pre"] for r in ok_rows], dtype=float)
        a2_post = np.array([r["a2_post"] for r in ok_rows], dtype=float)
        da2 = np.array([r["da2"] for r in ok_rows], dtype=float)
        com = np.array([r["com_drift_kpc"] for r in ok_rows], dtype=float)
        stacked = np.vstack(
            [_interp_a2(t_ref, *pair) for pair in series_raw[tag]]
        )
        summary_arms[tag] = {
            "ok": True,
            "n_ok": len(ok_rows),
            "a2_pre_mean": float(a2_pre.mean()),
            "a2_pre_std": float(a2_pre.std(ddof=0)),
            "a2_post_mean": float(a2_post.mean()),
            "a2_post_std": float(a2_post.std(ddof=0)),
            "da2_mean": float(da2.mean()),
            "da2_std": float(da2.std(ddof=0)),
            "com_drift_mean": float(com.mean()),
            "com_drift_std": float(com.std(ddof=0)),
            "t_gyr": t_ref.tolist(),
            "a2_t_mean": stacked.mean(axis=0).tolist(),
            "a2_t_std": stacked.std(axis=0, ddof=0).tolist(),
        }
        plot_series[tag] = {
            "a2_mean": stacked.mean(axis=0),
            "a2_std": stacked.std(axis=0, ddof=0),
        }

    report = {
        "dt": args.dt,
        "evolve_gyr": args.evolve_gyr,
        "n_steps": int(np.ceil(args.evolve_gyr / args.dt)),
        "n_disk": n_disk_target,
        "n_halo_approx": int(round(n_ev * 2 / 7)),
        "n_bulge_approx": int(round(n_ev * 1 / 7)),
        "n_evolve": n_ev,
        "n_resample": int(args.n_resample),
        "n_seeds": int(args.n_seeds),
        "base_seed": int(args.seed),
        "omp": int(args.omp),
        "force": "bh_c",
        "teacher": str(args.teacher),
        "data_path": str(data_path),
        "methods": ["data", *methods],
        "count_mix": "disk:halo:bulge ≈ 4:2:1",
        "count_convention": (
            f"disk alone ≈ {n_disk_target:,}; "
            f"halo ≈ {int(round(n_ev * 2 / 7)):,}; "
            f"bulge ≈ {int(round(n_ev * 1 / 7)):,}; "
            f"total = {n_ev:,}"
        ),
        "shared_centering": (
            "data: full soft COM; generative: morphological origin + VCOM-only"
        ),
        "index": "PCA (not LDA)",
        "per_seed": per_seed,
        "summary": summary_arms,
    }

    a2_png = args.out / "a2_t_bands.png"
    _plot_bands(a2_png, plot_series, t_ref)
    report["figures"] = {"a2_t_bands": str(a2_png)}

    out_json = args.out / "verdict.json"
    out_json.write_text(json.dumps(report, indent=2))

    lines = [
        "# Multi-seed evolve statistics",
        "",
        f"Teacher: `{args.teacher}`",
        f"**Disk alone ≈ {n_disk_target:,}**; total N = **{n_ev:,}** "
        f"(disk:halo:bulge ≈ 4:2:1 ⇒ halo≈{int(round(n_ev * 2 / 7)):,}, "
        f"bulge≈{int(round(n_ev * 1 / 7)):,}).",
        f"Seeds = **{args.n_seeds}**; evolve = {args.evolve_gyr} Gyr, dt={args.dt}; "
        f"OpenMP={args.omp}.",
        "",
        "| Arm | A₂(t₀) | A₂(t_end) | ΔA₂ | COM drift [kpc] | n_ok |",
        "|-----|--------|-----------|-----|-----------------|------|",
    ]
    for tag, row in summary_arms.items():
        if not row.get("ok"):
            lines.append(f"| `{tag}` | FAIL | — | — | — | 0 |")
            continue
        lines.append(
            f"| `{tag}` | "
            f"{row['a2_pre_mean']:.3f}±{row['a2_pre_std']:.3f} | "
            f"{row['a2_post_mean']:.3f}±{row['a2_post_std']:.3f} | "
            f"{row['da2_mean']:+.3f}±{row['da2_std']:.3f} | "
            f"{row['com_drift_mean']:.4f}±{row['com_drift_std']:.4f} | "
            f"{row['n_ok']} |"
        )
    lines += ["", f"Figure: `{a2_png.name}` (mean±1σ bands).", ""]
    (args.out / "SUMMARY.md").write_text("\n".join(lines))
    print("\n" + (args.out / "SUMMARY.md").read_text(), flush=True)

    if args.paper_figures is not None:
        args.paper_figures.mkdir(parents=True, exist_ok=True)
        prefix = args.paper_prefix or "fig7_multiseed"
        dest = args.paper_figures / f"{prefix}_a2_t_bands.png"
        shutil.copy2(a2_png, dest)
        shutil.copy2(args.out / "SUMMARY.md", Path("papers/mnras_noneq_ics/results") / "evolve_multiseed_SUMMARY.md")
        shutil.copy2(out_json, Path("papers/mnras_noneq_ics/results") / "evolve_multiseed_verdict.json")
        print(f"paper ← {dest.name}", flush=True)

    print(f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
