#!/usr/bin/env python3
"""OOD dens+vel DF analysis for residual_f0 vs FFT recon (t=0).

Uses min_count estimators via ood_theta_df_compare._bundle / _compare_to_ref.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evolve_resample_compare import (  # noqa: E402
    _a2_disk,
    _recon_particles,
    _residual_f0_particles,
    _stratified_down,
    _vcom_only,
)
from galacticsics.ml.fields.feature_library import load_frozen_teacher_bundle  # noqa: E402
from ood_theta_compare import CORPUS, DEFAULT_OOD_CASES, _ensure_eps, _load_ic  # noqa: E402
from ood_theta_df_compare import (  # noqa: E402
    ARM_LABELS,
    _bundle,
    _compare_to_ref,
    _plot_dens_panel,
    _plot_vel_panel,
)
import ood_theta_df_compare as odf  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-disk", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--cases",
        type=str,
        default="heavy_ext_quiet,heavy_bar_forming,light_compact_bar",
    )
    p.add_argument(
        "--morph-bar",
        type=Path,
        default=Path("runs/mw_morton_corpus_v2/906c4af73543/evolution/particles/step_003200.npz"),
    )
    p.add_argument("--bar-alpha", type=float, default=2.5)
    p.add_argument(
        "--bar-morph-source",
        choices=("teacher_recon", "deposit", "blend"),
        default="teacher_recon",
    )
    p.add_argument("--bar-blend-weight", type=float, default=0.9)
    p.add_argument("--paper-figures", type=Path, default=None)
    p.add_argument("--paper-results", type=Path, default=None)
    p.add_argument("--faceon-bins", type=int, default=96)
    args = p.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "2")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.paper_figures is not None:
        args.paper_figures.mkdir(parents=True, exist_ok=True)

    teacher, cfg, stats = load_frozen_teacher_bundle(args.teacher)
    rng = np.random.default_rng(args.seed)
    n_tot = int(round(args.n_disk * 7 / 4))
    plot_order = ("galactics_ic", "fft_recon", "residual_f0_self", "residual_f0_bar906")
    odf.ARM_ORDER = plot_order

    want = {x.strip() for x in args.cases.split(",") if x.strip()}
    cases = [c for c in DEFAULT_OOD_CASES if c["name"] in want]
    if not cases:
        raise SystemExit(f"no cases from {want}")

    print(
        f"=== OOD residual_f0 n_disk={args.n_disk:,} → N={n_tot:,} "
        f"cases={[c['name'] for c in cases]} ===",
        flush=True,
    )

    rows: list[dict] = []
    for spec in cases:
        h = spec["run_hash"]
        ic_path = Path(CORPUS) / h / "ic_state.npz"
        print(f"=== {spec['name']} ({h[:12]}) ===", flush=True)
        ic = _ensure_eps(_load_ic(ic_path, n_tot, rng))
        recon = _ensure_eps(
            _vcom_only(
                _recon_particles(
                    ic_path, teacher, cfg, stats, n_resample=n_tot, rng=rng
                )
            )
        )
        res = _ensure_eps(
            _vcom_only(
                _stratified_down(
                    _residual_f0_particles(
                        ic_path,
                        ic_path,
                        teacher,
                        cfg,
                        stats,
                        n_resample=n_tot,
                        rng=rng,
                        morph_source="teacher_recon",
                        alpha=1.0,
                        dens_resid_kind="m2",
                        residual_scale="multiplicative",
                        velocity_mode="transplant",
                    ),
                    n_tot,
                    rng,
                )
            )
        )
        res_bar = _ensure_eps(
            _vcom_only(
                _stratified_down(
                    _residual_f0_particles(
                        ic_path,
                        args.morph_bar,
                        teacher,
                        cfg,
                        stats,
                        n_resample=n_tot,
                        rng=rng,
                        morph_source=str(args.bar_morph_source),
                        alpha=float(args.bar_alpha),
                        dens_resid_kind="m2",
                        residual_scale="multiplicative",
                        velocity_mode="transplant",
                        blend_weight=float(args.bar_blend_weight),
                    ),
                    n_tot,
                    rng,
                )
            )
        )
        particles = {
            "galactics_ic": ic,
            "fft_recon": recon,
            "residual_f0_self": res,
            "residual_f0_bar906": res_bar,
        }
        bundles = {
            k: _bundle(v, faceon_bins=args.faceon_bins) for k, v in particles.items()
        }
        ref = bundles["galactics_ic"]
        metrics = {
            k: _compare_to_ref(ref, bundles[k])
            for k in particles
            if k != "galactics_ic"
        }
        a2s = {k: float(_a2_disk(v)) for k, v in particles.items()}
        dens_png = args.out / f"{spec['name']}_dens_t0.png"
        vel_png = args.out / f"{spec['name']}_vel_t0.png"
        _plot_dens_panel(
            dens_png, bundles, title=rf"OOD {spec['name']}: dens DF at $t=0$"
        )
        _plot_vel_panel(
            vel_png, bundles, title=rf"OOD {spec['name']}: vel DF at $t=0$"
        )
        if args.paper_figures is not None:
            for src, name in (
                (dens_png, f"fig_residual_f0_ood_{spec['name']}_dens_t0.png"),
                (vel_png, f"fig_residual_f0_ood_{spec['name']}_vel_t0.png"),
            ):
                dst = args.paper_figures / name
                dst.write_bytes(src.read_bytes())
                print(f"  paper ← {dst.name}", flush=True)
        rows.append({"case": spec["name"], "hash": h, "a2": a2s, "metrics": metrics})

    arms = ("fft_recon", "residual_f0_self", "residual_f0_bar906")
    labels = [r["case"].replace("_", "\n") for r in rows]
    x = np.arange(len(rows))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for i, arm in enumerate(arms):
        vals = [r["metrics"][arm]["kin_mean_vphi_abs_mse"] for r in rows]
        ax.bar(x + (i - 1) * width, vals, width, label=ARM_LABELS[arm])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"disk $\langle v_\phi\rangle$ abs MSE vs GalactICS")
    ax.set_title(r"OOD residual $f_0$ vs FFT recon: rotation match")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    score = args.out / "vphi_scoreboard.png"
    fig.savefig(score, dpi=160, bbox_inches="tight")
    plt.close(fig)
    if args.paper_figures is not None:
        dst = args.paper_figures / "fig_residual_f0_ood_vphi_scoreboard.png"
        dst.write_bytes(score.read_bytes())
        print(f"paper ← {dst.name}", flush=True)

    lines = [
        "# OOD residual_f0 vs FFT recon (DF proximity to GalactICS IC)",
        "",
        f"n_disk={args.n_disk:,} (corpus-scale). "
        "residual_f0_self = morph=IC teacher; "
        f"residual_f0_bar906 = morph=906c4 teacher α={args.bar_alpha}. "
        "Metrics: min_count annular / deposit estimators "
        "(`results/NOTE_min_count_estimators.md`).",
        "",
        "Figures: `fig_residual_f0_ood_*_{dens,vel}_t0.png`, "
        "`fig_residual_f0_ood_vphi_scoreboard.png`.",
        "",
        "| Case | arm | A₂ | faceon dens MSE | disk dens rel MSE | disk ⟨vφ⟩ MSE |",
        "|------|-----|----|-----------------|-------------------|---------------|",
    ]
    for r in rows:
        for arm in arms:
            m = r["metrics"][arm]
            lines.append(
                f"| {r['case']} | `{arm}` | {r['a2'][arm]:.3f} | "
                f"{m['faceon_disk_mse']:.4g} | {m['dens_disk_rel_mse']:.4g} | "
                f"{m['kin_mean_vphi_abs_mse']:.4g} |"
            )
    summary = "\n".join(lines) + "\n"
    (args.out / "SUMMARY.md").write_text(summary)
    (args.out / "verdict.json").write_text(
        json.dumps(
            {
                "n_disk": args.n_disk,
                "n_total": n_tot,
                "teacher": str(args.teacher),
                "bar_alpha": args.bar_alpha,
                "cases": rows,
            },
            indent=2,
            default=float,
        )
        + "\n"
    )
    if args.paper_results is not None:
        args.paper_results.mkdir(parents=True, exist_ok=True)
        (args.paper_results / "ood_residual_f0_SUMMARY.md").write_text(summary)
    print(summary, flush=True)
    print("=== OOD residual_f0 DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
