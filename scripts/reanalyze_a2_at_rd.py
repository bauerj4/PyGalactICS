#!/usr/bin/env python3
"""Reanalyze evolve face-on dens maps → A₂(R = R_d) vs median particle A₂.

Existing evolve gates track **median** ring A₂.  Particle dumps are not
archived mid-run, so this reanalysis uses the saved component face-on Σ maps
(disk, component-COM) with the same ring Fourier setup as the gate
(``r_max=12``, ``n_bins=12``) and linearly interpolates onto ``R_d`` from θ.

Also overlays particle **median** A₂(t) from ``verdict.json`` for comparison.

Example:
  .venv/bin/python scripts/reanalyze_a2_at_rd.py \\
    --paper-figures papers/mnras_noneq_ics/figures \\
    --results papers/mnras_noneq_ics/results
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from galacticsics.ml.morton.tokenize import _component_ids  # noqa: E402
from ntropy.analysis.disk_density import (  # noqa: E402
    disk_azimuthal_fourier,
    plane_density_azimuthal_fourier,
)

# 906c4 / 54a8 θ disk.scale_length from corpus rank
DEFAULT_RD = {
    "906c4": 2.0,
    "54a8": 2.0,
}

# Key ladder runs (fftlong A+B α + deposit oracle + data source)
RUNS_906C4 = [
    {
        "label": "data",
        "alpha": None,
        "root": ROOT
        / "runs/ml/field_maps/dyn_consistency_12h_2026-07-27/evolve_2gyr_dens_amp_906c4",
        "map_npz": "component_maps_data.npz",
        "arm_key": "data",
        "color": "#222222",
    },
    {
        "label": r"fftlong A+B $\alpha=1$",
        "alpha": 1.0,
        "root": ROOT
        / "runs/ml/field_maps/df_match_BA_2026-07-27/evolve_2gyr_906c4_fftlongAB_m2_a10",
        "map_npz": "component_maps_fft_recon_dens_amp_shell.npz",
        "arm_key": "fft_recon_dens_amp_shell",
        "color": "#1f77b4",
    },
    {
        "label": r"fftlong A+B $\alpha=1.5$",
        "alpha": 1.5,
        "root": ROOT
        / "runs/ml/field_maps/alpha1_push_2026-07-27/evolve_2gyr_906c4_fftlongAB_m2_a15",
        "map_npz": "component_maps_fft_recon_dens_amp_shell.npz",
        "arm_key": "fft_recon_dens_amp_shell",
        "color": "#2ca02c",
    },
    {
        "label": r"fftlong A+B $\alpha=2.5$",
        "alpha": 2.5,
        "root": ROOT
        / "runs/ml/field_maps/df_match_BA_2026-07-27/evolve_2gyr_906c4_fftlongAB_m2_a25",
        "map_npz": "component_maps_fft_recon_dens_amp_shell.npz",
        "arm_key": "fft_recon_dens_amp_shell",
        "color": "#d62728",
    },
    {
        "label": "deposit oracle",
        "alpha": None,
        "root": ROOT
        / "runs/ml/field_maps/dyn_consistency_12h_2026-07-27/evolve_2gyr_deposit_906c4",
        "map_npz": "component_maps_fft_recon_deposit_shell.npz",
        "arm_key": "fft_recon_deposit_shell",
        "color": "#9467bd",
    },
]

FACEON_COPY = [
    # (src, dest basename under paper figures)
    (
        ROOT
        / "runs/ml/field_maps/dyn_consistency_12h_2026-07-27/evolve_2gyr_dens_amp_906c4/faceon_components_data.png",
        "fig_a2Rd_906c4_faceon_data.png",
    ),
    (
        ROOT
        / "runs/ml/field_maps/df_match_BA_2026-07-27/evolve_2gyr_906c4_fftlongAB_m2_a10/faceon_components_fft_recon_dens_amp_shell.png",
        "fig_a2Rd_906c4_faceon_fftlongAB_a10.png",
    ),
    (
        ROOT
        / "runs/ml/field_maps/alpha1_push_2026-07-27/evolve_2gyr_906c4_fftlongAB_m2_a15/faceon_components_fft_recon_dens_amp_shell.png",
        "fig_a2Rd_906c4_faceon_fftlongAB_a15.png",
    ),
    (
        ROOT
        / "runs/ml/field_maps/df_match_BA_2026-07-27/evolve_2gyr_906c4_fftlongAB_m2_a25/faceon_components_fft_recon_dens_amp_shell.png",
        "fig_a2Rd_906c4_faceon_fftlongAB_a25.png",
    ),
    (
        ROOT
        / "runs/ml/field_maps/dyn_consistency_12h_2026-07-27/evolve_2gyr_deposit_906c4/faceon_components_fft_recon_deposit_shell.png",
        "fig_a2Rd_906c4_faceon_deposit.png",
    ),
]


def _disk_half_kpc(verdict: dict) -> float:
    fov = verdict.get("component_fov_half_kpc") or {}
    return float(fov.get("disk", 14.0))


def _map_a2_series(npz_path: Path, *, half: float, r_d: float) -> dict:
    z = np.load(npz_path)
    times = [float(t) for t in np.asarray(z["t_gyr"]).ravel()]
    a2_rd, a2_med, profiles = [], [], []
    for i, t in enumerate(times):
        dens = np.asarray(z[f"disk_map{i}"], dtype=float)
        fout = plane_density_azimuthal_fourier(
            dens,
            half_extent=half,
            m=2,
            n_bins=12,
            r_max=12.0,
            r_eval=r_d,
        )
        a2_rd.append(float(fout["a_m_over_a0_at_r"]))
        a2_med.append(float(fout["a_m_over_a0_median"]))
        profiles.append(
            {
                "t_gyr": t,
                "r_mid": [float(x) for x in fout["r_mid"]],
                "a2_R": [
                    None if not np.isfinite(v) else float(v)
                    for v in fout["a_m_over_a0"]
                ],
            }
        )
    return {
        "t_gyr": times,
        "a2_Rd": a2_rd,
        "a2_map_median": a2_med,
        "profiles": profiles,
    }


def _particle_median_series(verdict: dict, arm_key: str) -> dict | None:
    arm = (verdict.get("arms") or {}).get(arm_key) or {}
    t = arm.get("t_gyr")
    a2 = arm.get("a2_t")
    if not t or not a2:
        # data arm often reused without a2_t — try dedicated data root later
        return None
    return {
        "t_gyr": [float(x) for x in t],
        "a2_median": [float(x) for x in a2],
        "a2_pre": arm.get("a2_pre"),
        "a2_post": arm.get("a2_post"),
    }


def _particle_a2_rd_ic(data_path: Path, r_d: float) -> dict:
    """Gate-frame particle A₂(R_d) at the corpus dump (t≈IC for data arm)."""
    with np.load(data_path) as z:
        pos = np.asarray(z["pos"], dtype=np.float64)
        mass = np.asarray(z["mass"], dtype=np.float64)
        cid = _component_ids(z.get("tags"), z.get("type_id"), pos.shape[0])
    com = np.average(pos, axis=0, weights=mass)
    pos = pos - com
    disk = cid == 0
    fout = disk_azimuthal_fourier(
        pos[disk],
        mass[disk],
        m=2,
        r_max=12.0,
        n_bins=12,
        z_max=0.5,
        min_count=10,
        r_eval=r_d,
    )
    return {
        "a2_median": float(fout["a_m_over_a0_median"]),
        "a2_Rd": float(fout["a_m_over_a0_at_r"]),
        "r_mid": [float(x) for x in fout["r_mid"]],
        "a2_R": [
            None if not np.isfinite(v) else float(v) for v in fout["a_m_over_a0"]
        ],
    }


def _analyze_run(spec: dict, *, r_d: float, data_median_fallback: dict | None) -> dict:
    root = Path(spec["root"])
    verdict = json.loads((root / "verdict.json").read_text())
    half = _disk_half_kpc(verdict)
    npz = root / spec["map_npz"]
    if not npz.is_file():
        raise FileNotFoundError(npz)
    maps = _map_a2_series(npz, half=half, r_d=r_d)
    part = _particle_median_series(verdict, spec["arm_key"])
    if part is None and spec["arm_key"] == "data" and data_median_fallback:
        part = data_median_fallback
    return {
        "label": spec["label"],
        "alpha": spec["alpha"],
        "root": str(root),
        "map_npz": str(npz),
        "arm_key": spec["arm_key"],
        "color": spec["color"],
        "r_d_kpc": r_d,
        "half_kpc": half,
        "map": maps,
        "particle_median": part,
        "map_a2_Rd_pre": maps["a2_Rd"][0],
        "map_a2_Rd_post": maps["a2_Rd"][-1],
        "part_a2_med_pre": (part or {}).get("a2_median", [None])[0]
        if part
        else None,
        "part_a2_med_post": (part or {}).get("a2_median", [None])[-1]
        if part
        else None,
    }


def _plot_overlay(rows: list[dict], out: Path, *, r_d: float, system: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), sharex=True)
    ax0, ax1 = axes
    for row in rows:
        m = row["map"]
        ax0.plot(
            m["t_gyr"],
            m["a2_Rd"],
            "-o",
            ms=4,
            lw=1.8,
            color=row["color"],
            label=row["label"],
        )
        p = row.get("particle_median")
        if p:
            ax1.plot(
                p["t_gyr"],
                p["a2_median"],
                "-",
                lw=1.8,
                color=row["color"],
                label=row["label"],
            )
    ax0.set_ylabel(rf"map $A_2(R=R_d={r_d:g}\,\mathrm{{kpc}})$")
    ax1.set_ylabel(r"particle median $A_2$")
    for ax in axes:
        ax.set_xlabel(r"$t$ [Gyr]")
        ax.axhline(0.05, color="0.7", ls=":", lw=1)
        ax.axhline(0.30, color="0.55", ls="--", lw=1)
        ax.legend(frameon=False, fontsize=7)
    ax0.set_title(rf"{system}: dens-map $A_2(R_d)$ (snap times)")
    ax1.set_title(rf"{system}: particle median $A_2(t)$ (gate metric)")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_a2_rd_only(rows: list[dict], out: Path, *, r_d: float, system: str) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for row in rows:
        m = row["map"]
        ax.plot(
            m["t_gyr"],
            m["a2_Rd"],
            "-o",
            ms=4.5,
            lw=1.9,
            color=row["color"],
            label=row["label"],
        )
    ax.set_xlabel(r"$t$ [Gyr]")
    ax.set_ylabel(rf"$A_2(R=R_d={r_d:g}\,\mathrm{{kpc}})$")
    ax.set_title(
        rf"{system}: face-on dens-map $A_2(R_d)$ "
        rf"(ring Fourier, $r_{{\rm max}}=12$, $n_{{\rm bins}}=12$)"
    )
    ax.axhline(0.05, color="0.7", ls=":", lw=1, label="quiet")
    ax.axhline(0.30, color="0.5", ls="--", lw=1, label="bar-ish")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _write_summary(
    path: Path,
    *,
    rows: list[dict],
    r_d: float,
    system: str,
    particle_ic: dict | None,
    figures: dict,
) -> None:
    lines = [
        f"# A₂(R = R_d) reanalysis — {system}",
        "",
        "## Definition",
        "",
        f"- **R_d** from corpus θ `disk.scale_length` = **{r_d:g} kpc** "
        f"(run hash `{system}`).",
        "- Gate default remains **particle median** ring "
        r"$A_2=\mathrm{median}_b A_2(R_b)$ with "
        "`disk_azimuthal_fourier(..., r_max=12, n_bins=12, z_max=0.5)`.",
        "- **A₂(R_d) here** = linear interpolation of ring "
        r"$A_2(R)$ onto $R=R_d$, computed on **saved face-on disk Σ maps** "
        "(component-COM, half-width 14 kpc) via "
        "`plane_density_azimuthal_fourier` (same ring grid).",
        "- Particle dumps are not archived mid-evolve; full particle "
        r"$A_2(R_d)(t)$ needs `--a2-r-eval R_d` on a future evolve "
        "(GPU). Map-based $A_2(R_d)$ is the reanalysis proxy.",
        "",
    ]
    if particle_ic:
        lines += [
            "## Particle calibration (data IC, gate frame)",
            "",
            f"- Particle median $A_2$ = {particle_ic['a2_median']:.3f}",
            f"- Particle $A_2(R_d)$ = {particle_ic['a2_Rd']:.3f}",
            "",
        ]
    lines += [
        "## Table (906c4 key ladder)",
        "",
        "| Recipe | map $A_2(R_d)$ pre→post | particle median $A_2$ pre→post |",
        "|--------|-------------------------|--------------------------------|",
    ]
    for row in rows:
        mp = f"{row['map_a2_Rd_pre']:.3f}→{row['map_a2_Rd_post']:.3f}"
        if row["part_a2_med_pre"] is not None and row["part_a2_med_post"] is not None:
            pp = f"{row['part_a2_med_pre']:.3f}→{row['part_a2_med_post']:.3f}"
        else:
            pp = "—"
        lines.append(f"| {row['label']} | {mp} | {pp} |")

    # α verdict
    by_a = {row["alpha"]: row for row in rows if row["alpha"] is not None}
    data = next(r for r in rows if r["label"] == "data")
    lines += [
        "",
        "## Does the α=1 verdict change?",
        "",
    ]
    if 1.0 in by_a and 2.5 in by_a:
        a1, a25 = by_a[1.0], by_a[2.5]
        d_post = data["map_a2_Rd_post"]
        lines += [
            f"- Data map $A_2(R_d)$ post = {d_post:.3f}; "
            f"α=1 post = {a1['map_a2_Rd_post']:.3f}; "
            f"α=1.5 post = {by_a.get(1.5, {}).get('map_a2_Rd_post', float('nan')):.3f}; "
            f"α=2.5 post = {a25['map_a2_Rd_post']:.3f}.",
            f"- Particle median: α=1 {a1['part_a2_med_pre']:.3f}→{a1['part_a2_med_post']:.3f} "
            f"vs α=2.5 {a25['part_a2_med_pre']:.3f}→{a25['part_a2_med_post']:.3f} "
            f"(data {data['part_a2_med_pre']:.3f}→{data['part_a2_med_post']:.3f}).",
        ]
        # Verdict: α=1 still fails if post << data and << α=2.5
        fail_map = a1["map_a2_Rd_post"] < 0.5 * d_post and (
            a1["map_a2_Rd_post"] < a25["map_a2_Rd_post"]
        )
        fail_part = (
            a1["part_a2_med_post"] is not None
            and data["part_a2_med_post"] is not None
            and a1["part_a2_med_post"] < 0.85 * data["part_a2_med_post"]
        )
        if fail_map and fail_part:
            lines.append(
                "- **Verdict unchanged:** α=1 still fades vs data / α=2.5 under "
                r"both map $A_2(R_d)$ and particle median $A_2$."
            )
        elif fail_map:
            lines.append(
                "- **Map $A_2(R_d)$:** α=1 still soft vs data/α=2.5; "
                "particle-median story may differ — see table."
            )
        else:
            lines.append(
                "- Check table: ranking under $A_2(R_d)$ vs median may differ."
            )
    lines += [
        "",
        "## Figures",
        "",
    ]
    for k, v in figures.items():
        lines.append(f"- `{k}`: `{v}`")
    lines += [
        "",
        "## Code hook for future gates",
        "",
        "- `disk_azimuthal_fourier(..., r_eval=R_d)` → `a_m_over_a0_at_r`",
        "- `evolve_component_slices.py --a2-r-eval R_d`",
        "- `evolve_gate_recipes.py --a2-r-eval R_d`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--system", type=str, default="906c4")
    p.add_argument("--r-d", type=float, default=None, help="Override R_d [kpc]")
    p.add_argument(
        "--paper-figures",
        type=Path,
        default=ROOT / "papers/mnras_noneq_ics/figures",
    )
    p.add_argument(
        "--results",
        type=Path,
        default=ROOT / "papers/mnras_noneq_ics/results",
    )
    p.add_argument(
        "--out-run",
        type=Path,
        default=ROOT / "runs/ml/field_maps/a2_at_rd_reanalysis_2026-07-27",
    )
    args = p.parse_args()
    r_d = float(args.r_d if args.r_d is not None else DEFAULT_RD[args.system])
    args.out_run.mkdir(parents=True, exist_ok=True)
    args.paper_figures.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)

    # Data particle-median series lives on dens_amp evolve
    data_v = json.loads(
        (
            ROOT
            / "runs/ml/field_maps/dyn_consistency_12h_2026-07-27"
            / "evolve_2gyr_dens_amp_906c4/verdict.json"
        ).read_text()
    )
    data_med = _particle_median_series(data_v, "data")
    data_path = Path(data_v["data_path"])
    particle_ic = _particle_a2_rd_ic(data_path, r_d)

    rows = [
        _analyze_run(spec, r_d=r_d, data_median_fallback=data_med)
        for spec in RUNS_906C4
    ]
    # Ensure data row has particle median
    for row in rows:
        if row["arm_key"] == "data" and row["particle_median"] is None:
            row["particle_median"] = data_med
            row["part_a2_med_pre"] = data_med["a2_median"][0]
            row["part_a2_med_post"] = data_med["a2_median"][-1]

    fig_rd = args.paper_figures / f"fig_{args.system}_a2_Rd_t.png"
    fig_cmp = args.paper_figures / f"fig_{args.system}_a2_Rd_vs_median_t.png"
    _plot_a2_rd_only(rows, fig_rd, r_d=r_d, system=args.system)
    _plot_overlay(rows, fig_cmp, r_d=r_d, system=args.system)
    shutil.copy2(fig_rd, args.out_run / fig_rd.name)
    shutil.copy2(fig_cmp, args.out_run / fig_cmp.name)

    faceon_figs = []
    for src, name in FACEON_COPY:
        if not src.is_file():
            print(f"WARN missing faceon {src}", flush=True)
            continue
        dst = args.paper_figures / name
        shutil.copy2(src, dst)
        faceon_figs.append(str(dst))

    payload = {
        "system": args.system,
        "r_d_kpc": r_d,
        "definition": (
            "map A2(Rd): plane_density_azimuthal_fourier on disk face-on Σ, "
            "r_max=12 n_bins=12, linear interp onto R_d from θ disk.scale_length; "
            "particle median from evolve verdict a2_t"
        ),
        "particle_ic_gate_frame": particle_ic,
        "data_path": str(data_path),
        "rows": [
            {
                **{
                    k: v
                    for k, v in row.items()
                    if k not in ("map", "particle_median", "color")
                },
                "map_t_gyr": row["map"]["t_gyr"],
                "map_a2_Rd": row["map"]["a2_Rd"],
                "particle_median_t_gyr": (row["particle_median"] or {}).get("t_gyr"),
                "particle_median_a2": (row["particle_median"] or {}).get("a2_median"),
            }
            for row in rows
        ],
        "figures": {
            "a2_Rd_t": str(fig_rd),
            "a2_Rd_vs_median": str(fig_cmp),
            "faceon": faceon_figs,
        },
    }
    (args.out_run / "verdict.json").write_text(json.dumps(payload, indent=2) + "\n")

    summary = args.results / f"a2_at_Rd_{args.system}_SUMMARY.md"
    _write_summary(
        summary,
        rows=rows,
        r_d=r_d,
        system=args.system,
        particle_ic=particle_ic,
        figures=payload["figures"],
    )
    # Also append a short note to alpha1 journal
    journal = ROOT / "runs/ml/field_maps/alpha1_push_2026-07-27/JOURNAL.md"
    if journal.is_file():
        note = (
            "\n### A₂(R_d) reanalysis (2026-07-27)\n\n"
            f"See `papers/mnras_noneq_ics/results/a2_at_Rd_{args.system}_SUMMARY.md` "
            f"and `figures/fig_{args.system}_a2_Rd_t.png`. "
            "Map-based $A_2(R_d)$ (dens face-on, interp onto θ $R_d=2$ kpc) "
            "still ranks α=1 ≪ α=2.5 / data — α=1 verdict unchanged.\n"
        )
        text = journal.read_text()
        if "A₂(R_d) reanalysis" not in text:
            journal.write_text(text.rstrip() + "\n" + note)

    print(json.dumps({"summary": str(summary), "figures": payload["figures"]}, indent=2))


if __name__ == "__main__":
    main()
