#!/usr/bin/env python3
"""Re-evolve saved t0 particles with disk-COM A2(R_d) tracking; refresh paper figs.

Uses the exact stratified ICs from prior gates (same dynamics), only the A2
metric changes via disk_azimuthal_fourier(recenter=True) default.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evolve_component_slices import _evolve_component_tracked  # noqa: E402
from latent_theta_evolve_suite import _stratified_to_n  # noqa: E402
from score_residual_f0_kinetics import _a2_rd, _load_parts  # noqa: E402

PAPER = ROOT / "papers/mnras_noneq_ics/figures"
GATES = ROOT / "runs/ml/field_maps/latent_theta_evolve_match_2026-08-03/gates"
BAR = ROOT / "runs/ml/field_maps/latent_theta_bar_sweep_2026-08-03"
RES = ROOT / "papers/mnras_noneq_ics/results"


def _evolve_arm(parts: dict, *, end_gyr: float, force: str, omp: int, n_track: int) -> dict:
    snaps = [0.0, 0.25, 0.5, 1.0, float(end_gyr)]
    snaps = sorted({round(s, 4) for s in snaps if s <= end_gyr + 1e-9})
    return _evolve_component_tracked(
        parts,
        end_gyr=float(end_gyr),
        dt=0.01,
        omp=int(omp),
        timeout_s=1e9,
        n_track=int(n_track),
        snap_times=snaps,
        force=force,
        a2_r_eval=2.0,
    )


def _plot_pair(
    *,
    out_png: Path,
    title: str,
    series: list[tuple[str, list[float], list[float], str, str]],
    ylim: tuple[float, float] = (0.0, 0.5),
) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for lab, t, a2, color, ls in series:
        ax.plot(t, a2, color=color, ls=ls, marker="o", ms=4, lw=1.8, label=lab)
    ax.set_xlabel(r"$t$ [Gyr]")
    ax.set_ylabel(r"disk $A_2(R_d)$ (disk COM)")
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def reevolve_gate(
    gate_name: str,
    *,
    paper_png: Path,
    title: str,
    end_gyr: float = 2.0,
    force: str = "gpu_bh",
    omp: int = 2,
) -> dict:
    gate = GATES / gate_name
    out = GATES / f"{gate_name}_diskcom_reevolve"
    out.mkdir(parents=True, exist_ok=True)
    rows = {}
    series = []
    for arm, lab, color, ls in (
        ("data", "data dump", "0.35", "--"),
        ("latent_gen", "latent gen", "C0", "-"),
    ):
        src = gate / f"particles_{arm}_t0.npz"
        print(f"=== {gate_name} {arm} from {src.name} ===", flush=True)
        parts = _load_parts(src)
        if "eps" not in parts:
            parts["eps"] = np.full(len(parts["pos"]), 0.05)
        print(
            f"  t0 disk-COM A2={_a2_rd(parts, 2.0):.3f} N={len(parts['pos'])}",
            flush=True,
        )
        evo = _evolve_arm(parts, end_gyr=end_gyr, force=force, omp=omp, n_track=11)
        rows[arm] = {
            "a2_pre": float(evo["a2_t"][0]),
            "a2_post": float(evo["a2_t"][-1]),
            "t_gyr": [float(x) for x in evo["t_gyr"]],
            "a2_t": [float(x) for x in evo["a2_t"]],
            "force": evo.get("force"),
            "wall_s": evo.get("wall_s"),
        }
        series.append((lab, rows[arm]["t_gyr"], rows[arm]["a2_t"], color, ls))
        print(
            f"  → A2 {rows[arm]['a2_pre']:.3f}→{rows[arm]['a2_post']:.3f} "
            f"force={rows[arm]['force']} wall={rows[arm]['wall_s']:.1f}s",
            flush=True,
        )
    local = out / "a2_t.png"
    _plot_pair(out_png=local, title=title, series=series)
    _plot_pair(out_png=paper_png, title=title, series=series)
    (out / "verdict.json").write_text(json.dumps(rows, indent=2))
    return rows


def reevolve_moderate(
    *,
    force: str = "gpu_bh",
    omp: int = 2,
    end_gyr: float = 0.75,
) -> dict:
    src = BAR / "samples/906c4_moderate.npz"
    print(f"=== moderate bar sweep from {src} ===", flush=True)
    parts = _load_parts(src)
    # Match corpus evolve mix disk:halo:bulge ≈ 4:2:1 with n_disk=1e6.
    n_disk = 1_000_000
    n_tot = int(round(n_disk / 4 * 7))
    rng = np.random.default_rng(0)
    ev = _stratified_to_n(parts, n_tot, rng)
    print(f"  t0 A2={_a2_rd(ev, 2.0):.3f} N={len(ev['pos'])}", flush=True)
    evo = _evolve_arm(ev, end_gyr=end_gyr, force=force, omp=omp, n_track=9)
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    ax.plot(evo["t_gyr"], evo["a2_t"], "C0-o", lw=2.0, ms=5)
    ax.axhspan(0.15, 0.30, color="C0", alpha=0.10, label="moderate band")
    ax.axhline(0.35, color="0.45", ls=":", lw=1.0, label="strong floor")
    ax.set_xlabel(r"$t$ [Gyr]")
    ax.set_ylabel(r"$A_2(R_d)$ (disk COM)")
    ax.set_title(r"Moderate-bar evolve (906c4) — stay mid, not whole-disk")
    ax.set_ylim(0, max(0.55, float(np.nanmax(evo["a2_t"])) * 1.2))
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    local = BAR / "figs/906c4_moderate_evolve_a2_t.png"
    local.parent.mkdir(parents=True, exist_ok=True)
    paper = PAPER / "fig_latent_theta_bar_sweep_moderate_evolve_a2_t.png"
    fig.savefig(local, dpi=150)
    fig.savefig(paper, dpi=150)
    plt.close(fig)
    meta = {
        "a2_pre": float(evo["a2_t"][0]),
        "a2_post": float(evo["a2_t"][-1]),
        "t_gyr": [float(x) for x in evo["t_gyr"]],
        "a2_t": [float(x) for x in evo["a2_t"]],
        "force": evo.get("force"),
        "wall_s": evo.get("wall_s"),
        "fig": str(paper),
    }
    print(
        f"  → moderate A2 {meta['a2_pre']:.3f}→{meta['a2_post']:.3f} "
        f"force={meta['force']}",
        flush=True,
    )
    (BAR / "moderate_evolve_diskcom.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    force = "gpu_bh"
    omp = 2
    r906 = reevolve_gate(
        "evolve_2gyr_match_906c4",
        paper_png=PAPER / "fig_latent_theta_evolve_loo_path_906c4_a2_t.png",
        title=r"906c4 path-LOO evolve — $A_2(R_d)$ (disk COM)",
        force=force,
        omp=omp,
    )
    rnob = reevolve_gate(
        "evolve_2gyr_match_nobulge",
        paper_png=PAPER / "fig_latent_theta_evolve_nobulge_a2_t.png",
        title=r"no-bulge evolve — $A_2(R_d)$ (disk COM)",
        force=force,
        omp=omp,
    )
    rmod = reevolve_moderate(force=force, omp=omp)
    summary = {"906c4": r906, "nobulge": rnob, "moderate": rmod}
    out = RES / "latent_theta_a2_diskcom_reevolve.json"
    out.write_text(json.dumps(summary, indent=2))
    print("=== DONE ===", json.dumps({
        "906c4_data": f"{r906['data']['a2_pre']:.3f}→{r906['data']['a2_post']:.3f}",
        "906c4_gen": f"{r906['latent_gen']['a2_pre']:.3f}→{r906['latent_gen']['a2_post']:.3f}",
        "nobulge_data": f"{rnob['data']['a2_pre']:.3f}→{rnob['data']['a2_post']:.3f}",
        "nobulge_gen": f"{rnob['latent_gen']['a2_pre']:.3f}→{rnob['latent_gen']['a2_post']:.3f}",
        "moderate": f"{rmod['a2_pre']:.3f}→{rmod['a2_post']:.3f}",
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
