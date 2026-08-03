#!/usr/bin/env python3
"""Fixed-θ bar-strength sweep via A₂-conditioned library retrieve+decode.

At fixed structural θ, retrieve same-campaign corpus dumps nearest target
``A₂(R_d)`` tiers (disk-COM recenter; ``disk_azimuthal_fourier(recenter=True)``)
and remass to GalactICS ``f₀(θ)`` (``particle_retrieve``).
Not free continuous ``decode(z)``; not eval-dump particle copy.

Reject dipole-like (A₁-dominated) dumps for mild/moderate/strong: require
``A₁(R_d) < A₂(R_d)`` after disk-COM recenter. Quiet allows noise-floor
axisymmetry. Prefer compact ``R_bar/R_d`` for mid tiers; for the top tier,
take the strongest *true* bar available (do not fake 0.5 with COM offset).

Recentered tiers (disk-COM A₂(R_d), R_d=2 kpc)::

    quiet     ≲ 0.05     (target ≈ 0.02)
    mild      ~ 0.06–0.14 (target ≈ 0.10)
    moderate  ~ 0.22–0.38 (target ≈ 0.30; may be corpus-limited)
    strong    ~ 0.40–0.60 (target ≈ 0.50; take max true bar if unreachable)

Paper figs::

    fig_latent_theta_bar_sweep_faceon.png
    fig_latent_theta_bar_sweep_edgeon.png
    fig_latent_theta_bar_sweep_faceon_edgeon.png
    fig_latent_theta_bar_sweep_a2.png
    fig_latent_theta_bar_sweep_vs_f0.png
    fig_latent_theta_bar_sweep_evolve_a2_t.png
    fig_latent_theta_bar_sweep_evolve_faceon.png

Replot dens (face-on + edge-on) from saved samples without re-retrieve/evolve::

    .venv/bin/python scripts/latent_theta_bar_sweep.py --replot-from-samples \\
      --out runs/ml/field_maps/latent_theta_bar_sweep_2026-08-03

Example::

    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \\
      .venv/bin/python scripts/latent_theta_bar_sweep.py \\
        --out runs/ml/field_maps/latent_theta_bar_sweep_2026-08-03 \\
        --hashes 906c4af73543 \\
        --targets 0.02,0.10,0.30,0.50 \\
        --n-resample 0 --n-disk 1000000 --evolve-gyr 0.5 --force gpu_bh
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from galacticsics.campaign.analysis import dens_array_log10  # noqa: E402
from galacticsics.ml.fields.feature_library import _hash_from_path  # noqa: E402
from latent_theta_evolve_suite import _stratified_to_n  # noqa: E402
from latent_theta_gen import (  # noqa: E402
    _component_masses,
    _decode_particle_retrieve,
    _disk_faceon,
    _ic_path_for_dump,
    _load_parts,
)
from ntropy.analysis.disk_density import disk_azimuthal_fourier  # noqa: E402
from ood_theta_df_compare import _disk_kinematics  # noqa: E402
from score_residual_f0_kinetics import _a2_rd  # noqa: E402

RANK = Path("runs/ml/field_maps/corpus_particle_a2_rank.json")
PAPER = Path("papers/mnras_noneq_ics/figures")
RESULTS = Path("papers/mnras_noneq_ics/results")

# Recentered disk-COM A₂(R_d) tiers; reject A₁≳A₂ dipoles for non-quiet.
DEFAULT_TIERS: list[dict] = [
    {"name": "quiet", "target_a2": 0.02, "lo": 0.0, "hi": 0.05},
    {"name": "mild", "target_a2": 0.10, "lo": 0.06, "hi": 0.14},
    {"name": "moderate", "target_a2": 0.30, "lo": 0.22, "hi": 0.38},
    {"name": "strong", "target_a2": 0.50, "lo": 0.40, "hi": 0.60},
]
DEFAULT_TARGETS = "0.02,0.10,0.30,0.50"
DEFAULT_BANDS = [(0.0, 0.05), (0.06, 0.14), (0.22, 0.38), (0.40, 0.60)]

DEFAULT_HASHES = ("906c4af73543", "ac023abb258d")
HASH_LABEL = {
    "906c4af73543": "906c4 (disk+halo+bulge)",
    "ac023abb258d": "no-bulge (disk+halo)",
}
TIER_COLORS = {
    "quiet": "0.45",
    "mild": "C0",
    "moderate": "C1",
    "strong": "C3",
}


def _a_m_at_rd(parts: dict, *, m: int, rd: float) -> float:
    disk = parts["component_id"] == 0
    fout = disk_azimuthal_fourier(
        parts["pos"][disk],
        parts["mass"][disk],
        m=int(m),
        r_max=12.0,
        n_bins=12,
        z_max=0.5,
        min_count=10,
        r_eval=float(rd),
    )
    return float(fout["a_m_over_a0_at_r"])


def _a2_profile(parts: dict, *, r_max: float = 12.0, n_bins: int = 24) -> dict:
    disk = parts["component_id"] == 0
    fout = disk_azimuthal_fourier(
        parts["pos"][disk],
        parts["mass"][disk],
        m=2,
        r_max=float(r_max),
        n_bins=int(n_bins),
        z_max=0.5,
        min_count=10,
    )
    r = np.asarray(fout["r_mid"], dtype=np.float64)
    a2 = np.asarray(fout["a_m_over_a0"], dtype=np.float64)
    return {"r_mid": r, "a2": a2, "a2_med": float(fout["a_m_over_a0_median"])}


def _bar_length_rd(
    r_mid: np.ndarray,
    a2: np.ndarray,
    *,
    rd: float,
    floor: float = 0.10,
    frac_peak: float = 0.5,
    r_peak_max: float = 6.0,
) -> tuple[float, float]:
    ok = np.isfinite(a2) & np.isfinite(r_mid) & (r_mid > 0)
    if not np.any(ok):
        return float("nan"), float("nan")
    rr, aa = r_mid[ok], a2[ok]
    inner = rr <= float(r_peak_max)
    if not np.any(inner):
        inner = np.ones_like(rr, dtype=bool)
    ipeak = int(np.where(inner)[0][int(np.nanargmax(aa[inner]))])
    peak = float(aa[ipeak])
    r_peak = float(rr[ipeak])
    if peak < floor:
        return 0.0, r_peak
    thr = max(floor, frac_peak * peak)
    r_bar = r_peak
    for i in range(ipeak, len(rr)):
        if aa[i] >= thr:
            r_bar = float(rr[i])
        else:
            break
    return r_bar / max(float(rd), 1e-6), r_peak


def _measure_dump_a2(path: Path, *, rd: float) -> dict:
    parts = _load_parts(path, None, np.random.default_rng(0))
    a2_rd = _a2_rd(parts, float(rd))
    a1_rd = _a_m_at_rd(parts, m=1, rd=float(rd))
    prof = _a2_profile(parts)
    bl, r_peak = _bar_length_rd(prof["r_mid"], prof["a2"], rd=float(rd))
    return {
        "path": str(path),
        "a2_rd": float(a2_rd),
        "a1_rd": float(a1_rd),
        "a2_med": float(prof["a2_med"]),
        "bar_len_rd": float(bl),
        "r_peak": float(r_peak),
        "true_bar": bool(np.isfinite(a1_rd) and np.isfinite(a2_rd) and a1_rd < a2_rd),
        "a2_profile": {
            "r_mid": prof["r_mid"].tolist(),
            "a2": prof["a2"].tolist(),
        },
    }


def _cache_campaign_a2rd(
    ranked: list[dict],
    *,
    run_hash: str,
    rd: float,
    cache_path: Path,
    t_min: float = 0.0,
) -> list[dict]:
    cache_tag = {"rd": float(rd), "a2_recenter": "disk_com", "a1_gate": "a1_lt_a2"}
    if cache_path.is_file():
        rows = json.loads(cache_path.read_text())
        if (
            rows
            and abs(float(rows[0].get("rd", rd)) - float(rd)) < 1e-9
            and rows[0].get("a2_recenter") == cache_tag["a2_recenter"]
            and rows[0].get("a1_gate") == cache_tag["a1_gate"]
            and "a1_rd" in rows[0]
        ):
            return rows
    out: list[dict] = []
    for row in ranked:
        h = row.get("run_hash") or _hash_from_path(row["path"])
        if h != run_hash:
            continue
        p = Path(row["path"])
        if "step_" not in p.name or not p.is_file():
            continue
        t = float(row.get("t_gyr") or 0.0)
        if t < t_min:
            continue
        print(f"  measure A₁/A₂(R_d) {p.name} …", flush=True)
        m = _measure_dump_a2(p, rd=rd)
        m.update(
            {
                "run_hash": run_hash,
                "t_gyr": t,
                "rank_a2": float(row["a2"]),
                "rd": float(rd),
                "a2_recenter": cache_tag["a2_recenter"],
                "a1_gate": cache_tag["a1_gate"],
            }
        )
        out.append(m)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out, indent=2) + "\n")
    return out


def _is_true_bar(m: dict, *, tier: str) -> bool:
    """Quiet: any dump. Else require disk-COM A₁(R_d) < A₂(R_d)."""
    if tier == "quiet":
        return True
    a1 = float(m.get("a1_rd", float("nan")))
    a2 = float(m.get("a2_rd", float("nan")))
    if not (np.isfinite(a1) and np.isfinite(a2)):
        return False
    return bool(a1 < a2)


def _pick_a2_neighbor(
    measured: list[dict],
    *,
    target_a2: float,
    tier: str,
    prefer_compact: bool = True,
    take_max_true: bool = False,
) -> dict:
    if not measured:
        raise RuntimeError("empty measured A₂(R_d) list")
    pool = [m for m in measured if _is_true_bar(m, tier=tier)]
    if not pool:
        raise RuntimeError(f"no true-bar (A₁<A₂) candidates for tier={tier}")
    if take_max_true or (tier == "strong" and float(target_a2) >= 0.45):
        # Strongest true m=2 bar; never invent 0.5 via COM-offset dipole.
        return max(pool, key=lambda m: float(m["a2_rd"]))
    scored: list[tuple[float, dict]] = []
    for m in pool:
        da = abs(float(m["a2_rd"]) - float(target_a2))
        pen = 0.0
        a1 = float(m.get("a1_rd", 0.0))
        a2 = float(m["a2_rd"])
        # Prefer clearer m=2 dominance (lower A1/A2).
        if a2 > 1e-6 and tier != "quiet":
            pen += 0.08 * max(0.0, a1 / a2)
        if prefer_compact and tier in ("mild", "moderate"):
            bl = float(m.get("bar_len_rd") or 0.0)
            rp = float(m.get("r_peak") or 0.0)
            if bl > 3.0:
                pen += 0.15 * (bl - 3.0)
            if rp > 4.0:
                pen += 0.05 * (rp - 4.0)
        scored.append((da + pen, m))
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def _pick_tiers(
    measured: list[dict],
    tiers: list[dict],
) -> list[tuple[dict, dict]]:
    """Pick distinct dumps; strong first so mid tiers do not steal the max bar."""
    used: set[str] = set()
    by_name: dict[str, dict] = {}
    ordered = sorted(tiers, key=lambda t: -float(t["target_a2"]))
    for tier in ordered:
        name = str(tier["name"])
        avail = [m for m in measured if str(m["path"]) not in used]
        take_max = name == "strong" or float(tier["target_a2"]) >= 0.45
        # If target is above corpus true-bar max, take max true bar and label later.
        true_pool = [m for m in avail if _is_true_bar(m, tier=name)]
        corpus_max = max((float(m["a2_rd"]) for m in true_pool), default=float("nan"))
        if (
            name != "quiet"
            and np.isfinite(corpus_max)
            and float(tier["target_a2"]) > corpus_max + 0.02
        ):
            take_max = True
        nn = _pick_a2_neighbor(
            avail,
            target_a2=float(tier["target_a2"]),
            tier=name,
            prefer_compact=True,
            take_max_true=take_max,
        )
        used.add(str(nn["path"]))
        by_name[name] = nn
    return [(t, by_name[str(t["name"])]) for t in tiers]


# Expanded dens FOV (match zoom-out ±15–20 kpc); edge-on uses thinner z span.
SWEEP_HALF_XY = 16.0
SWEEP_HALF_Z = 6.0
SWEEP_NBIN = 128
# Evolve snap maps default disk half in evolve_component_slices.COMPONENTS.
EVOLVE_SNAP_HALF = 16.0


def _cbar_beside(ax, mappable, *, label: str, width: float = 0.018, pad: float = 0.01):
    """Colorbar matched to one axes height in figure coords (label not clipped)."""
    fig = ax.figure
    fig.canvas.draw()
    bbox = ax.get_position()
    cax = fig.add_axes(
        [bbox.x1 + pad, bbox.y0, width, bbox.height]
    )
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(label)
    return cbar


def _cbar_span(ax_top, ax_bot, mappable, *, label: str, width: float = 0.018, pad: float = 0.01):
    """Colorbar spanning the vertical range from ax_bot to ax_top (same column)."""
    fig = ax_top.figure
    fig.canvas.draw()
    bt = ax_top.get_position()
    bb = ax_bot.get_position()
    cax = fig.add_axes([bb.x1 + pad, bb.y0, width, bt.y1 - bb.y0])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(label)
    return cbar


def _tier_panel_title(row: dict) -> str:
    """Honest tier label: corpus-max true bar when target is unreachable."""
    tier = str(row.get("tier", ""))
    tgt = float(row.get("target_a2", float("nan")))
    a2 = float(row.get("a2_rd", float("nan")))
    reachable = row.get("target_reachable")
    if reachable is None and np.isfinite(a2) and np.isfinite(tgt):
        reachable = a2 >= tgt - 0.03
    if reachable is False or (
        np.isfinite(a2) and np.isfinite(tgt) and a2 < tgt - 0.03 and tier != "quiet"
    ):
        if tier == "strong" or (np.isfinite(tgt) and tgt >= 0.45):
            return f"{tier}\n(corpus-max true bar)"
        return f"{tier}\n(corpus-limited; tgt {tgt:.2f})"
    return f"{tier}\n(target {tgt:.2f})"


def _disk_com_center(parts: dict) -> dict:
    disk = parts["component_id"] == 0
    com = np.average(parts["pos"][disk], axis=0, weights=parts["mass"][disk])
    return {**parts, "pos": np.asarray(parts["pos"], dtype=float) - com}


def _bar_pa_rad(parts: dict, *, rd: float) -> float:
    """Mass-weighted m=2 phase → bar major-axis PA (disk-COM frame)."""
    disk = parts["component_id"] == 0
    pos = np.asarray(parts["pos"][disk], dtype=float)
    mass = np.asarray(parts["mass"][disk], dtype=float)
    if pos.size == 0:
        return 0.0
    r = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    z = pos[:, 2]
    r_lo, r_hi = 0.5 * float(rd), 4.0 * float(rd)
    msk = (r >= r_lo) & (r <= r_hi) & (np.abs(z) < 0.5)
    if int(msk.sum()) < 200:
        msk = (r >= r_lo) & (r <= r_hi)
    if int(msk.sum()) < 50:
        return 0.0
    phi = np.arctan2(pos[msk, 1], pos[msk, 0])
    c = np.sum(mass[msk] * np.exp(2j * phi))
    if not np.isfinite(c) or abs(c) < 1e-30:
        return 0.0
    return 0.5 * float(np.angle(c))


def _rotate_z(parts: dict, pa: float) -> dict:
    """Rotate positions (and velocities) by −PA about z so bar → x-axis."""
    c, s = float(np.cos(-pa)), float(np.sin(-pa))
    pos = np.asarray(parts["pos"], dtype=float).copy()
    x, y = pos[:, 0].copy(), pos[:, 1].copy()
    pos[:, 0] = c * x - s * y
    pos[:, 1] = s * x + c * y
    out = {**parts, "pos": pos}
    if "vel" in parts:
        vel = np.asarray(parts["vel"], dtype=float).copy()
        vx, vy = vel[:, 0].copy(), vel[:, 1].copy()
        vel[:, 0] = c * vx - s * vy
        vel[:, 1] = s * vx + c * vy
        out["vel"] = vel
    return out


def _disk_edgeon(
    parts: dict,
    *,
    nbin: int = SWEEP_NBIN,
    half_x: float = SWEEP_HALF_XY,
    half_z: float = SWEEP_HALF_Z,
) -> np.ndarray:
    """Projected Σ(x, z) for disk particles (histogram mass)."""
    m = parts["component_id"] == 0
    H, _, _ = np.histogram2d(
        parts["pos"][m, 0],
        parts["pos"][m, 2],
        bins=nbin,
        range=[[-half_x, half_x], [-half_z, half_z]],
        weights=parts["mass"][m],
    )
    return H


def _shared_log_limits(imgs: list[np.ndarray]) -> tuple[float, float]:
    stack = np.concatenate([np.asarray(im).ravel() for im in imgs])
    pos = stack[np.isfinite(stack) & (stack > 0)]
    if pos.size:
        floor = max(
            float(np.percentile(pos, 20)) * 1e-2,
            float(np.percentile(pos, 99.0)) * 1e-4,
            1e-12,
        )
        vmax = float(np.percentile(pos, 98))
    else:
        floor, vmax = 1e-12, 1.0
    return floor, vmax


def _faceon_panel(ax, parts: dict, title: str, *, rd: float, half: float = SWEEP_HALF_XY) -> float:
    centered = _disk_com_center(parts)
    img = _disk_faceon(centered, nbin=SWEEP_NBIN, half=half)
    show, vmin_s, vmax_s, _ = dens_array_log10(img, vmax_pct=98.0)
    ax.imshow(
        show.T,
        origin="lower",
        cmap="magma",
        vmin=vmin_s,
        vmax=vmax_s,
        extent=[-half, half, -half, half],
    )
    a2 = _a2_rd(parts, float(rd))
    a1 = _a_m_at_rd(parts, m=1, rd=float(rd))
    ax.set_title(title, fontsize=10)
    ax.text(
        0.02,
        0.98,
        rf"$A_2(R_d)={a2:.3f}$" "\n" rf"$A_1(R_d)={a1:.3f}$",
        transform=ax.transAxes,
        va="top",
        color="w",
        fontsize=8,
    )
    th = np.linspace(0, 2 * np.pi, 128)
    ax.plot(rd * np.cos(th), rd * np.sin(th), "w--", lw=0.7, alpha=0.55)
    ax.set_xlabel(r"$x$ [kpc]")
    ax.set_ylabel(r"$y$ [kpc]")
    ax.set_aspect("equal")
    return a2


def _faceon_from_snap_map(
    ax,
    img: np.ndarray,
    title: str,
    *,
    half: float = SWEEP_HALF_XY,
    floor: float | None = None,
    vmax: float | None = None,
):
    """Plot log10(Σ+ε) face-on; return (AxesImage, floor, vmax)."""
    show, vmin_s, vmax_s, floor_used = dens_array_log10(
        img, floor=floor, vmax=vmax, vmax_pct=98.0
    )
    im = ax.imshow(
        show.T,
        origin="lower",
        cmap="magma",
        vmin=vmin_s,
        vmax=vmax_s,
        extent=[-half, half, -half, half],
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(r"$x$ [kpc]")
    ax.set_ylabel(r"$y$ [kpc]")
    ax.set_aspect("equal")
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    pos = np.asarray(img)[np.isfinite(img) & (np.asarray(img) > 0)]
    vmax_used = float(np.percentile(pos, 98)) if pos.size else float(vmax or 1.0)
    return im, float(floor_used), float(max(vmax_used if vmax is None else vmax, vmax_used))


def _faceon_delta_map(
    ax,
    img0: np.ndarray,
    img1: np.ndarray,
    title: str,
    *,
    half: float = SWEEP_HALF_XY,
    floor: float,
    lim: float | None = None,
):
    """Signed Δ log10(Σ+ε) so mild 0.5 Gyr evolution is visible."""
    eps = max(float(floor), 1e-30)
    a0 = np.log10(np.maximum(np.asarray(img0, dtype=np.float64), 0.0) + eps)
    a1 = np.log10(np.maximum(np.asarray(img1, dtype=np.float64), 0.0) + eps)
    d = a1 - a0
    if lim is None:
        lim = (
            float(np.percentile(np.abs(d[np.isfinite(d)]), 99))
            if np.any(np.isfinite(d))
            else 1.0
        )
        lim = max(lim, 1e-3)
    im = ax.imshow(
        d.T,
        origin="lower",
        cmap="coolwarm",
        vmin=-lim,
        vmax=lim,
        extent=[-half, half, -half, half],
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(r"$x$ [kpc]")
    ax.set_ylabel(r"$y$ [kpc]")
    ax.set_aspect("equal")
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    return im, float(lim)


def plot_sweep_faceon(
    rows: list[dict],
    out_png: Path,
    *,
    theta_label: str,
    rd: float,
    half: float = SWEEP_HALF_XY,
) -> None:
    n = len(rows)
    fig, axes = plt.subplots(
        1, n, figsize=(2.85 * n + 1.0, 3.45)
    )
    if n == 1:
        axes = [axes]
    imgs = []
    for row in rows:
        centered = _disk_com_center(row["parts"])
        imgs.append(_disk_faceon(centered, nbin=SWEEP_NBIN, half=half))
    floor, vmax = _shared_log_limits(imgs)
    last_im = None
    for j, (ax, row, img) in enumerate(zip(axes, rows, imgs)):
        show, vmin_s, vmax_s, _ = dens_array_log10(img, floor=floor, vmax=vmax)
        last_im = ax.imshow(
            show.T,
            origin="lower",
            cmap="magma",
            vmin=vmin_s,
            vmax=vmax_s,
            extent=[-half, half, -half, half],
        )
        a2 = _a2_rd(row["parts"], float(rd))
        a1 = _a_m_at_rd(row["parts"], m=1, rd=float(rd))
        # Prefer stored a2_rd for title reachability when present.
        title_row = {**row, "a2_rd": a2}
        ax.set_title(_tier_panel_title(title_row), fontsize=10)
        ax.text(
            0.02,
            0.98,
            rf"$A_2(R_d)={a2:.3f}$" "\n" rf"$A_1(R_d)={a1:.3f}$",
            transform=ax.transAxes,
            va="top",
            color="w",
            fontsize=8,
        )
        th = np.linspace(0, 2 * np.pi, 128)
        ax.plot(rd * np.cos(th), rd * np.sin(th), "w--", lw=0.7, alpha=0.55)
        ax.set_xlabel(r"$x$ [kpc]")
        if j == 0:
            ax.set_ylabel(r"$y$ [kpc]")
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        ax.set_aspect("equal")
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
    fig.suptitle(
        rf"Fixed-θ bar-strength sweep — {theta_label}"
        "\n"
        r"disk-COM $A_2$; retrieve + particle_retrieve → $f_0(\theta)$"
        rf" (face-on $\log_{{10}}\Sigma$, $\pm{half:g}$ kpc)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 0.92, 0.90])
    if last_im is not None:
        _cbar_beside(axes[-1], last_im, label=r"$\log_{10}\Sigma$")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_sweep_edgeon(
    rows: list[dict],
    out_png: Path,
    *,
    theta_label: str,
    rd: float,
    half_x: float = SWEEP_HALF_XY,
    half_z: float = SWEEP_HALF_Z,
) -> None:
    """Edge-on Σ(x,z) with bar major axis rotated into the sky plane (along x)."""
    n = len(rows)
    fig, axes = plt.subplots(
        1, n, figsize=(2.85 * n + 1.0, 2.65)
    )
    if n == 1:
        axes = [axes]
    imgs = []
    pas: list[float] = []
    for row in rows:
        centered = _disk_com_center(row["parts"])
        pa = _bar_pa_rad(centered, rd=float(rd))
        pas.append(pa)
        aligned = _rotate_z(centered, pa)
        imgs.append(
            _disk_edgeon(aligned, nbin=SWEEP_NBIN, half_x=half_x, half_z=half_z)
        )
    floor, vmax = _shared_log_limits(imgs)
    last_im = None
    for j, (ax, row, img, pa) in enumerate(zip(axes, rows, imgs, pas)):
        show, vmin_s, vmax_s, _ = dens_array_log10(img, floor=floor, vmax=vmax)
        last_im = ax.imshow(
            show.T,
            origin="lower",
            cmap="magma",
            vmin=vmin_s,
            vmax=vmax_s,
            extent=[-half_x, half_x, -half_z, half_z],
            aspect="auto",
        )
        a2 = _a2_rd(row["parts"], float(rd))
        ax.set_title(_tier_panel_title(row), fontsize=10)
        ax.text(
            0.02,
            0.98,
            rf"$A_2={a2:.3f}$" "\n" rf"PA$={np.degrees(pa):.0f}^\circ$",
            transform=ax.transAxes,
            va="top",
            color="w",
            fontsize=8,
        )
        ax.axhline(0.0, color="w", lw=0.4, alpha=0.35, ls=":")
        ax.set_xlabel(r"$x$ [kpc] (bar $\parallel x$)")
        if j == 0:
            ax.set_ylabel(r"$z$ [kpc]")
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
    fig.suptitle(
        rf"Fixed-θ bar-strength sweep — {theta_label}"
        "\n"
        r"edge-on $\log_{10}\Sigma(x,z)$ after disk-COM; bar PA $\to x$"
        rf" ($\pm{half_x:g}\times\pm{half_z:g}$ kpc)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 0.92, 0.88])
    if last_im is not None:
        _cbar_beside(axes[-1], last_im, label=r"$\log_{10}\Sigma$")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_sweep_faceon_edgeon(
    rows: list[dict],
    out_png: Path,
    *,
    theta_label: str,
    rd: float,
    half_xy: float = SWEEP_HALF_XY,
    half_z: float = SWEEP_HALF_Z,
) -> None:
    """Two-row gallery: face-on (row 0) + bar-aligned edge-on (row 1)."""
    n = len(rows)
    fig, axes = plt.subplots(
        2,
        n,
        figsize=(2.85 * n, 5.4),
        gridspec_kw={"height_ratios": [1.0, 0.72], "hspace": 0.28, "wspace": 0.18},
    )
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    face_imgs = []
    edge_imgs = []
    pas: list[float] = []
    for row in rows:
        centered = _disk_com_center(row["parts"])
        face_imgs.append(_disk_faceon(centered, nbin=SWEEP_NBIN, half=half_xy))
        pa = _bar_pa_rad(centered, rd=float(rd))
        pas.append(pa)
        aligned = _rotate_z(centered, pa)
        edge_imgs.append(
            _disk_edgeon(aligned, nbin=SWEEP_NBIN, half_x=half_xy, half_z=half_z)
        )

    # Shared colorbars per row (fixed; no hanging).
    face_floor, face_vmax = _shared_log_limits(face_imgs)
    edge_floor, edge_vmax = _shared_log_limits(edge_imgs)
    last_face = last_edge = None
    for j, row in enumerate(rows):
        a2 = _a2_rd(row["parts"], float(rd))
        a1 = _a_m_at_rd(row["parts"], m=1, rd=float(rd))
        show_f, vmin_f, vmax_f, _ = dens_array_log10(
            face_imgs[j], floor=face_floor, vmax=face_vmax
        )
        last_face = axes[0, j].imshow(
            show_f.T,
            origin="lower",
            cmap="magma",
            vmin=vmin_f,
            vmax=vmax_f,
            extent=[-half_xy, half_xy, -half_xy, half_xy],
        )
        axes[0, j].set_title(_tier_panel_title(row), fontsize=10)
        axes[0, j].text(
            0.02,
            0.98,
            rf"$A_2={a2:.3f}$" "\n" rf"$A_1={a1:.3f}$",
            transform=axes[0, j].transAxes,
            va="top",
            color="w",
            fontsize=8,
        )
        th = np.linspace(0, 2 * np.pi, 128)
        axes[0, j].plot(
            rd * np.cos(th), rd * np.sin(th), "w--", lw=0.7, alpha=0.55
        )
        axes[0, j].set_xlabel(r"$x$ [kpc]")
        axes[0, j].set_ylabel(r"$y$ [kpc]" if j == 0 else "")
        axes[0, j].set_aspect("equal")

        show_e, vmin_e, vmax_e, _ = dens_array_log10(
            edge_imgs[j], floor=edge_floor, vmax=edge_vmax
        )
        last_edge = axes[1, j].imshow(
            show_e.T,
            origin="lower",
            cmap="magma",
            vmin=vmin_e,
            vmax=vmax_e,
            extent=[-half_xy, half_xy, -half_z, half_z],
            aspect="auto",
        )
        axes[1, j].text(
            0.02,
            0.98,
            rf"PA$={np.degrees(pas[j]):.0f}^\circ$",
            transform=axes[1, j].transAxes,
            va="top",
            color="w",
            fontsize=8,
        )
        axes[1, j].axhline(0.0, color="w", lw=0.4, alpha=0.35, ls=":")
        axes[1, j].set_xlabel(r"$x$ [kpc] (bar $\parallel x$)")
        axes[1, j].set_ylabel(r"$z$ [kpc]" if j == 0 else "")

    axes[0, 0].annotate(
        "face-on",
        xy=(-0.22, 0.5),
        xycoords="axes fraction",
        rotation=90,
        va="center",
        ha="right",
        fontsize=10,
    )
    axes[1, 0].annotate(
        "edge-on",
        xy=(-0.22, 0.5),
        xycoords="axes fraction",
        rotation=90,
        va="center",
        ha="right",
        fontsize=10,
    )
    fig.suptitle(
        rf"Fixed-θ bar-strength sweep — {theta_label}"
        "\n"
        r"disk-COM dens $\log_{10}\Sigma$; edge-on bar PA$\to x$"
        rf" (FOV $\pm{half_xy:g}$ / $\pm{half_z:g}$ kpc)",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.88, left=0.08, right=0.88, bottom=0.08)
    if last_face is not None:
        _cbar_beside(axes[0, -1], last_face, label=r"$\log_{10}\Sigma$")
    if last_edge is not None:
        _cbar_beside(axes[1, -1], last_edge, label=r"$\log_{10}\Sigma$")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_sweep_a2(
    rows: list[dict],
    out_png: Path,
    *,
    theta_label: str,
    rd: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    ax = axes[0]
    for row in rows:
        c = TIER_COLORS.get(row["tier"], "C0")
        prof = row["a2_profile"]
        ax.plot(
            prof["r_mid"],
            prof["a2"],
            color=c,
            lw=2.0,
            label=rf"{row['tier']}  $A_2(R_d)={row['a2_rd']:.3f}$",
        )
    ax.axvline(rd, color="0.4", ls="--", lw=1.0, label=rf"$R_d={rd:g}$")
    ax.axhspan(0.06, 0.14, color="0.5", alpha=0.08, label="mild band")
    ax.set_xlabel(r"$R$ [kpc]")
    ax.set_ylabel(r"$A_2(R)$ (disk COM)")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, None)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.set_title(r"$A_2(R)$ by strength tier")

    ax = axes[1]
    xs = np.arange(len(rows))
    a2s = [row["a2_rd"] for row in rows]
    lbs = [row["bar_len_rd"] for row in rows]
    cols = [TIER_COLORS.get(r["tier"], "C0") for r in rows]
    ax.plot(xs, a2s, "-", color="0.3", lw=1.4, zorder=1)
    ax.scatter(xs, a2s, c=cols, s=55, zorder=2, label=r"$A_2(R_d)$")
    ax2 = ax.twinx()
    ax2.plot(xs, lbs, "s--", color="0.45", lw=1.4, ms=6, label=r"$R_\mathrm{bar}/R_d$")
    ax.set_xticks(xs)
    ax.set_xticklabels([row["tier"] for row in rows])
    ax.set_ylabel(r"$A_2(R_d)$ (disk COM)")
    ax2.set_ylabel(r"$R_\mathrm{bar}/R_d$")
    ax.set_ylim(0, max(0.55, max(a2s) * 1.25))
    ax2.set_ylim(0, max(3.0, max(lbs) * 1.2 if np.isfinite(lbs).any() else 3.0))
    ax.axhspan(0.06, 0.14, color="0.5", alpha=0.08)
    ax.grid(True, alpha=0.25)
    ax.set_title("strength & bar length vs tier")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc="upper left")

    fig.suptitle(rf"Bar-strength diagnostics — {theta_label} (disk COM)", fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_vs_f0(
    rows: list[dict],
    f0: dict,
    out_png: Path,
    *,
    theta_label: str,
    rd: float,
) -> dict:
    """Generative tiers vs best-fit quiet GalactICS f0(θ): face-on + dens/kin."""
    half = SWEEP_HALF_XY
    n = len(rows) + 1
    fig = plt.figure(figsize=(2.55 * n, 7.6))
    gs = fig.add_gridspec(3, n, height_ratios=[1.15, 1.0, 1.0], hspace=0.38, wspace=0.28)

    # Row 0: face-on f0 + tiers
    panels = [("f0 quiet", f0)] + [(r["tier"], r["parts"]) for r in rows]
    for j, (lab, parts) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        _faceon_panel(ax, parts, lab, rd=rd, half=half)
        if j > 0:
            ax.set_ylabel("")

    # Row 1: Σ(R) and A2(R)
    ax_s = fig.add_subplot(gs[1, 0: n // 2])
    ax_a = fig.add_subplot(gs[1, n // 2 :])
    # Row 2: <vφ>, σ_R
    ax_v = fig.add_subplot(gs[2, 0: n // 2])
    ax_sr = fig.add_subplot(gs[2, n // 2 :])

    def _sigma(parts):
        disk = parts["component_id"] == 0
        com = np.average(parts["pos"][disk], axis=0, weights=parts["mass"][disk])
        pos = parts["pos"][disk] - com
        from ntropy.analysis.disk_density import bin_midplane_surface_density

        return bin_midplane_surface_density(pos, parts["mass"][disk], r_max=12.0, n_bins=24, z_max=0.5)

    for lab, parts, c, ls in (
        [("f0 quiet", f0, "0.35", "--")]
        + [
            (r["tier"], r["parts"], TIER_COLORS.get(r["tier"], "C0"), "-")
            for r in rows
        ]
    ):
        prof = _sigma(parts)
        r = np.asarray(prof.r_mid)
        sig = np.asarray(prof.sigma)
        ok = np.asarray(prof.counts) > 0
        ax_s.semilogy(r[ok], np.maximum(sig[ok], 1e-30), color=c, ls=ls, lw=1.6, label=lab)
        a2p = _a2_profile(parts)
        ax_a.plot(a2p["r_mid"], a2p["a2"], color=c, ls=ls, lw=1.6, label=lab)
        k = _disk_kinematics(parts)
        kok = np.asarray(k["counts"]) >= 20
        ax_v.plot(np.asarray(k["r_mid"])[kok], np.asarray(k["mean_vphi"])[kok], color=c, ls=ls, lw=1.6, label=lab)
        ax_sr.plot(np.asarray(k["r_mid"])[kok], np.asarray(k["sig_r"])[kok], color=c, ls=ls, lw=1.6, label=lab)

    ax_s.set_ylabel(r"disk $\Sigma(R)$")
    ax_s.set_xlabel(r"$R$ [kpc]")
    ax_s.legend(fontsize=7, frameon=False, ncol=2)
    ax_s.grid(True, alpha=0.25)
    ax_s.set_title(r"surface density vs quiet $f_0(\theta)$")
    ax_a.axvline(rd, color="0.5", ls=":", lw=0.9)
    ax_a.set_ylabel(r"$A_2(R)$ (disk COM)")
    ax_a.set_xlabel(r"$R$ [kpc]")
    ax_a.set_xlim(0, 12)
    ax_a.grid(True, alpha=0.25)
    ax_a.set_title("non-axisymmetry: residual / non-eq content")
    ax_v.set_ylabel(r"$\langle v_\varphi\rangle$")
    ax_v.set_xlabel(r"$R$ [kpc]")
    ax_v.grid(True, alpha=0.25)
    ax_v.set_title("rotation")
    ax_sr.set_ylabel(r"$\sigma_R$")
    ax_sr.set_xlabel(r"$R$ [kpc]")
    ax_sr.grid(True, alpha=0.25)
    ax_sr.set_title("radial dispersion")

    fig.suptitle(
        rf"Generative bar sweep vs best-fit GalactICS $f_0(\theta)$ — {theta_label}",
        fontsize=11,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    # Scalar metrics: dens/kin residual of strongest tier vs f0.
    strong = rows[-1]["parts"]
    from score_residual_f0_kinetics import _score_vs_ref

    sc = _score_vs_ref(strong, f0)
    return {
        "strong_vs_f0_kin_mse": sc.get("mse", {}),
        "f0_a2_rd": _a2_rd(f0, float(rd)),
        "strong_a2_rd": _a2_rd(strong, float(rd)),
    }


def evolve_all_tiers(
    parts_by_tier: dict[str, dict],
    *,
    out: Path,
    tag: str,
    n_disk: int,
    evolve_gyr: float,
    force: str,
    omp: int,
    a2_r_eval: float,
    paper_a2_png: Path,
    paper_face_png: Path,
) -> dict:
    import evolve_component_slices as ecs
    from evolve_component_slices import _evolve_component_tracked

    # Match sweep face-on FOV when recording disk maps (default COMPONENTS is ±14).
    _old_components = ecs.COMPONENTS
    ecs.COMPONENTS = (
        ("disk", 0, "faceon", float(EVOLVE_SNAP_HALF), max(18.0, float(EVOLVE_SNAP_HALF) + 2.0)),
        _old_components[1],
        _old_components[2],
    )
    try:
        return _evolve_all_tiers_impl(
            parts_by_tier,
            out=out,
            tag=tag,
            n_disk=n_disk,
            evolve_gyr=evolve_gyr,
            force=force,
            omp=omp,
            a2_r_eval=a2_r_eval,
            paper_a2_png=paper_a2_png,
            paper_face_png=paper_face_png,
            _evolve_component_tracked=_evolve_component_tracked,
        )
    finally:
        ecs.COMPONENTS = _old_components


def _evolve_all_tiers_impl(
    parts_by_tier: dict[str, dict],
    *,
    out: Path,
    tag: str,
    n_disk: int,
    evolve_gyr: float,
    force: str,
    omp: int,
    a2_r_eval: float,
    paper_a2_png: Path,
    paper_face_png: Path,
    _evolve_component_tracked,
) -> dict:
    rng = np.random.default_rng(0)
    snaps_want = [0.0, 0.25, float(evolve_gyr)]
    snaps_want = sorted({round(s, 4) for s in snaps_want if s <= evolve_gyr + 1e-9})
    results: dict[str, dict] = {}

    for tier, parts in parts_by_tier.items():
        n_tot = int(round(n_disk / 4 * 7))
        if not np.any(parts["component_id"] == 2):
            n_tot = int(round(n_disk * 1.5))
        ev = _stratified_to_n(parts, n_tot, rng)
        if "eps" not in ev:
            ev["eps"] = np.full(len(ev["pos"]), 0.05)
        print(
            f"=== evolve {tier} {tag} T={evolve_gyr} Gyr N={len(ev['pos'])} "
            f"force={force} ===",
            flush=True,
        )
        out_ev = _evolve_component_tracked(
            ev,
            end_gyr=float(evolve_gyr),
            dt=0.01,
            omp=int(omp),
            timeout_s=1e9,
            n_track=9,
            snap_times=snaps_want,
            force=force,
            a2_r_eval=float(a2_r_eval),
        )
        results[tier] = {
            "a2_pre": float(out_ev["a2_t"][0]),
            "a2_post": float(out_ev["a2_t"][-1]),
            "t_gyr": [float(x) for x in out_ev["t_gyr"]],
            "a2_t": [float(x) for x in out_ev["a2_t"]],
            "force": out_ev.get("force_method") or out_ev.get("force"),
            "wall_s": out_ev.get("wall_s"),
            "snap_t": [float(x) for x in out_ev.get("snap_t_gyr", [])],
            "snaps": out_ev.get("snaps", []),
        }
        print(
            f"  {tier}: A2 {results[tier]['a2_pre']:.3f}→{results[tier]['a2_post']:.3f} "
            f"force={results[tier]['force']}",
            flush=True,
        )

    # Combined A2(t) panel.
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bands = {
        "quiet": (0.0, 0.05),
        "mild": (0.06, 0.14),
        "moderate": (0.22, 0.38),
        "strong": (0.40, 0.60),
    }
    for tier, res in results.items():
        c = TIER_COLORS.get(tier, "C0")
        ax.plot(
            res["t_gyr"],
            res["a2_t"],
            "-o",
            color=c,
            lw=2.0,
            ms=4,
            label=rf"{tier}  {res['a2_pre']:.3f}→{res['a2_post']:.3f}",
        )
        lo, hi = bands.get(tier, (None, None))
        if lo is not None:
            ax.axhspan(lo, hi, color=c, alpha=0.06)
    ax.set_xlabel(r"$t$ [Gyr]")
    ax.set_ylabel(r"$A_2(R_d)$ (disk COM)")
    ax.set_title(rf"Bar-sweep evolve ({tag}) — all tiers, {evolve_gyr:g} Gyr")
    ax.set_ylim(0, max(0.55, max(max(r["a2_t"]) for r in results.values()) * 1.15))
    ax.legend(fontsize=8, frameon=False, loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    local_a2 = out / "figs" / f"{tag}_all_tiers_evolve_a2_t.png"
    local_a2.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(local_a2, dpi=150)
    paper_a2_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper_a2_png, dpi=150)
    plt.close(fig)

    # Face-on t=0, t=end, and Δ grid.
    # Shared dens stretch across all tiers/times; signed Δ row; shared cbars.
    tiers = list(results.keys())
    half = float(SWEEP_HALF_XY)
    snap_half = float(EVOLVE_SNAP_HALF)  # must match COMPONENTS disk FOV at record time
    fig, axes = plt.subplots(
        3,
        len(tiers),
        figsize=(2.7 * len(tiers) + 1.1, 7.8),
    )
    if len(tiers) == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])
    maps_archive: dict[str, dict] = {}
    dens_bundle: list[tuple] = []  # (j, img0, img1, t0, t1)
    for j, tier in enumerate(tiers):
        snaps = results[tier]["snaps"]
        st = results[tier]["snap_t"]
        if not snaps:
            for r in range(3):
                axes[r, j].set_axis_off()
            continue
        t0 = float(st[0]) if st else 0.0
        t1 = float(st[-1]) if st else float(evolve_gyr)
        img0 = np.asarray(snaps[0]["maps"]["disk"], dtype=np.float32).copy()
        img1 = np.asarray(snaps[-1]["maps"]["disk"], dtype=np.float32).copy()
        if img0.shape == img1.shape:
            maxdiff = float(np.nanmax(np.abs(img0.astype(np.float64) - img1.astype(np.float64))))
            if maxdiff < 1e-12:
                print(
                    f"  WARNING: {tier} evolve face-on t0/t1 maps identical "
                    f"(max|Δ|={maxdiff:.2e}) — check snap recording",
                    flush=True,
                )
        img0c = _crop_faceon_half(img0, src_half=snap_half, half=half)
        img1c = _crop_faceon_half(img1, src_half=snap_half, half=half)
        maps_archive[tier] = {
            "t0": float(t0),
            "t1": float(t1),
            "disk_t0": img0c,
            "disk_t1": img1c,
        }
        dens_bundle.append((j, tier, img0c, img1c, t0, t1))

    # Global dens + Δ scales across the whole grid (not per-column).
    all_pos = []
    for _, _, img0c, img1c, _, _ in dens_bundle:
        for im in (img0c, img1c):
            p = im[np.isfinite(im) & (im > 0)]
            if p.size:
                all_pos.append(p.ravel())
    if all_pos:
        cat = np.concatenate(all_pos)
        floor = max(
            float(np.percentile(cat, 20)) * 1e-2,
            float(np.percentile(cat, 99.0)) * 1e-4,
            1e-12,
        )
        vmax = float(np.percentile(cat, 98))
    else:
        floor, vmax = 1e-12, 1.0

    delta_lim = 1e-3
    for _, _, img0c, img1c, _, _ in dens_bundle:
        eps = max(float(floor), 1e-30)
        d = np.log10(np.maximum(img1c.astype(np.float64), 0.0) + eps) - np.log10(
            np.maximum(img0c.astype(np.float64), 0.0) + eps
        )
        if np.any(np.isfinite(d)):
            delta_lim = max(
                delta_lim, float(np.percentile(np.abs(d[np.isfinite(d)]), 99))
            )

    last_dens = last_delta = None
    for j, tier, img0c, img1c, t0, t1 in dens_bundle:
        im0, _, _ = _faceon_from_snap_map(
            axes[0, j],
            img0c,
            rf"{tier}  $t={t0:.2f}$"
            f"\n"
            rf"$A_2={results[tier]['a2_pre']:.3f}$",
            half=half,
            floor=floor,
            vmax=vmax,
        )
        im1, _, _ = _faceon_from_snap_map(
            axes[1, j],
            img1c,
            rf"{tier}  $t={t1:.2f}$"
            f"\n"
            rf"$A_2={results[tier]['a2_post']:.3f}$",
            half=half,
            floor=floor,
            vmax=vmax,
        )
        imd, _ = _faceon_delta_map(
            axes[2, j],
            img0c,
            img1c,
            rf"{tier}  $\Delta\log_{{10}}\Sigma$",
            half=half,
            floor=floor,
            lim=delta_lim,
        )
        last_dens = im1
        last_delta = imd
        if j > 0:
            for r in range(3):
                axes[r, j].set_ylabel("")
                axes[r, j].tick_params(labelleft=False)
        for r in range(2):
            axes[r, j].set_xlabel("")
            axes[r, j].tick_params(labelbottom=False)

    fig.suptitle(
        rf"Bar-sweep face-on evolve ({tag}) — disk COM $A_2$; {evolve_gyr:g} Gyr"
        r" (rows: $t_0$, $t_\mathrm{end}$, $\Delta\log_{10}\Sigma$;"
        rf" dens $\log_{{10}}$, $\pm{half:g}$ kpc)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 0.90, 0.94])
    if last_dens is not None:
        _cbar_span(axes[0, -1], axes[1, -1], last_dens, label=r"$\log_{10}\Sigma$")
    if last_delta is not None:
        _cbar_beside(axes[2, -1], last_delta, label=r"$\Delta\log_{10}\Sigma$")
    local_f = out / "figs" / f"{tag}_all_tiers_evolve_faceon.png"
    fig.savefig(local_f, dpi=150)
    fig.savefig(paper_face_png, dpi=150)
    plt.close(fig)

    maps_path = out / "figs" / f"{tag}_all_tiers_evolve_faceon_maps.npz"
    np.savez_compressed(
        maps_path,
        **{
            f"{tier}_{k}": v
            for tier, bundle in maps_archive.items()
            for k, v in bundle.items()
            if isinstance(v, np.ndarray)
        },
        **{
            f"{tier}_{k}": np.asarray(v)
            for tier, bundle in maps_archive.items()
            for k, v in bundle.items()
            if not isinstance(v, np.ndarray)
        },
    )

    # Drop heavy snaps from returned JSON.
    slim = {}
    for tier, res in results.items():
        slim[tier] = {k: v for k, v in res.items() if k != "snaps"}
    slim["_figs"] = {
        "a2_t": str(paper_a2_png),
        "faceon": str(paper_face_png),
        "local_a2": str(local_a2),
        "local_faceon": str(local_f),
        "maps_npz": str(maps_path),
    }
    return slim


def _crop_faceon_half(
    img: np.ndarray, *, src_half: float, half: float
) -> np.ndarray:
    """Center-crop a square face-on map from ±src_half to ±half."""
    if half >= src_half - 1e-9:
        return np.asarray(img)
    n = int(img.shape[0])
    frac = float(half) / float(src_half)
    n_keep = max(8, int(round(n * frac)))
    if n_keep % 2 != n % 2:
        n_keep = min(n, n_keep + 1)
    lo = (n - n_keep) // 2
    return np.asarray(img[lo : lo + n_keep, lo : lo + n_keep])


def run_campaign(
    *,
    run_hash: str,
    ranked: list[dict],
    tiers: list[dict],
    out: Path,
    rd: float,
    n_max: int | None,
    t_min: float,
    promote_paper: bool,
) -> dict:
    label = HASH_LABEL.get(run_hash, run_hash[:8])
    print(f"\n=== bar sweep @ θ={label} ({run_hash}) ===", flush=True)
    camp_rows = [
        r
        for r in ranked
        if (r.get("run_hash") or _hash_from_path(r["path"])) == run_hash
    ]
    if not camp_rows:
        raise RuntimeError(f"no ranked dumps for {run_hash}")
    seed_row = max(camp_rows, key=lambda r: float(r.get("t_gyr") or 0.0))
    ic_path = _ic_path_for_dump(Path(seed_row["path"]))
    if ic_path is None:
        raise RuntimeError(f"no ic_state for {run_hash}")
    rng = np.random.default_rng(0)
    f0 = _load_parts(ic_path, n_max, rng)
    mass_prior = _component_masses(_load_parts(ic_path, None, rng))
    print(f"  f0 mass prior from {ic_path}: {mass_prior}", flush=True)

    measured = _cache_campaign_a2rd(
        ranked,
        run_hash=run_hash,
        rd=float(rd),
        cache_path=out / "logs" / f"a2rd_cache_{run_hash[:8]}.json",
        t_min=float(t_min),
    )
    print(
        f"  measured {len(measured)} dumps; "
        f"A₂(R_d) range "
        f"{min(m['a2_rd'] for m in measured):.3f}–"
        f"{max(m['a2_rd'] for m in measured):.3f}; "
        f"true-bar (A₁<A₂) "
        f"{sum(1 for m in measured if m.get('true_bar'))}/"
        f"{len(measured)}; "
        f"max true A₂="
        f"{max((m['a2_rd'] for m in measured if m.get('true_bar')), default=float('nan')):.3f}",
        flush=True,
    )

    rows_out: list[dict] = []
    for tier, nn in _pick_tiers(measured, tiers):
        nn_path = Path(nn["path"])
        gen = _decode_particle_retrieve(nn_path, mass_prior, n_max, rng)
        a2_rd = _a2_rd(gen, float(rd))
        a1_rd = _a_m_at_rd(gen, m=1, rd=float(rd))
        prof = _a2_profile(gen)
        bl, r_peak = _bar_length_rd(prof["r_mid"], prof["a2"], rd=float(rd))
        sample = out / "samples" / f"{run_hash[:5]}_{tier['name']}.npz"
        sample.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            sample,
            pos=gen["pos"],
            vel=gen["vel"],
            mass=gen["mass"],
            component_id=gen["component_id"],
            eps=gen.get("eps", np.full(len(gen["pos"]), 0.05)),
            neighbor_path=str(nn_path),
            tier=tier["name"],
            target_a2=float(tier["target_a2"]),
            a2_rd=float(a2_rd),
            a1_rd=float(a1_rd),
        )
        reachable = a2_rd >= float(tier["target_a2"]) - 0.03
        row = {
            "tier": tier["name"],
            "target_a2": float(tier["target_a2"]),
            "lo": float(tier["lo"]),
            "hi": float(tier["hi"]),
            "neighbor_path": str(nn_path),
            "neighbor_a2_rank": float(nn.get("rank_a2", float("nan"))),
            "neighbor_a2_rd": float(nn["a2_rd"]),
            "neighbor_a1_rd": float(nn.get("a1_rd", float("nan"))),
            "neighbor_t_gyr": float(nn.get("t_gyr") or 0.0),
            "a2_rd": float(a2_rd),
            "a1_rd": float(a1_rd),
            "a2_med": float(prof["a2_med"]),
            "bar_len_rd": float(bl),
            "r_peak": float(r_peak),
            "true_bar": bool(a1_rd < a2_rd) if tier["name"] != "quiet" else True,
            "target_reachable": bool(reachable),
            "in_band": float(tier["lo"]) <= a2_rd <= float(tier["hi"]),
            "sample": str(sample),
            "parts": gen,
            "a2_profile": {
                "r_mid": prof["r_mid"].tolist(),
                "a2": prof["a2"].tolist(),
            },
        }
        rows_out.append(row)
        print(
            f"  {tier['name']:8s} target={tier['target_a2']:.2f}  "
            f"nn={nn_path.name} dumpA2Rd={nn['a2_rd']:.3f} "
            f"dumpA1Rd={nn.get('a1_rd', float('nan')):.3f} "
            f"t={nn.get('t_gyr', 0):.2f}  "
            f"gen A2={a2_rd:.3f} A1={a1_rd:.3f}  Rbar/Rd={bl:.2f}  "
            f"{'IN' if row['in_band'] else 'OUT'} band"
            f"{'' if reachable else '  [target unreachable — max true bar]'}",
            flush=True,
        )

    tag = run_hash[:5]
    face = out / "figs" / f"{tag}_bar_sweep_faceon.png"
    edge = out / "figs" / f"{tag}_bar_sweep_edgeon.png"
    gallery = out / "figs" / f"{tag}_bar_sweep_faceon_edgeon.png"
    a2png = out / "figs" / f"{tag}_bar_sweep_a2.png"
    vs_f0 = out / "figs" / f"{tag}_bar_sweep_vs_f0.png"
    plot_sweep_faceon(rows_out, face, theta_label=label, rd=rd)
    plot_sweep_edgeon(rows_out, edge, theta_label=label, rd=rd)
    plot_sweep_faceon_edgeon(rows_out, gallery, theta_label=label, rd=rd)
    plot_sweep_a2(rows_out, a2png, theta_label=label, rd=rd)
    vs_meta = plot_vs_f0(rows_out, f0, vs_f0, theta_label=label, rd=rd)

    if promote_paper:
        paper_face = PAPER / "fig_latent_theta_bar_sweep_faceon.png"
        paper_edge = PAPER / "fig_latent_theta_bar_sweep_edgeon.png"
        paper_gal = PAPER / "fig_latent_theta_bar_sweep_faceon_edgeon.png"
        paper_a2 = PAPER / "fig_latent_theta_bar_sweep_a2.png"
        paper_vs = PAPER / "fig_latent_theta_bar_sweep_vs_f0.png"
        paper_face.write_bytes(face.read_bytes())
        paper_edge.write_bytes(edge.read_bytes())
        paper_gal.write_bytes(gallery.read_bytes())
        paper_a2.write_bytes(a2png.read_bytes())
        paper_vs.write_bytes(vs_f0.read_bytes())
        print(
            f"  promoted → {paper_face.name}, {paper_edge.name}, "
            f"{paper_gal.name}, {paper_a2.name}, {paper_vs.name}",
            flush=True,
        )
    else:
        paper_face = PAPER / f"fig_latent_theta_bar_sweep_{tag}_faceon.png"
        paper_edge = PAPER / f"fig_latent_theta_bar_sweep_{tag}_edgeon.png"
        paper_gal = PAPER / f"fig_latent_theta_bar_sweep_{tag}_faceon_edgeon.png"
        paper_a2 = PAPER / f"fig_latent_theta_bar_sweep_{tag}_a2.png"
        paper_vs = PAPER / f"fig_latent_theta_bar_sweep_{tag}_vs_f0.png"
        paper_face.write_bytes(face.read_bytes())
        paper_edge.write_bytes(edge.read_bytes())
        paper_gal.write_bytes(gallery.read_bytes())
        paper_a2.write_bytes(a2png.read_bytes())
        paper_vs.write_bytes(vs_f0.read_bytes())
        print(
            f"  wrote secondary → {paper_face.name}, {paper_edge.name}, {paper_a2.name}",
            flush=True,
        )

    meta_rows = []
    for r in rows_out:
        meta_rows.append({k: v for k, v in r.items() if k not in ("parts",)})
    return {
        "run_hash": run_hash,
        "label": label,
        "rd": float(rd),
        "ic_path": str(ic_path),
        "mass_prior": mass_prior,
        "tiers": meta_rows,
        "vs_f0": vs_meta,
        "figs": {
            "faceon": str(face),
            "edgeon": str(edge),
            "faceon_edgeon": str(gallery),
            "a2": str(a2png),
            "vs_f0": str(vs_f0),
            "paper_faceon": str(paper_face),
            "paper_edgeon": str(paper_edge),
            "paper_faceon_edgeon": str(paper_gal),
            "paper_a2": str(paper_a2),
            "paper_vs_f0": str(paper_vs),
        },
        "_parts_by_tier": {r["tier"]: r["parts"] for r in rows_out},
        "_f0": f0,
    }


def write_docs(payload: dict, args) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    primary = payload["campaigns"][0]
    tiers = primary["tiers"]
    max_true = max(
        (float(t["a2_rd"]) for t in tiers if t.get("true_bar", True) or t["tier"] != "quiet"),
        default=float("nan"),
    )
    lines = [
        "# SCOREBOARD — fixed-θ bar-strength sweep",
        "",
        "Generative = **A₂-conditioned retrieve + particle_retrieve** remass to",
        f"$f_0(\\theta)$; not free `decode(z)`; not eval-dump copy.",
        "",
        "**A₂ centering:** disk mass-COM before annular Fourier",
        "(`disk_azimuthal_fourier(recenter=True)`, default ON).",
        "",
        "**Dipole gate:** mild/moderate/strong require disk-COM $A_1(R_d)<A_2(R_d)$",
        "(reject A₁-dominated off-center junk). Quiet allows noise floor.",
        "",
        f"Primary θ: **{primary['label']}** (`{primary['run_hash']}`).",
        f"$R_d={primary['rd']:g}$ kpc. Evidence: `{args.out}/`.",
        "",
        "## Tier definition (recentered + A₁ gate)",
        "",
        "| Tier | Target $A_2(R_d)$ | Band | Role |",
        "|------|-------------------|------|------|",
        "| quiet | ≈0.02 | ≲0.05 | axisym / noise |",
        "| mild | ≈0.10 | 0.06–0.14 | short compact true bar |",
        "| moderate | ≈0.30 | 0.22–0.38 | compact true bar (corpus-limited) |",
        "| strong | ≈0.50 | 0.40–0.60 | max true bar if 0.5 unreachable |",
        "",
        "Selection: nearest same-campaign dump in **measured particle $A_2(R_d)$**",
        "(disk COM), A₁ gate for non-quiet, compactness preference for mild/moderate,",
        "strong = max true bar when target exceeds corpus.",
        "",
        "## Measured (primary)",
        "",
        "| Tier | target | nn $A_2$ | nn $A_1$ | gen $A_2$ | gen $A_1$ | $R_\\mathrm{bar}/R_d$ | reachable |",
        "|------|--------|----------|----------|-----------|-----------|----------------------|-----------|",
    ]
    for t in tiers:
        lines.append(
            f"| {t['tier']} | {t['target_a2']:.2f} | "
            f"{t.get('neighbor_a2_rd', float('nan')):.3f} | "
            f"{t.get('neighbor_a1_rd', float('nan')):.3f} | "
            f"**{t['a2_rd']:.3f}** | {t.get('a1_rd', float('nan')):.3f} | "
            f"{t['bar_len_rd']:.2f} | "
            f"{'yes' if t.get('target_reachable', t['in_band']) else 'no (max true)'} |"
        )
    lines += [
        "",
        f"Campaign max **true** bar $A_2(R_d)\\approx{max_true:.3f}$ "
        f"(target 0.50 {'reachable' if max_true >= 0.47 else '**not reachable**'}).",
    ]
    ev = payload.get("evolve") or {}
    lines += ["", f"## All-tier evolve ({args.evolve_gyr:g} Gyr)", ""]
    if ev:
        for tier in ("quiet", "mild", "moderate", "strong"):
            row = ev.get(tier)
            if not row:
                continue
            lines.append(
                f"- **{tier}**: $A_2(R_d)$ {row['a2_pre']:.3f}→{row['a2_post']:.3f}"
            )
        figs = ev.get("_figs") or {}
        if figs.get("a2_t"):
            lines.append(f"- Figs: `{Path(figs['a2_t']).name}`, `{Path(figs.get('faceon','')).name}`")
    else:
        lines.append("- skipped (`--evolve-gyr 0` or `--skip-evolve`).")
    vs = primary.get("vs_f0") or {}
    lines += [
        "",
        "## vs GalactICS $f_0(\\theta)$",
        "",
        f"- quiet $f_0$ $A_2(R_d)$ = {vs.get('f0_a2_rd', float('nan')):.3f}; "
        f"strong gen = {vs.get('strong_a2_rd', float('nan')):.3f}",
        "- Panel: `fig_latent_theta_bar_sweep_vs_f0.png` (face-on + dens/kin).",
        "",
        "## Live figures",
        "",
        "- `fig_latent_theta_bar_sweep_faceon.png`",
        "- `fig_latent_theta_bar_sweep_edgeon.png` (bar PA→x after disk-COM)",
        "- `fig_latent_theta_bar_sweep_faceon_edgeon.png` (2-row gallery)",
        "- `fig_latent_theta_bar_sweep_a2.png`",
        "- `fig_latent_theta_bar_sweep_vs_f0.png`",
        "- `fig_latent_theta_bar_sweep_evolve_a2_t.png`",
        "- `fig_latent_theta_bar_sweep_evolve_faceon.png`",
        "",
        "CLI::",
        "",
        "```bash",
        "CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \\",
        "  .venv/bin/python scripts/latent_theta_bar_sweep.py \\",
        f"    --out {args.out} \\",
        "    --hashes 906c4af73543,ac023abb258d \\",
        f"    --targets {args.targets} \\",
        f"    --n-resample 0 --n-disk {args.n_disk} --evolve-gyr {args.evolve_gyr:g} --force gpu_bh",
        "```",
        "",
        "Dens replot from saved samples (no GPU)::",
        "",
        "```bash",
        ".venv/bin/python scripts/latent_theta_bar_sweep.py --replot-from-samples \\",
        f"  --out {args.out}",
        "```",
        "",
    ]
    # Secondary campaign table if present.
    if len(payload["campaigns"]) > 1:
        sec = payload["campaigns"][1]
        lines += [
            f"## Secondary ({sec['label']})",
            "",
            "| Tier | gen $A_2$ | gen $A_1$ | reachable |",
            "|------|-----------|-----------|-----------|",
        ]
        for t in sec["tiers"]:
            lines.append(
                f"| {t['tier']} | {t['a2_rd']:.3f} | {t.get('a1_rd', float('nan')):.3f} | "
                f"{'yes' if t.get('target_reachable', t['in_band']) else 'no'} |"
            )
        lines += [
            "",
            f"Figs: `fig_latent_theta_bar_sweep_{sec['run_hash'][:5]}_{{faceon,edgeon,faceon_edgeon,a2,vs_f0}}.png`.",
            "",
        ]
    (RESULTS / "latent_theta_bar_sweep_SCOREBOARD.md").write_text("\n".join(lines) + "\n")
    (args.out / "SCOREBOARD.md").write_text("\n".join(lines) + "\n")
    (args.out / "verdict.json").write_text(
        json.dumps(
            {
                k: v
                for k, v in payload.items()
                if k != "campaigns"
            }
            | {
                "campaigns": [
                    {kk: vv for kk, vv in c.items() if not kk.startswith("_")}
                    for c in payload["campaigns"]
                ]
            },
            indent=2,
        )
        + "\n"
    )


def replot_from_samples(
    *,
    out: Path,
    hashes: list[str],
    rd: float,
    targets: list[float],
) -> list[dict]:
    """Reload saved sample NPZs and rewrite dens face-on / edge-on figs."""
    names = ["quiet", "mild", "moderate", "strong"]
    campaigns: list[dict] = []
    for i, h in enumerate(hashes):
        tag = h[:5]
        label = HASH_LABEL.get(h, h[:8])
        rows_out: list[dict] = []
        for j, name in enumerate(names):
            sample = out / "samples" / f"{tag}_{name}.npz"
            if not sample.is_file():
                print(f"  skip missing {sample}", flush=True)
                continue
            z = np.load(sample, allow_pickle=True)
            parts = {
                "pos": np.asarray(z["pos"]),
                "vel": np.asarray(z["vel"]),
                "mass": np.asarray(z["mass"]),
                "component_id": np.asarray(z["component_id"]),
            }
            tgt = float(targets[j]) if j < len(targets) else float(z["target_a2"])
            a2 = float(z["a2_rd"]) if "a2_rd" in z.files else _a2_rd(parts, rd)
            a1 = float(z["a1_rd"]) if "a1_rd" in z.files else float("nan")
            rows_out.append(
                {
                    "tier": name,
                    "target_a2": tgt,
                    "parts": parts,
                    "a2_rd": a2,
                    "a1_rd": a1,
                    "target_reachable": bool(a2 >= tgt - 0.03),
                    "sample": str(sample),
                }
            )
        if not rows_out:
            continue
        print(f"=== replot dens @ θ={label} ({h}) n={len(rows_out)} ===", flush=True)
        face = out / "figs" / f"{tag}_bar_sweep_faceon.png"
        edge = out / "figs" / f"{tag}_bar_sweep_edgeon.png"
        gallery = out / "figs" / f"{tag}_bar_sweep_faceon_edgeon.png"
        plot_sweep_faceon(rows_out, face, theta_label=label, rd=rd)
        plot_sweep_edgeon(rows_out, edge, theta_label=label, rd=rd)
        plot_sweep_faceon_edgeon(rows_out, gallery, theta_label=label, rd=rd)
        promote = i == 0
        if promote:
            paper_face = PAPER / "fig_latent_theta_bar_sweep_faceon.png"
            paper_edge = PAPER / "fig_latent_theta_bar_sweep_edgeon.png"
            paper_gal = PAPER / "fig_latent_theta_bar_sweep_faceon_edgeon.png"
        else:
            paper_face = PAPER / f"fig_latent_theta_bar_sweep_{tag}_faceon.png"
            paper_edge = PAPER / f"fig_latent_theta_bar_sweep_{tag}_edgeon.png"
            paper_gal = PAPER / f"fig_latent_theta_bar_sweep_{tag}_faceon_edgeon.png"
        PAPER.mkdir(parents=True, exist_ok=True)
        paper_face.write_bytes(face.read_bytes())
        paper_edge.write_bytes(edge.read_bytes())
        paper_gal.write_bytes(gallery.read_bytes())
        print(
            f"  → {face.name}, {edge.name}, {gallery.name}"
            f"  promoted={promote} → {paper_edge.name}",
            flush=True,
        )
        campaigns.append(
            {
                "run_hash": h,
                "label": label,
                "tiers": [
                    {k: v for k, v in r.items() if k != "parts"} for r in rows_out
                ],
                "figs": {
                    "faceon": str(face),
                    "edgeon": str(edge),
                    "faceon_edgeon": str(gallery),
                    "paper_faceon": str(paper_face),
                    "paper_edgeon": str(paper_edge),
                    "paper_faceon_edgeon": str(paper_gal),
                },
            }
        )
    return campaigns


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("runs/ml/field_maps/latent_theta_bar_sweep_2026-08-03"),
    )
    p.add_argument("--rank", type=Path, default=RANK)
    p.add_argument(
        "--hashes",
        type=str,
        default=",".join(DEFAULT_HASHES),
        help="Comma-separated campaign hashes (first = primary paper figs).",
    )
    p.add_argument("--rd", type=float, default=2.0)
    p.add_argument("--t-min", type=float, default=0.05)
    p.add_argument(
        "--n-resample",
        type=int,
        default=250_000,
        help="Stratified N for t=0 decode (0 = full-N).",
    )
    p.add_argument("--n-disk", type=int, default=1_000_000, help="Disk N for evolve.")
    p.add_argument("--evolve-gyr", type=float, default=0.5)
    p.add_argument("--force", type=str, default="gpu_bh")
    p.add_argument("--omp", type=int, default=2)
    p.add_argument("--skip-evolve", action="store_true")
    p.add_argument(
        "--targets",
        type=str,
        default=DEFAULT_TARGETS,
        help="Comma A₂(R_d) targets for quiet,mild,moderate,strong.",
    )
    p.add_argument(
        "--replot-from-samples",
        action="store_true",
        help="Only rewrite dens face-on/edge-on figs from samples/*.npz (no retrieve/evolve).",
    )
    args = p.parse_args()

    targets = [float(x) for x in args.targets.split(",") if x.strip()]
    hashes = [h.strip() for h in args.hashes.split(",") if h.strip()]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "figs").mkdir(exist_ok=True)

    if args.replot_from_samples:
        t0 = time.time()
        camps = replot_from_samples(
            out=out, hashes=hashes, rd=float(args.rd), targets=targets
        )
        # Refresh scoreboard live-figure list if prior verdict exists.
        prior = out / "verdict.json"
        if prior.is_file() and camps:
            payload = json.loads(prior.read_text())
            # Keep evolve / campaign metrics; only refresh fig paths for dens.
            for c_new in camps:
                for c_old in payload.get("campaigns") or []:
                    if c_old.get("run_hash") == c_new["run_hash"]:
                        figs = dict(c_old.get("figs") or {})
                        figs.update(c_new.get("figs") or {})
                        c_old["figs"] = figs
            write_docs(payload, args)
        print(f"\nReplot done in {time.time() - t0:.1f}s → {out}", flush=True)
        return

    names = ["quiet", "mild", "moderate", "strong"]
    tiers = []
    for i, tgt in enumerate(targets):
        name = names[i] if i < len(names) else f"t{i}"
        lo, hi = DEFAULT_BANDS[i] if i < len(DEFAULT_BANDS) else (tgt - 0.03, tgt + 0.03)
        tiers.append({"name": name, "target_a2": tgt, "lo": lo, "hi": hi})

    ranked = json.loads(Path(args.rank).read_text())
    (out / "samples").mkdir(exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)

    n_max = None if int(args.n_resample) <= 0 else int(args.n_resample)
    t0 = time.time()
    campaigns = []
    for i, h in enumerate(hashes):
        camp = run_campaign(
            run_hash=h,
            ranked=ranked,
            tiers=tiers,
            out=out,
            rd=float(args.rd),
            n_max=n_max,
            t_min=float(args.t_min),
            promote_paper=(i == 0),
        )
        campaigns.append(camp)
        (out / "logs" / f"sweep_{h[:8]}.json").write_text(
            json.dumps({k: v for k, v in camp.items() if not k.startswith("_")}, indent=2)
            + "\n"
        )

    evolve_meta = None
    if not args.skip_evolve and float(args.evolve_gyr) > 0:
        primary = campaigns[0]
        # Full-N decode per tier for evolve (if t=0 used n_max downsample).
        rng = np.random.default_rng(1)
        mass_prior = primary["mass_prior"]
        full_by_tier = {}
        for trow in primary["tiers"]:
            full_by_tier[trow["tier"]] = _decode_particle_retrieve(
                Path(trow["neighbor_path"]), mass_prior, None, rng
            )
        evolve_meta = evolve_all_tiers(
            full_by_tier,
            out=out,
            tag=primary["run_hash"][:5],
            n_disk=int(args.n_disk),
            evolve_gyr=float(args.evolve_gyr),
            force=str(args.force),
            omp=int(args.omp),
            a2_r_eval=float(args.rd),
            paper_a2_png=PAPER / "fig_latent_theta_bar_sweep_evolve_a2_t.png",
            paper_face_png=PAPER / "fig_latent_theta_bar_sweep_evolve_faceon.png",
        )
        # Keep legacy moderate-only name as copy of all-tier A2 fig for old links.
        mod_legacy = PAPER / "fig_latent_theta_bar_sweep_moderate_evolve_a2_t.png"
        src = Path(evolve_meta["_figs"]["a2_t"])
        if src.is_file():
            mod_legacy.write_bytes(src.read_bytes())
        evolve_meta["tag"] = primary["run_hash"][:5]

    payload = {
        "campaigns": campaigns,
        "evolve": evolve_meta,
        "elapsed_s": time.time() - t0,
        "cli": {
            "hashes": hashes,
            "targets": targets,
            "n_resample": args.n_resample,
            "n_disk": args.n_disk,
            "evolve_gyr": args.evolve_gyr,
            "force": args.force,
            "a2_recenter": "disk_com",
        },
    }
    write_docs(payload, args)
    print(f"\nDone in {payload['elapsed_s']:.1f}s → {out}", flush=True)


if __name__ == "__main__":
    main()
