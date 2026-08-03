#!/usr/bin/env python3
"""OOD θ DF proximity: dens + velocity moments vs GalactICS IC (and evolved).

For held-out structural θ (combinatorial OOD), compare:
  * GalactICS IC
  * FFT teacher recon (encode→decode→resample)
  * θ-nearest quiet library retrieve (generative; no native θ API)

Diagnostics (t=0 and optional post-evolve):
  * Density: face-on Σ, disk Σ(R), bulge/halo ρ(r)
  * Velocity: disk ⟨v_φ⟩(R), σ_R(R), σ_z(R); midplane dens / ⟨v_φ⟩ / σ maps
  * Relative profile residuals + face-on dens MSE; A₂(t) when evolved

Example::

    OMP_NUM_THREADS=2 .venv/bin/python scripts/ood_theta_df_compare.py \\
      --teacher runs/ml/field_maps/fft_morph_ft_long_2026-07-25/multitower_slice_ae.pt \\
      --out runs/ml/field_maps/ood_theta_df_compare_2026-07-27 \\
      --cases heavy_ext_quiet,heavy_bar_forming,thick_stable \\
      --n-disk 1000000 --evolve-gyr 0.50 --dt 0.01 --force gpu_bh \\
      --paper-figures papers/mnras_noneq_ics/figures \\
      --paper-results papers/mnras_noneq_ics/results
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evolve_component_slices import (  # noqa: E402
    COMPONENTS,
    _faceon_comp,
    _profile_comp,
)
from evolve_resample_compare import (  # noqa: E402
    _a2_disk,
    _evolve_tracked,
    _faceon,
    _metrics,
    _recon_particles,
    _vcom_only,
)
from ood_theta_compare import (  # noqa: E402
    CORPUS,
    DEFAULT_OOD_CASES,
    MANIFEST,
    THETA_DIST_KEYS,
    _dmin_to_train,
    _ensure_eps,
    _load_ic,
    _outside_box,
    _theta_from_model,
    _theta_nearest_sample,
    _train_coverage,
    _vec,
)
from sample_latent_ic import RANK, build_library, default_teacher  # noqa: E402

# FFT teacher / §4.4c OOD definition: 19-model field-map train coverage
# (manifest may now list the full 47-model corpus — do not use that for OOD).
DEFAULT_TRAIN_COVERAGE = Path(
    "runs/ml/field_maps/ood_theta_compare_2026-07-26/train_theta_coverage.json"
)

from galacticsics.ml.fields.resample import transport_ot_lite  # noqa: E402
from galacticsics.ml.profiles import cylindrical_radius, v_phi_cylindrical  # noqa: E402
from ntropy.analysis.disk_density import bin_plane_density  # noqa: E402

ARM_LABELS = {
    "galactics_ic": "GalactICS IC",
    "fft_recon": "FFT recon",
    "theta_nearest": r"θ-nearest lib",
    "ot_lite": r"OT-lite F (θ-nn→IC)",
    "residual_f0_self": r"residual $f_0$ (self)",
    "residual_f0_bar906": r"residual $f_0$ (906c4 morph)",
}
ARM_ORDER = (
    "galactics_ic",
    "fft_recon",
    "theta_nearest",
    "ot_lite",
    "residual_f0_self",
    "residual_f0_bar906",
)

_ARM_STYLES = {
    "galactics_ic": ("-", 2.0),
    "fft_recon": ("--", 1.7),
    "theta_nearest": (":", 1.8),
    "ot_lite": ("-.", 1.8),
    "residual_f0_self": ("-", 1.8),
    "residual_f0_bar906": ("-.", 1.7),
}
_ARM_COLORS = {
    "galactics_ic": "C0",
    "fft_recon": "C1",
    "theta_nearest": "C2",
    "ot_lite": "C3",
    "residual_f0_self": "C4",
    "residual_f0_bar906": "C5",
}


def _coverage_from_json(path: Path) -> dict:
    """Rebuild OOD distance tables from a saved train_theta_coverage.json."""
    raw = json.loads(path.read_text())
    by_run = {
        h: {k: float(v[k]) for k in THETA_DIST_KEYS} for h, v in raw["theta_by_run"].items()
    }
    arr = np.stack([_vec(v, THETA_DIST_KEYS) for v in by_run.values()])
    lo = np.asarray([float(raw["box_lo"][k]) for k in THETA_DIST_KEYS])
    hi = np.asarray([float(raw["box_hi"][k]) for k in THETA_DIST_KEYS])
    return {
        "n_models": int(raw["n_models"]),
        "n_snapshots": int(raw.get("n_snapshots", 0)),
        "keys": list(THETA_DIST_KEYS),
        "run_hashes": sorted(by_run.keys()),
        "theta_by_run": by_run,
        "box_lo": {k: float(raw["box_lo"][k]) for k in THETA_DIST_KEYS},
        "box_hi": {k: float(raw["box_hi"][k]) for k in THETA_DIST_KEYS},
        "span": np.maximum(hi - lo, 1e-6),
        "points": arr,
        "source": str(path),
    }


def _disk_kinematics(
    parts: dict,
    *,
    r_max: float = 15.0,
    n_bins: int = 28,
    z_max: float = 0.5,
    min_count: int = 20,
) -> dict:
    """Mass-weighted disk midplane ⟨v_φ⟩, σ_R, σ_z vs cylindrical R."""
    cid = parts["component_id"]
    disk = cid == 0
    pos = np.asarray(parts["pos"][disk], dtype=np.float64)
    vel = np.asarray(parts["vel"][disk], dtype=np.float64)
    mass = np.asarray(parts["mass"][disk], dtype=np.float64)
    mid = np.abs(pos[:, 2]) <= float(z_max)
    pos, vel, mass = pos[mid], vel[mid], mass[mid]
    R = cylindrical_radius(pos)
    vphi = v_phi_cylindrical(pos, vel)
    # cylindrical v_R
    x, y = pos[:, 0], pos[:, 1]
    vr = np.zeros_like(R)
    np.divide(x * vel[:, 0] + y * vel[:, 1], R, out=vr, where=R > 1e-8)
    vz = vel[:, 2]

    edges = np.linspace(0.0, float(r_max), int(n_bins) + 1)
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    mean_vphi = np.full(n_bins, np.nan)
    sig_r = np.full(n_bins, np.nan)
    sig_z = np.full(n_bins, np.nan)
    sig_phi = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(n_bins):
        m = (R >= edges[i]) & (R < edges[i + 1])
        n = int(m.sum())
        counts[i] = n
        if n < max(int(min_count), 1):
            continue
        w = mass[m]
        if float(np.sum(w)) <= 0.0:
            continue
        mean_vphi[i] = float(np.average(vphi[m], weights=w))
        # mass-weighted std
        def _wstd(x: np.ndarray) -> float:
            mu = float(np.average(x, weights=w))
            return float(np.sqrt(np.average((x - mu) ** 2, weights=w)))

        sig_r[i] = _wstd(vr[m])
        sig_z[i] = _wstd(vz[m])
        sig_phi[i] = _wstd(vphi[m])
    return {
        "r_mid": r_mid,
        "mean_vphi": mean_vphi,
        "sig_r": sig_r,
        "sig_z": sig_z,
        "sig_phi": sig_phi,
        "counts": counts,
    }


def _midplane_moment_maps(
    parts: dict,
    *,
    half: float = 12.0,
    n_bins: int = 96,
    z_max: float = 0.5,
    min_count: int = 5,
) -> dict[str, np.ndarray]:
    """Disk midplane dens / ⟨v_φ⟩ / σ_los≈σ_z maps (mass-weighted).

    Velocity moments in pixels with ``n < min_count`` are NaN (masked in
    plots) — never zero-filled, which falsely paints empty sky as ``v=0``.
    """
    cid = parts["component_id"]
    disk = cid == 0
    pos = np.asarray(parts["pos"][disk], dtype=np.float64)
    vel = np.asarray(parts["vel"][disk], dtype=np.float64)
    mass = np.asarray(parts["mass"][disk], dtype=np.float64)
    mid = np.abs(pos[:, 2]) <= float(z_max)
    pos, vel, mass = pos[mid], vel[mid], mass[mid]
    dens = bin_plane_density(
        pos, mass, axes=(0, 1), n_bins=n_bins, half_extent=half
    ).density.astype(np.float32)
    vphi = v_phi_cylindrical(pos, vel)
    # Digitize into same grid as dens
    edges = np.linspace(-half, half, n_bins + 1)
    ix = np.clip(np.digitize(pos[:, 0], edges) - 1, 0, n_bins - 1)
    iy = np.clip(np.digitize(pos[:, 1], edges) - 1, 0, n_bins - 1)
    sum_m = np.zeros((n_bins, n_bins), dtype=np.float64)
    sum_v = np.zeros((n_bins, n_bins), dtype=np.float64)
    sum_v2 = np.zeros((n_bins, n_bins), dtype=np.float64)
    sum_sz = np.zeros((n_bins, n_bins), dtype=np.float64)
    sum_sz2 = np.zeros((n_bins, n_bins), dtype=np.float64)
    counts = np.zeros((n_bins, n_bins), dtype=np.int64)
    np.add.at(sum_m, (ix, iy), mass)
    np.add.at(sum_v, (ix, iy), mass * vphi)
    np.add.at(sum_v2, (ix, iy), mass * vphi * vphi)
    np.add.at(sum_sz, (ix, iy), mass * vel[:, 2])
    np.add.at(sum_sz2, (ix, iy), mass * vel[:, 2] ** 2)
    np.add.at(counts, (ix, iy), 1)
    occ = counts >= max(int(min_count), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_v = np.where(occ & (sum_m > 0), sum_v / sum_m, np.nan)
        var_v = np.where(occ & (sum_m > 0), sum_v2 / sum_m - mean_v**2, np.nan)
        mean_sz = np.where(occ & (sum_m > 0), sum_sz / sum_m, np.nan)
        var_sz = np.where(occ & (sum_m > 0), sum_sz2 / sum_m - mean_sz**2, np.nan)
    sig_v = np.sqrt(np.maximum(var_v, 0.0))
    sig_z = np.sqrt(np.maximum(var_sz, 0.0))
    return {
        "dens": dens,
        "mean_vphi": mean_v.astype(np.float32),
        "sig_vphi": sig_v.astype(np.float32),
        "sig_z": sig_z.astype(np.float32),
    }


def _bundle(parts: dict, *, faceon_bins: int = 96) -> dict:
    pos = parts["pos"]
    mass = parts["mass"]
    cid = parts["component_id"]
    maps, profiles = {}, {}
    for name, comp_id, _k, half, r_max in COMPONENTS:
        maps[name] = _faceon_comp(pos, mass, cid, comp_id, n_bins=faceon_bins, half=half)
        profiles[name] = _profile_comp(pos, mass, cid, name, comp_id, r_max=r_max)
    kin = _disk_kinematics(parts)
    mom = _midplane_moment_maps(parts, n_bins=faceon_bins)
    return {
        "maps": maps,
        "profiles": profiles,
        "disk_kin": kin,
        "moment_maps": mom,
        "a2": float(_a2_disk(parts)),
        "metrics": _metrics(parts),
    }


def _rel_profile_mse(
    y_ref: np.ndarray,
    y: np.ndarray,
    *,
    counts: np.ndarray | None = None,
    floor: float = 1e-8,
    min_count: int = 20,
) -> float:
    """Mean squared relative residual on well-populated bins only."""
    ok = np.isfinite(y_ref) & np.isfinite(y) & (np.abs(y_ref) > floor)
    if counts is not None:
        ok = ok & (np.asarray(counts) >= int(min_count))
    if not np.any(ok):
        return float("nan")
    ref = y_ref[ok]
    err = (y[ok] - ref) / np.maximum(np.abs(ref), floor)
    return float(np.mean(err**2))


def _med_abs_log_resid(
    y_ref: np.ndarray,
    y: np.ndarray,
    *,
    counts: np.ndarray | None = None,
    floor: float = 1e-8,
    min_count: int = 20,
) -> float:
    """Median |log10(y/y_ref)| on populated bins (robust dens mismatch)."""
    ok = np.isfinite(y_ref) & np.isfinite(y) & (y_ref > floor) & (y > floor)
    if counts is not None:
        ok = ok & (np.asarray(counts) >= int(min_count))
    if not np.any(ok):
        return float("nan")
    return float(np.median(np.abs(np.log10(y[ok] / y_ref[ok]))))


def _abs_profile_mse(
    y_ref: np.ndarray,
    y: np.ndarray,
    *,
    counts: np.ndarray | None = None,
    min_count: int = 20,
) -> float:
    ok = np.isfinite(y_ref) & np.isfinite(y)
    if counts is not None:
        ok = ok & (np.asarray(counts) >= int(min_count))
    if not np.any(ok):
        return float("nan")
    return float(np.mean((y[ok] - y_ref[ok]) ** 2))


def _map_mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return float("nan")
    return float(np.mean((a[mask] - b[mask]) ** 2))


def _compare_to_ref(ref: dict, other: dict) -> dict:
    out: dict = {}
    for name, *_ in COMPONENTS:
        pr, po = ref["profiles"][name], other["profiles"][name]
        y_r = np.asarray(pr["y"])
        y_o = np.asarray(po["y"])
        # Prefer reference counts for masking (same binning).
        cnt = np.asarray(pr.get("counts", np.zeros_like(y_r)))
        out[f"dens_{name}_rel_mse"] = _rel_profile_mse(y_r, y_o, counts=cnt)
        out[f"dens_{name}_abs_mse"] = _abs_profile_mse(y_r, y_o, counts=cnt)
        out[f"dens_{name}_med_abs_log10"] = _med_abs_log_resid(y_r, y_o, counts=cnt)
        out[f"faceon_{name}_mse"] = _map_mse(ref["maps"][name], other["maps"][name])
    kr, ko = ref["disk_kin"], other["disk_kin"]
    kcnt = np.asarray(kr.get("counts", np.zeros_like(kr["mean_vphi"])))
    for key in ("mean_vphi", "sig_r", "sig_z", "sig_phi"):
        out[f"kin_{key}_abs_mse"] = _abs_profile_mse(kr[key], ko[key], counts=kcnt)
        out[f"kin_{key}_rel_mse"] = _rel_profile_mse(
            kr[key], ko[key], counts=kcnt, floor=1e-4
        )
    mr, mo = ref["moment_maps"], other["moment_maps"]
    for key in ("dens", "mean_vphi", "sig_vphi", "sig_z"):
        out[f"mommap_{key}_mse"] = _map_mse(mr[key], mo[key])
    out["a2_ref"] = float(ref["a2"])
    out["a2_other"] = float(other["a2"])
    out["a2_abs_diff"] = float(abs(other["a2"] - ref["a2"]))
    return out


def _imshow_log(ax, img: np.ndarray, *, cmap: str = "inferno") -> None:
    from galacticsics.campaign.analysis import dens_array_log10

    show, vmin_s, vmax_s, _ = dens_array_log10(
        np.nan_to_num(img, nan=0.0), vmax_pct=99.0
    )
    ax.imshow(
        show,
        origin="lower",
        cmap=cmap,
        vmin=vmin_s,
        vmax=vmax_s,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _imshow_sym(ax, img: np.ndarray, *, cmap: str = "coolwarm") -> None:
    """Symmetric diverging map (residuals). Empty → masked, not zero-filled."""
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        ax.axis("off")
        return
    vmax = float(np.percentile(np.abs(finite), 98)) or 1.0
    ax.imshow(
        np.ma.masked_invalid(img),
        origin="lower",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _imshow_vphi(
    ax,
    img: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
) -> None:
    """⟨v_φ⟩ map: sequential cmap; NaN/empty masked (never shown as v=0)."""
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        ax.axis("off")
        return
    if vmax is None:
        vmax = float(np.percentile(finite, 98))
    if vmin is None:
        vmin = float(max(0.0, np.percentile(finite, 2)))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(finite)) if finite.size else 1.0
        vmin = float(np.nanmin(finite)) if finite.size else 0.0
    ax.imshow(
        np.ma.masked_invalid(img),
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_dens_panel(
    path: Path,
    arms: dict[str, dict],
    *,
    title: str,
) -> None:
    """Rows: face-on disk / dens profiles; cols: GalactICS / recon / θ-nn."""
    tags = [t for t in ARM_ORDER if t in arms]
    n = len(tags)
    fig = plt.figure(figsize=(3.1 * n, 7.2))
    gs = fig.add_gridspec(3, n, height_ratios=[1.15, 1.0, 1.0], hspace=0.32, wspace=0.28)

    # Row 0: face-on disk
    for j, tag in enumerate(tags):
        ax = fig.add_subplot(gs[0, j])
        _imshow_log(ax, arms[tag]["maps"]["disk"])
        ax.set_title(
            f"{ARM_LABELS[tag]}\nA₂={arms[tag]['a2']:.3f}",
            fontsize=9,
        )
        if j == 0:
            ax.set_ylabel("disk face-on", fontsize=9)

    # Row 1: dens profiles overlay (all arms on shared axes for disk/bulge/halo)
    # Use full-width panels via nested layout: one axes spanning columns per component
    # Simpler: three profile axes below spanning all columns.
    ax_d = fig.add_subplot(gs[1, :])
    ax_b = fig.add_subplot(gs[2, 0])
    ax_h = fig.add_subplot(gs[2, 1:])
    for tag in tags:
        ls, lw = _ARM_STYLES.get(tag, ("-", 1.5))
        c = _ARM_COLORS.get(tag, "C7")
        for ax, name in ((ax_d, "disk"), (ax_b, "bulge"), (ax_h, "halo")):
            pr = arms[tag]["profiles"][name]
            r = np.asarray(pr["r_mid"])
            y = np.asarray(pr["y"])
            ok = np.isfinite(y) & (y > 0)
            if not np.any(ok):
                continue
            ax.plot(r[ok], y[ok], ls=ls, lw=lw, color=c, label=ARM_LABELS[tag] if ax is ax_d else None)
    for ax, name, xlab in (
        (ax_d, "disk", r"$R$ [kpc]"),
        (ax_b, "bulge", r"$r$ [kpc]"),
        (ax_h, "halo", r"$r$ [kpc]"),
    ):
        ax.set_yscale("log")
        kind = arms[tags[0]]["profiles"][name]["kind"]
        if kind == "rho":
            ax.set_xscale("log")
        ax.set_xlabel(xlab, fontsize=8)
        ylab = r"$\Sigma(R)$" if kind == "sigma" else r"$\rho(r)$"
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_title(name, fontsize=9)
    ax_d.legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle(title, fontsize=11, y=0.995)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_vel_panel(
    path: Path,
    arms: dict[str, dict],
    *,
    title: str,
) -> None:
    """⟨v_φ⟩ / σ maps + radial kinematics."""
    tags = [t for t in ARM_ORDER if t in arms]
    n = len(tags)
    fig = plt.figure(figsize=(3.0 * n, 8.4))
    gs = fig.add_gridspec(4, n, height_ratios=[1.0, 1.0, 1.05, 1.05], hspace=0.35, wspace=0.25)

    # Shared ⟨v_φ⟩ color scale across arms (occupied pixels only).
    vphi_vals = []
    for tag in tags:
        mv = np.asarray(arms[tag]["moment_maps"]["mean_vphi"], dtype=np.float64)
        fin = mv[np.isfinite(mv)]
        if fin.size:
            vphi_vals.append(fin)
    if vphi_vals:
        all_v = np.concatenate(vphi_vals)
        vphi_vmin = float(max(0.0, np.percentile(all_v, 2)))
        vphi_vmax = float(np.percentile(all_v, 98))
    else:
        vphi_vmin, vphi_vmax = 0.0, 2.0

    for j, tag in enumerate(tags):
        mm = arms[tag]["moment_maps"]
        ax0 = fig.add_subplot(gs[0, j])
        _imshow_log(ax0, mm["dens"])
        ax0.set_title(f"{ARM_LABELS[tag]}\ndens", fontsize=8)
        ax1 = fig.add_subplot(gs[1, j])
        _imshow_vphi(ax1, mm["mean_vphi"], vmin=vphi_vmin, vmax=vphi_vmax)
        ax1.set_title(r"$\langle v_\phi\rangle$", fontsize=8)
        if j == 0:
            ax0.set_ylabel("midplane map", fontsize=8)
            ax1.set_ylabel("midplane map", fontsize=8)

    # kinematics profiles spanning columns
    ax_v = fig.add_subplot(gs[2, :])
    ax_s = fig.add_subplot(gs[3, :])
    for tag in tags:
        ls, lw = _ARM_STYLES.get(tag, ("-", 1.5))
        c = _ARM_COLORS.get(tag, "C7")
        k = arms[tag]["disk_kin"]
        r = np.asarray(k["r_mid"], dtype=np.float64)
        vphi = np.asarray(k["mean_vphi"], dtype=np.float64)
        sig_r = np.asarray(k["sig_r"], dtype=np.float64)
        sig_z = np.asarray(k["sig_z"], dtype=np.float64)
        ok_v = np.isfinite(vphi)
        ok_r = np.isfinite(sig_r)
        ok_z = np.isfinite(sig_z)
        if np.any(ok_v):
            ax_v.plot(r[ok_v], vphi[ok_v], ls=ls, lw=lw, color=c, label=ARM_LABELS[tag])
        if np.any(ok_r):
            ax_s.plot(
                r[ok_r], sig_r[ok_r], ls=ls, lw=lw, color=c, label=rf"{ARM_LABELS[tag]} $\sigma_R$"
            )
        if np.any(ok_z):
            ax_s.plot(
                r[ok_z],
                sig_z[ok_z],
                ls=ls,
                lw=lw,
                color=c,
                alpha=0.45,
                label=rf"{ARM_LABELS[tag]} $\sigma_z$",
            )
    ax_v.axvline(12.0, color="0.6", ls=":", lw=0.8, alpha=0.7)
    ax_v.set_xlabel(r"$R$ [kpc]")
    ax_v.set_ylabel(r"$\langle v_\phi\rangle$ [code$\approx 100\,\mathrm{km\,s^{-1}}$]")
    ax_v.set_title("disk midplane rotation", fontsize=9)
    ax_v.legend(frameon=False, fontsize=7)
    ax_s.set_xlabel(r"$R$ [kpc]")
    ax_s.set_ylabel(r"$\sigma$ [code]")
    ax_s.set_title(r"disk velocity dispersions", fontsize=9)
    ax_s.legend(frameon=False, fontsize=6.5, ncol=2)
    fig.suptitle(title, fontsize=11, y=0.995)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_evolved_dens(
    path: Path,
    arms_t0: dict[str, dict],
    arms_tf: dict[str, dict],
    *,
    title: str,
    t_end: float,
) -> None:
    """Face-on t=0 vs t_end for GalactICS vs recon (and θ-nn if present)."""
    tags = [t for t in ARM_ORDER if t in arms_t0 and t in arms_tf]
    n = len(tags)
    fig, axes = plt.subplots(2, n, figsize=(2.9 * n, 5.6), squeeze=False)
    for j, tag in enumerate(tags):
        for i, (bundle, lab) in enumerate(
            ((arms_t0[tag], "t=0"), (arms_tf[tag], rf"$t={t_end:.2g}$"))
        ):
            ax = axes[i, j]
            _imshow_log(ax, bundle["maps"]["disk"])
            ax.set_title(f"{ARM_LABELS[tag]} {lab}\nA₂={bundle['a2']:.3f}", fontsize=8)
            if j == 0:
                ax.set_ylabel(lab, fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_a2_overlay(path: Path, evolve_arms: dict[str, dict], *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    for tag in ARM_ORDER:
        row = evolve_arms.get(tag)
        if not row or not row.get("ok"):
            continue
        ax.plot(row["t_gyr"], row["a2_t"], lw=1.7, label=ARM_LABELS[tag])
    ax.axhline(0.10, color="0.6", ls=":", lw=1)
    ax.set_xlabel(r"$t$ [Gyr]")
    ax.set_ylabel(r"disk median $A_2$")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {
            k: _jsonable(v)
            for k, v in obj.items()
            if k
            not in (
                "maps",
                "profiles",
                "disk_kin",
                "moment_maps",
                "faceon_maps",
                "pos_final",
                "face_pre",
                "face_post",
                "parts",
            )
        }
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("runs/ml/field_maps/ood_theta_df_compare_2026-07-27"),
    )
    p.add_argument("--teacher", type=Path, default=None)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument(
        "--train-coverage",
        type=Path,
        default=DEFAULT_TRAIN_COVERAGE,
        help="19-model FFT-teacher train θ coverage (not the full 47-model corpus manifest).",
    )
    p.add_argument("--rank", type=Path, default=RANK)
    p.add_argument("--corpus", type=Path, default=CORPUS)
    p.add_argument("--n-bar", type=int, default=24)
    p.add_argument("--n-quiet", type=int, default=14)
    p.add_argument("--n-mid", type=int, default=8)
    p.add_argument("--n-rot-bar", type=int, default=4)
    p.add_argument("--bar-floor", type=float, default=0.20)
    p.add_argument("--quiet-ceil", type=float, default=0.06)
    p.add_argument("--enc-grid", type=int, default=4)
    p.add_argument("--n-pc", type=int, default=64)
    p.add_argument("--n-disk", type=int, default=1_000_000)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--evolve-gyr", type=float, default=0.50)
    p.add_argument("--omp", type=int, default=2)
    p.add_argument("--timeout-s", type=float, default=7200.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--theta-knn", type=int, default=3)
    p.add_argument("--faceon-bins", type=int, default=96)
    p.add_argument(
        "--cases",
        type=str,
        default="heavy_ext_quiet,heavy_bar_forming,thick_stable",
    )
    p.add_argument("--paper-figures", type=Path, default=None)
    p.add_argument("--paper-results", type=Path, default=None)
    p.add_argument("--skip-evolve", action="store_true")
    p.add_argument(
        "--force",
        type=str,
        default=None,
        choices=("gpu_bh", "bh_c", "bh"),
        help="Force backend for evolve (default: gpu_bh if available).",
    )
    p.add_argument(
        "--save-particles",
        action="store_true",
        help="Save IC npz per arm (large).",
    )
    p.add_argument(
        "--velocity-frame",
        choices=("cartesian", "cylindrical"),
        default="cylindrical",
        help="Resample velocity draw frame (default cylindrical = DF option B).",
    )
    p.add_argument(
        "--match-cell-moments",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Affine-correct per-cell mean/σ after resample (DF option B; default on).",
    )
    p.add_argument(
        "--with-ot-lite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add OT-lite F arm: radial CDF + ⟨v_φ⟩ transport from θ-nearest → IC.",
    )
    args = p.parse_args()

    if args.teacher is None:
        args.teacher = default_teacher()
    n_tot = int(round(args.n_disk * 7 / 4))
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = str(args.omp)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    cov_path = args.train_coverage
    if cov_path is not None and cov_path.is_file():
        cov = _coverage_from_json(cov_path)
        print(
            f"train coverage (from {cov_path}): {cov['n_models']} models "
            f"(FFT-teacher / §4.4c OOD definition)",
            flush=True,
        )
    else:
        cov = _train_coverage(args.manifest)
        print(
            f"train coverage (from manifest {args.manifest}): {cov['n_models']} models; "
            f"n-disk={args.n_disk:,} → N={n_tot:,}",
            flush=True,
        )
    print(f"n-disk={args.n_disk:,} → N={n_tot:,}", flush=True)
    want = {x.strip() for x in args.cases.split(",") if x.strip()}
    cases_spec = [c for c in DEFAULT_OOD_CASES if c["name"] in want]
    if not cases_spec:
        raise SystemExit(f"no cases from {want}")

    print(f"=== teacher {args.teacher} ===", flush=True)
    print("=== build stratified library ===", flush=True)
    lib = build_library(args)

    case_summaries: list[dict] = []
    for ci, spec in enumerate(cases_spec):
        h = spec["run_hash"]
        run_dir = args.corpus / h
        model = json.loads((run_dir / "model.json").read_text())
        theta = _theta_from_model(model)
        dmin = _dmin_to_train(theta, cov)
        outside = _outside_box(theta, cov)
        ic_path = run_dir / "ic_state.npz"
        if not ic_path.is_file():
            raise SystemExit(f"missing IC {ic_path}")
        if h in cov["run_hashes"]:
            raise SystemExit(f"{h} is in train set — not OOD")

        print(f"\n=== case {ci+1}/{len(cases_spec)} {spec['name']} ({h[:12]}) ===", flush=True)
        print(
            f"  θ: M={theta['disk.mass']} Rd={theta['disk.scale_length']} "
            f"zd={theta['disk.scale_height']} Q={theta['disk_kinematics.toomre_q_target']} "
            f"| dmin={dmin:.3f}",
            flush=True,
        )
        case_dir = args.out / spec["name"]
        case_dir.mkdir(parents=True, exist_ok=True)

        # --- build arms (particles) ---
        particles: dict[str, dict] = {}
        print("  arm: galactics_ic", flush=True)
        ic = _ensure_eps(_load_ic(ic_path, n_tot, rng))
        particles["galactics_ic"] = ic
        print(f"    A₂={_a2_disk(ic):.4f}", flush=True)

        print("  arm: fft_recon", flush=True)
        recon = _ensure_eps(
            _vcom_only(
                _recon_particles(
                    ic_path,
                    lib.teacher,
                    lib.cfg,
                    lib.stats,
                    n_resample=n_tot,
                    rng=rng,
                    velocity_frame=str(args.velocity_frame),
                    match_cell_moments=bool(args.match_cell_moments),
                )
            )
        )
        particles["fft_recon"] = recon
        print(f"    A₂={_a2_disk(recon):.4f}", flush=True)

        print("  arm: theta_nearest", flush=True)
        gen, gen_meta = _theta_nearest_sample(
            lib,
            theta,
            cov,
            prefer_quiet=True,
            quiet_ceil=args.quiet_ceil,
            k=args.theta_knn,
            rng=rng,
            n_resample=n_tot,
            velocity_frame=str(args.velocity_frame),
            match_cell_moments=bool(args.match_cell_moments),
        )
        gen = _ensure_eps(gen)
        particles["theta_nearest"] = gen
        print(
            f"    A₂={_a2_disk(gen):.4f} nn={[n['run_hash'][:8] for n in gen_meta['nn']]}",
            flush=True,
        )

        ot_meta: dict | None = None
        if args.with_ot_lite:
            print("  arm: ot_lite (θ-nearest → GalactICS radial+vφ transport)", flush=True)
            ot_parts, ot_meta = transport_ot_lite(gen, ic)
            ot_parts = _ensure_eps(_vcom_only(ot_parts))
            particles["ot_lite"] = ot_parts
            print(f"    A₂={_a2_disk(ot_parts):.4f}", flush=True)

        compare_tags = tuple(
            t for t in ("fft_recon", "theta_nearest", "ot_lite") if t in particles
        )

        if args.save_particles:
            for tag, parts in particles.items():
                np.savez_compressed(
                    case_dir / f"particles_{tag}.npz",
                    pos=parts["pos"],
                    vel=parts["vel"],
                    mass=parts["mass"],
                    eps=parts["eps"],
                    component_id=parts["component_id"],
                )

        # --- t=0 DF bundles ---
        bundles_t0 = {tag: _bundle(parts, faceon_bins=args.faceon_bins) for tag, parts in particles.items()}
        ref = bundles_t0["galactics_ic"]
        metrics_t0 = {
            tag: _compare_to_ref(ref, bundles_t0[tag])
            for tag in compare_tags
        }
        print("  t=0 dens/vel residuals vs GalactICS:", flush=True)
        for tag, m in metrics_t0.items():
            print(
                f"    {tag}: dens_disk med|log10|={m['dens_disk_med_abs_log10']:.3f} "
                f"relMSE={m['dens_disk_rel_mse']:.4f} "
                f"bulge_medlog={m['dens_bulge_med_abs_log10']:.3f} "
                f"halo_medlog={m['dens_halo_med_abs_log10']:.3f} "
                f"vφ_mse={m['kin_mean_vphi_abs_mse']:.4e} "
                f"σR_mse={m['kin_sig_r_abs_mse']:.4e} "
                f"faceon_disk_mse={m['faceon_disk_mse']:.4e}",
                flush=True,
            )

        dens_png = case_dir / "df_dens_t0.png"
        vel_png = case_dir / "df_vel_t0.png"
        _plot_dens_panel(
            dens_png,
            bundles_t0,
            title=rf"OOD {spec['name']}: density DF match at $t=0$ (dmin={dmin:.2f})",
        )
        _plot_vel_panel(
            vel_png,
            bundles_t0,
            title=rf"OOD {spec['name']}: velocity DF match at $t=0$",
        )

        evolve_report: dict[str, dict] = {}
        bundles_tf: dict[str, dict] = {}
        metrics_tf: dict[str, dict] = {}
        if not args.skip_evolve:
            for tag, parts in particles.items():
                print(f"  evolve {tag} ({args.evolve_gyr} Gyr)…", flush=True)
                t0 = time.time()
                ev = _evolve_tracked(
                    parts,
                    end_gyr=args.evolve_gyr,
                    dt=args.dt,
                    omp=args.omp,
                    timeout_s=args.timeout_s,
                    faceon_times=[0.0, args.evolve_gyr],
                    faceon_bins=args.faceon_bins,
                    force=args.force,
                )
                print(
                    f"    ok={ev.get('ok')} A₂ {bundles_t0[tag]['a2']:.3f}→"
                    f"{(ev.get('a2_t') or [float('nan')])[-1]:.3f} "
                    f"force={ev.get('force_method')} wall={time.time()-t0:.0f}s",
                    flush=True,
                )
                evolve_report[tag] = {
                    "ok": bool(ev.get("ok")),
                    "t_gyr": ev.get("t_gyr"),
                    "a2_t": ev.get("a2_t"),
                    "com_norm_t": ev.get("com_norm_t"),
                    "com_drift_kpc": ev.get("com_drift_kpc"),
                    "force_method": ev.get("force_method"),
                    "wall_s": ev.get("wall_s"),
                    "n_steps": ev.get("n_steps"),
                }
                if ev.get("ok") and ev.get("pos_final") is not None:
                    final_parts = {
                        "pos": np.asarray(ev["pos_final"], dtype=np.float64),
                        "vel": np.asarray(
                            ev.get("vel_final", parts["vel"]), dtype=np.float64
                        ),
                        "mass": parts["mass"],
                        "eps": parts["eps"],
                        "component_id": parts["component_id"],
                    }
                    bundles_tf[tag] = _bundle(final_parts, faceon_bins=args.faceon_bins)
                    if ev.get("a2_t"):
                        bundles_tf[tag]["a2"] = float(ev["a2_t"][-1])

            if bundles_tf and "galactics_ic" in bundles_tf:
                ref_f = bundles_tf["galactics_ic"]
                metrics_tf = {
                    tag: _compare_to_ref(ref_f, bundles_tf[tag])
                    for tag in compare_tags
                    if tag in bundles_tf
                }
                print("  post-evolve dens/vel residuals vs GalactICS:", flush=True)
                for tag, m in metrics_tf.items():
                    print(
                        f"    {tag}: dens_disk_rel_mse={m['dens_disk_rel_mse']:.4f} "
                        f"vφ_mse={m['kin_mean_vphi_abs_mse']:.4e} "
                        f"σR_mse={m['kin_sig_r_abs_mse']:.4e}",
                        flush=True,
                    )
                _plot_evolved_dens(
                    case_dir / "df_dens_evolved.png",
                    bundles_t0,
                    bundles_tf,
                    title=rf"OOD {spec['name']}: evolved face-on dens",
                    t_end=args.evolve_gyr,
                )
                _plot_vel_panel(
                    case_dir / "df_vel_evolved.png",
                    bundles_tf,
                    title=rf"OOD {spec['name']}: velocity DF at $t={args.evolve_gyr:.2g}$",
                )
                _plot_dens_panel(
                    case_dir / "df_dens_profiles_evolved.png",
                    bundles_tf,
                    title=rf"OOD {spec['name']}: dens profiles at $t={args.evolve_gyr:.2g}$",
                )
                _plot_a2_overlay(
                    case_dir / "a2_t.png",
                    evolve_report,
                    title=rf"OOD {spec['name']} $A_2(t)$",
                )

        # paper copies
        if args.paper_figures is not None:
            args.paper_figures.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                dens_png,
                args.paper_figures / f"fig_ood_theta_df_dens_{spec['name']}.png",
            )
            shutil.copy2(
                vel_png,
                args.paper_figures / f"fig_ood_theta_df_vel_{spec['name']}.png",
            )
            # DF-match aliases used in paper §4.4c / OT-lite writeups
            for sfx in ("", "_otF") if args.with_ot_lite else ("",):
                shutil.copy2(
                    dens_png,
                    args.paper_figures
                    / f"fig_dfmatch_ood_{spec['name']}_dens_t0{sfx}.png",
                )
                shutil.copy2(
                    vel_png,
                    args.paper_figures
                    / f"fig_dfmatch_ood_{spec['name']}_vel_t0{sfx}.png",
                )
            evo_face = case_dir / "df_dens_evolved.png"
            if evo_face.is_file():
                shutil.copy2(
                    evo_face,
                    args.paper_figures / f"fig_ood_theta_df_evolved_{spec['name']}.png",
                )

        report = {
            "name": spec["name"],
            "run_hash": h,
            "note": spec["note"],
            "ood_how": spec["ood_how"],
            "theta": {
                k: theta[k]
                for k in (
                    "disk.mass",
                    "disk.scale_length",
                    "disk.scale_height",
                    "disk_kinematics.toomre_q_target",
                    "halo.v0",
                    "halo.a",
                    "bulge.v0",
                    "bulge.a",
                )
            },
            "dmin_to_train": dmin,
            "outside_train_box": bool(outside),
            "n_disk": args.n_disk,
            "n_total": n_tot,
            "teacher": str(args.teacher),
            "theta_nearest_meta": gen_meta,
            "ot_lite_meta": ot_meta,
            "metrics_t0": metrics_t0,
            "metrics_tf": metrics_tf,
            "evolve": evolve_report,
            "a2_t0": {tag: float(bundles_t0[tag]["a2"]) for tag in bundles_t0},
            "figures": {
                "dens_t0": str(dens_png),
                "vel_t0": str(vel_png),
            },
        }
        (case_dir / "verdict.json").write_text(json.dumps(_jsonable(report), indent=2))
        case_summaries.append(report)

    # aggregate summary
    lines = [
        "# OOD θ DF proximity (dens + velocity)",
        "",
        f"Teacher: `{args.teacher}`",
        f"Train models: **{cov['n_models']}** "
        f"(coverage `{cov.get('source', args.train_coverage or args.manifest)}`).",
        f"Settings: disk$={args.n_disk:,}$, total$={n_tot:,}$, "
        f"$t_{{\\rm end}}={args.evolve_gyr}$ Gyr, $dt={args.dt}$, "
        f"force=`{args.force or 'auto'}`, OMP={args.omp}.",
        "",
        "## Limit",
        "",
        "`TeacherFeatureLibrary` has **no** θ-conditioning API. "
        "Generative arm = θ-nearest quiet retrieve of **raw GalactICS snapshot "
        "particles** (k=3; not AE feature decode — AE moments collapse outer "
        "⟨v_φ⟩). OT-lite F transports that trusted DF toward the OOD IC.",
        "",
        "## t=0 residuals vs GalactICS IC",
        "",
        "| Case | arm | disk med‖log₁₀ρ‖ | bulge | halo | dens disk relMSE | ⟨v_φ⟩ MSE | σ_R MSE | face-on dens MSE | A₂ |",
        "|------|-----|-------------------|-------|------|------------------|-----------|---------|------------------|----|",
    ]
    for rep in case_summaries:
        for tag, m in rep["metrics_t0"].items():
            lines.append(
                f"| `{rep['name']}` | {tag} | "
                f"{m['dens_disk_med_abs_log10']:.3f} | {m['dens_bulge_med_abs_log10']:.3f} | "
                f"{m['dens_halo_med_abs_log10']:.3f} | {m['dens_disk_rel_mse']:.4f} | "
                f"{m['kin_mean_vphi_abs_mse']:.3e} | {m['kin_sig_r_abs_mse']:.3e} | "
                f"{m['faceon_disk_mse']:.3e} | {m['a2_other']:.4f} |"
            )
    if any(r.get("metrics_tf") for r in case_summaries):
        lines += [
            "",
            f"## Post-evolve ($t={args.evolve_gyr}$) residuals vs GalactICS",
            "",
            "| Case | arm | dens disk relMSE | ⟨v_φ⟩ MSE | σ_R MSE | A₂→ |",
            "|------|-----|------------------|-----------|---------|-----|",
        ]
        for rep in case_summaries:
            for tag, m in (rep.get("metrics_tf") or {}).items():
                a2_end = (rep.get("evolve") or {}).get(tag, {}).get("a2_t") or [float("nan")]
                lines.append(
                    f"| `{rep['name']}` | {tag} | "
                    f"{m['dens_disk_rel_mse']:.4f} | {m['kin_mean_vphi_abs_mse']:.3e} | "
                    f"{m['kin_sig_r_abs_mse']:.3e} | {a2_end[-1]:.4f} |"
                )
    lines += [
        "",
        "## Verdict notes",
        "",
        "- **FFT recon** should track GalactICS dens + moments closely when the "
        "teacher towers resolve the component (bulge cusp residual expected).",
        "- **θ-nearest generative** matches a *nearby train* mass model, not the "
        "held-out θ — dens/vel residuals to the OOD GalactICS IC are larger by construction.",
        "",
        f"Run dir: `{args.out}/`",
        "Paper figs: `fig_ood_theta_df_dens_*.png`, `fig_ood_theta_df_vel_*.png`, "
        "optional `fig_ood_theta_df_evolved_*.png`.",
        "",
    ]
    summary_md = "\n".join(lines)
    (args.out / "SUMMARY.md").write_text(summary_md)
    verdict = {
        "teacher": str(args.teacher),
        "n_disk": args.n_disk,
        "n_total": n_tot,
        "evolve_gyr": args.evolve_gyr,
        "force": args.force,
        "cases": _jsonable(case_summaries),
    }
    (args.out / "verdict.json").write_text(json.dumps(verdict, indent=2))
    if args.paper_results is not None:
        args.paper_results.mkdir(parents=True, exist_ok=True)
        (args.paper_results / "ood_theta_df_SUMMARY.md").write_text(summary_md)
        (args.paper_results / "ood_theta_df_verdict.json").write_text(
            json.dumps(verdict, indent=2)
        )
        shutil.copy2(args.out / "SUMMARY.md", args.paper_results / "ood_theta_df_SUMMARY.md")

    print("\n" + summary_md, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
