#!/usr/bin/env python3
"""Per-component face-on Σ maps + radial density profiles over evolve.

Evolves corpus-scale ICs (disk=1e6 → total ≈1.75e6, mix 4:2:1) and records
**disk / bulge / halo** face-on surface-density maps and radial profiles at
multiple times. Existing ``evolve_*disk1e6*`` dumps only kept disk face-ons,
so this script re-evolves with on-the-fly component diagnostics (no full
particle snapshot archive — maps/profiles only, to avoid OOM).

Example::

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 \\
      .venv/bin/python scripts/evolve_component_slices.py \\
        --teacher runs/ml/field_maps/fft_morph_ft_long_2026-07-25/multitower_slice_ae.pt \\
        --out runs/ml/field_maps/evolve_component_slices_2026-07-26 \\
        --n-disk 1000000 --evolve-gyr 0.50 --dt 0.01 --omp 2 \\
        --methods '' --with-recon --reuse-data-from \\
          runs/ml/field_maps/evolve_component_slices_2026-07-26 \\
        --faceon-times 0,0.12,0.25,0.38,0.5 \\
        --paper-figures papers/mnras_noneq_ics/figures
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evolve_resample_compare import (  # noqa: E402
    _a2_disk,
    _full_com,
    _full_dyn_residual_particles,
    _ic_has_bulge,
    _metrics,
    _n_total_for_disk,
    _recon_particles,
    _residual_f0_particles,
    _sample_gen,
    _stratified_down,
    _subsample_dump,
    _vcom_only,
)
from sample_latent_ic import RANK, build_library, default_teacher  # noqa: E402

from galacticsics.ml.fields.feature_library import load_frozen_teacher_bundle  # noqa: E402
from ntropy.analysis.density import bin_spherical_density  # noqa: E402
from ntropy.analysis.disk_density import (  # noqa: E402
    bin_midplane_surface_density,
    bin_plane_density,
)
from ntropy.analysis.disk_density import disk_azimuthal_fourier  # noqa: E402

# component_id: 0=disk, 1=halo, 2=bulge
COMPONENTS = (
    ("disk", 0, "faceon", 14.0, 15.0),
    ("bulge", 2, "faceon", 5.0, 6.0),
    ("halo", 1, "faceon", 40.0, 50.0),
)
# name, cid, map_kind (unused reserved), faceon_half_kpc, profile_rmax_kpc


def _component_frame(
    pos: np.ndarray,
    mass: np.ndarray,
    cid: np.ndarray,
    comp_id: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Select one component and recenter on its mass-weighted COM.

    Global soft COM is halo-dominated, so bulge/disk often sit ~1–2 kpc off
    the origin; profiles/maps about (0,0) then look like hollow rings.
    """
    mask = np.asarray(cid) == int(comp_id)
    if not np.any(mask):
        return None
    p = np.asarray(pos[mask], dtype=np.float64)
    m = np.asarray(mass[mask], dtype=np.float64)
    com = np.average(p, axis=0, weights=m)
    return p - com, m


def _faceon_comp(
    pos: np.ndarray,
    mass: np.ndarray,
    cid: np.ndarray,
    comp_id: int,
    *,
    n_bins: int,
    half: float,
) -> np.ndarray:
    framed = _component_frame(pos, mass, cid, comp_id)
    if framed is None:
        return np.zeros((n_bins, n_bins), dtype=np.float32)
    p, m = framed
    return bin_plane_density(
        p,
        m,
        axes=(0, 1),
        n_bins=n_bins,
        half_extent=half,
    ).density.astype(np.float32)


def _profile_comp(
    pos: np.ndarray,
    mass: np.ndarray,
    cid: np.ndarray,
    name: str,
    comp_id: int,
    *,
    r_max: float,
    n_bins: int = 28,
) -> dict:
    framed = _component_frame(pos, mass, cid, comp_id)
    if framed is None:
        r = np.linspace(0.0, r_max, n_bins)
        z = np.full(n_bins, np.nan)
        return {"kind": "empty", "r_mid": r, "y": z, "counts": np.zeros(n_bins, dtype=int)}
    p, m = framed
    if name in ("halo", "bulge"):
        # Spherical ρ(r) for quasi-spherical components (about component COM).
        r_min = 0.05 if name == "bulge" else 0.5
        prof = bin_spherical_density(
            p, m, n_bins=n_bins, r_max=r_max, log_bins=True, r_min=r_min
        )
        return {
            "kind": "rho",
            "ylabel": r"$\rho(r)$",
            "xlabel": r"$r$ [kpc]",
            "r_mid": np.asarray(prof.r_mid, dtype=np.float64),
            "y": np.asarray(prof.rho, dtype=np.float64),
            "counts": np.asarray(prof.counts, dtype=np.int64),
        }
    # Disk: cylindrical midplane surface density Σ(R).
    prof = bin_midplane_surface_density(p, m, n_bins=n_bins, r_max=r_max, z_max=0.5)
    sigma = np.asarray(prof.sigma, dtype=np.float64)
    counts = np.asarray(prof.counts, dtype=np.int64)
    # Empty rings → NaN so log plots do not draw pops to zero.
    sigma = np.where(counts > 0, sigma, np.nan)
    return {
        "kind": "sigma",
        "ylabel": r"$\Sigma(R)$",
        "xlabel": r"$R$ [kpc]",
        "r_mid": np.asarray(prof.r_mid, dtype=np.float64),
        "y": sigma,
        "counts": counts,
    }


def _snapshot_bundle(
    pos: np.ndarray,
    mass: np.ndarray,
    cid: np.ndarray,
    *,
    n_bins: int,
) -> dict:
    maps = {}
    profiles = {}
    for name, comp_id, _kind, half, r_max in COMPONENTS:
        maps[name] = _faceon_comp(pos, mass, cid, comp_id, n_bins=n_bins, half=half)
        profiles[name] = _profile_comp(pos, mass, cid, name, comp_id, r_max=r_max)
    return {"maps": maps, "profiles": profiles}


def _default_force_backend() -> str:
    """Prefer gpu_bh when CuPy/CUDA BH is available; else OpenMP bh_c."""
    try:
        from ntropy.forces.gpu_bh import gpu_bh_available

        if gpu_bh_available():
            return "gpu_bh"
    except Exception:  # noqa: BLE001
        pass
    return "bh_c"


def _evolve_component_tracked(
    parts: dict,
    *,
    end_gyr: float,
    dt: float,
    omp: int,
    timeout_s: float,
    n_track: int = 11,
    snap_times: list[float] | None = None,
    faceon_bins: int = 128,
    force: str | None = None,
    a2_r_eval: float | None = None,
) -> dict:
    """Leapfrog evolve; at snap times record per-component maps + profiles.

    ``a2_r_eval``: if set (e.g. disk scale length ``R_d``), track
    ``A₂(R=a2_r_eval)`` via ring interpolation instead of median ``A₂``.
    """
    os.environ["OMP_NUM_THREADS"] = str(max(1, int(omp)))
    n_steps = max(1, int(np.ceil(float(end_gyr) / max(float(dt), 1e-9))))
    pos0 = np.asarray(parts["pos"], dtype=np.float64)
    vel0 = np.asarray(parts["vel"], dtype=np.float64)
    mass = np.asarray(parts["mass"], dtype=np.float64)
    eps = np.asarray(parts["eps"], dtype=np.float64)
    cid = np.asarray(parts["component_id"])
    method = (force or _default_force_backend()).strip().lower()
    if method not in ("gpu_bh", "bh_c", "bh"):
        raise ValueError(f"unknown force backend {method!r}; use gpu_bh|bh_c|bh")

    want_t = [0.0, float(end_gyr)]
    if snap_times:
        want_t.extend(float(t) for t in snap_times)
    want_t = sorted({min(max(0.0, t), float(end_gyr)) for t in want_t})
    snap_steps = sorted(
        {
            0 if t <= 0 else n_steps if t >= float(end_gyr) else int(round(t / max(dt, 1e-9)))
            for t in want_t
        }
    )

    def _forces(p):
        nonlocal method
        if method == "gpu_bh":
            try:
                from ntropy.forces.gpu_bh import compute_forces_gpu_bh, gpu_bh_available

                if gpu_bh_available():
                    return compute_forces_gpu_bh(p, mass, eps, theta=0.8)
                print("  gpu_bh_available=False → falling back to bh_c", flush=True)
                method = "bh_c"
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  gpu_bh FAILED ({type(exc).__name__}: {exc}) → bh_c",
                    flush=True,
                )
                method = "bh_c"
        if method == "bh_c":
            try:
                from ntropy.forces.bhtree_c import compute_forces_bh_c, extension_available

                if extension_available():
                    return compute_forces_bh_c(p, mass, eps, theta=0.8)
            except Exception:  # noqa: BLE001
                method = "bh"
        import importlib

        bhtree = importlib.import_module("ntropy.forces.bhtree")
        return bhtree.compute_forces_bh(p, mass, eps, theta=0.8)

    # Probe once on the main thread so logs show the real backend + s/step.
    t_probe = time.time()
    _ = _forces(pos0)
    probe_s = time.time() - t_probe
    print(
        f"  force backend={method} N={len(pos0)} probe={probe_s:.3f}s "
        f"(~{probe_s:.2f} s/step; prior bh_c ~20–27 s/step)",
        flush=True,
    )

    track_steps = sorted(
        {0, n_steps}
        | {int(round(i * n_steps / max(n_track - 1, 1))) for i in range(n_track)}
        | set(snap_steps)
    )

    def _a2_now(pos):
        disk = cid == 0
        fout = disk_azimuthal_fourier(
            pos[disk],
            mass[disk],
            m=2,
            r_max=12.0,
            n_bins=12,
            z_max=0.5,
            min_count=10,
            r_eval=a2_r_eval,
        )
        if a2_r_eval is not None:
            return float(fout["a_m_over_a0_at_r"])
        return float(fout["a_m_over_a0_median"])

    def _run():
        t0 = time.time()
        pos = pos0.copy()
        vel = vel0.copy()
        times, a2s, coms = [], [], []
        snap_t, snaps = [], []
        snap_set = set(snap_steps)

        def _record(step: int):
            times.append(step * dt if step else 0.0)
            a2s.append(_a2_now(pos))
            coms.append(float(np.linalg.norm(np.average(pos, axis=0, weights=mass))))
            if step in snap_set:
                snap_t.append(step * dt if step else 0.0)
                snaps.append(_snapshot_bundle(pos, mass, cid, n_bins=faceon_bins))

        _record(0)
        # Reuse probe acceleration for the opening half-kick when possible.
        acc = _forces(pos)
        vel = vel + 0.5 * dt * acc
        next_i = 1
        for step in range(1, n_steps + 1):
            pos = pos + dt * vel
            acc = _forces(pos)
            vel = vel + dt * acc
            if next_i < len(track_steps) and step == track_steps[next_i]:
                _record(step)
                next_i += 1
        vel = vel - 0.5 * dt * acc
        return pos, vel, time.time() - t0, times, a2s, coms, snap_t, snaps

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            (
                pos_f,
                vel_f,
                wall,
                times,
                a2s,
                coms,
                snap_t,
                snaps,
            ) = ex.submit(_run).result(timeout=float(timeout_s))
    except FuturesTimeout:
        return {
            "ok": False,
            "timed_out": True,
            "timeout_s": float(timeout_s),
            "n_steps": n_steps,
            "error": f"evolve timed out after {timeout_s}s",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "timed_out": False,
            "n_steps": n_steps,
            "error": f"{type(exc).__name__}: {exc}",
        }

    com0 = np.average(pos0, axis=0, weights=mass)
    com1 = np.average(pos_f, axis=0, weights=mass)
    s_per_step = float(wall) / max(int(n_steps), 1)
    return {
        "ok": True,
        "wall_s": wall,
        "n_steps": n_steps,
        "force_method": method,
        "force_probe_s": float(probe_s),
        "s_per_step": s_per_step,
        "com_drift_kpc": float(np.linalg.norm(com1 - com0)),
        "t_gyr": [float(t) for t in times],
        "a2_t": [float(a) for a in a2s],
        "com_norm_t": [float(c) for c in coms],
        "snap_t_gyr": [float(t) for t in snap_t],
        "snaps": snaps,
        "pos_final": pos_f,
        "vel_final": vel_f,
        "mass": mass,
        "eps": eps,
        "component_id": cid,
    }


def _plot_a2_t(path: Path, report: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    arm_labels = report.get("arm_labels") or {}
    for tag, row in report.get("arms", {}).items():
        if not row.get("ok"):
            continue
        t = row.get("t_gyr")
        a2 = row.get("a2_t")
        if not t or not a2:
            continue
        ax.plot(t, a2, lw=1.8, label=arm_labels.get(tag, tag))
    ax.set_xlabel(r"$t$ [Gyr]")
    r_eval = report.get("a2_r_eval")
    if r_eval is not None:
        ax.set_ylabel(rf"disk $A_2(R={r_eval:g}\,\mathrm{{kpc}})$")
        title_a2 = rf"$A_2(R={r_eval:g})$"
    else:
        ax.set_ylabel(r"disk median $A_2$")
        title_a2 = r"median $A_2$"
    ax.set_title(
        rf"Disk {title_a2}$(t)$ ($\mathrm{{d}}t={report.get('dt')}$, "
        rf"$t_{{\rm end}}={report.get('evolve_gyr')}\,\mathrm{{Gyr}}$)"
    )
    ax.axhline(0.05, color="0.6", ls=":", lw=1, label=r"$A_2{=}0.05$")
    ax.axhline(0.30, color="0.4", ls="--", lw=1, label=r"$A_2{=}0.30$")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_faceon_grid(
    path: Path,
    snaps: list[dict],
    times: list[float],
    *,
    arm: str,
    title: str | None = None,
) -> None:
    """Rows = disk/bulge/halo, cols = times."""
    comp_names = [c[0] for c in COMPONENTS]
    n_r, n_c = len(comp_names), len(times)
    fig, axes = plt.subplots(n_r, n_c, figsize=(2.4 * n_c, 2.35 * n_r), squeeze=False)
    halves = {c[0]: c[3] for c in COMPONENTS}
    for i, name in enumerate(comp_names):
        # Floor from the full time stack so empty frames stay comparable.
        # All components use log10(Σ+ε); shared floor keeps time columns matched.
        stack_all = []
        for s in snaps:
            pos = s["maps"][name]
            pos = pos[pos > 0]
            if pos.size:
                stack_all.append(pos)
        if stack_all:
            cat = np.concatenate(stack_all)
            floor = max(
                float(np.percentile(cat, 20)),
                float(np.percentile(cat, 99.5)) * 1e-3,
                1e-8,
            )
        else:
            floor = 1e-3
        for j, t in enumerate(times):
            ax = axes[i, j]
            img = snaps[j]["maps"][name] if j < len(snaps) else snaps[-1]["maps"][name]
            pos = img[img > 0]
            vmax = float(np.percentile(pos, 99.5)) if pos.size else 1.0
            vmax = max(vmax, 10.0 * floor)
            show = np.log10(np.maximum(img, 0.0) + floor)
            vmin_s = np.log10(floor)
            vmax_s = np.log10(vmax + floor)
            # bin_plane_density / histogram2d layout is dens[i,j] = (x_i, y_j).
            # imshow treats [row,col] as (y,x), so transpose for sky axes.
            ax.imshow(
                show.T,
                origin="lower",
                cmap="inferno",
                vmin=vmin_s,
                vmax=vmax_s,
                extent=(-halves[name], halves[name], -halves[name], halves[name]),
            )
            if i == 0:
                ax.set_title(rf"$t={t:.3g}\,\mathrm{{Gyr}}$", fontsize=9)
            if j == 0:
                ax.set_ylabel(name, fontsize=10)
            if i == n_r - 1:
                ax.set_xlabel(r"$x$ [kpc]", fontsize=8)
            ax.tick_params(labelsize=7)
            if j > 0:
                ax.set_yticklabels([])
    fig.suptitle(
        title or rf"{arm}: component face-on $\Sigma$ evolution",
        y=1.01,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_profiles(
    path: Path,
    snaps: list[dict],
    times: list[float],
    *,
    arm: str,
    title: str | None = None,
) -> None:
    """One panel per component; curves = time slices."""
    comp_names = [c[0] for c in COMPONENTS]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), squeeze=False)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, max(len(times), 2)))
    for i, name in enumerate(comp_names):
        ax = axes[0, i]
        kind = snaps[0]["profiles"][name]["kind"]
        for j, t in enumerate(times):
            prof = snaps[j]["profiles"][name]
            r = np.asarray(prof["r_mid"], dtype=float)
            y = np.asarray(prof["y"], dtype=float)
            ok = np.isfinite(y) & (y > 0)
            if not np.any(ok):
                continue
            ax.plot(r[ok], y[ok], color=cmap[j], lw=1.6, label=rf"$t={t:.3g}$")
        ax.set_yscale("log")
        if kind == "rho":
            ax.set_xscale("log")
        ax.set_xlabel(snaps[0]["profiles"][name].get("xlabel", r"$R$ [kpc]"))
        ax.set_ylabel(snaps[0]["profiles"][name].get("ylabel", r"dens"))
        ax.set_title(f"{name} ({'ρ(r)' if kind == 'rho' else 'Σ(R)'})", fontsize=10)
        ax.legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle(title or rf"{arm}: radial density time slices", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _serialize_snaps(snaps: list[dict], times: list[float]) -> dict:
    """JSON-safe summary (no huge arrays); maps go to npz separately."""
    out = {"t_gyr": times, "components": {}}
    for name, *_ in COMPONENTS:
        out["components"][name] = {
            "profiles": [
                {
                    "kind": s["profiles"][name]["kind"],
                    "r_mid": [float(x) for x in s["profiles"][name]["r_mid"]],
                    "y": [
                        None if not np.isfinite(v) else float(v)
                        for v in s["profiles"][name]["y"]
                    ],
                    "counts": [int(c) for c in s["profiles"][name]["counts"]],
                }
                for s in snaps
            ],
            "faceon_half_kpc": next(c[3] for c in COMPONENTS if c[0] == name),
            "profile_rmax_kpc": next(c[4] for c in COMPONENTS if c[0] == name),
        }
    return out


def _save_maps_npz(path: Path, snaps: list[dict], times: list[float], arm: str) -> None:
    payload = {
        "t_gyr": np.asarray(times, dtype=np.float64),
        "arm": np.asarray(arm),
    }
    for name, *_ in COMPONENTS:
        for i, s in enumerate(snaps):
            payload[f"{name}_map{i}"] = np.asarray(s["maps"][name], dtype=np.float32)
            payload[f"{name}_r{i}"] = np.asarray(s["profiles"][name]["r_mid"], dtype=np.float64)
            payload[f"{name}_y{i}"] = np.asarray(s["profiles"][name]["y"], dtype=np.float64)
            payload[f"{name}_kind"] = np.asarray(s["profiles"][name]["kind"])
    np.savez_compressed(path, **payload)


def _load_maps_npz(path: Path) -> tuple[list[dict], list[float], str]:
    """Rebuild snap bundles from ``component_maps_*.npz`` (maps + profiles)."""
    z = np.load(path, allow_pickle=True)
    times = [float(t) for t in np.asarray(z["t_gyr"]).ravel()]
    arm = str(np.asarray(z["arm"]).item()) if "arm" in z.files else path.stem
    snaps: list[dict] = []
    for i in range(len(times)):
        maps, profiles = {}, {}
        for name, *_rest in COMPONENTS:
            maps[name] = np.asarray(z[f"{name}_map{i}"], dtype=np.float32)
            kind = (
                str(np.asarray(z[f"{name}_kind"]).item())
                if f"{name}_kind" in z.files
                else ("rho" if name == "halo" else "sigma")
            )
            r = np.asarray(z[f"{name}_r{i}"], dtype=np.float64)
            y = np.asarray(z[f"{name}_y{i}"], dtype=np.float64)
            profiles[name] = {
                "kind": kind,
                "ylabel": r"$\rho(r)$" if kind == "rho" else r"$\Sigma(R)$",
                "xlabel": r"$r$ [kpc]" if kind == "rho" else r"$R$ [kpc]",
                "r_mid": r,
                "y": y,
                "counts": np.zeros(len(r), dtype=np.int64),
            }
        snaps.append({"maps": maps, "profiles": profiles})
    return snaps, times, arm


def _plot_profiles_compare(
    path: Path,
    snaps_a: list[dict],
    snaps_b: list[dict],
    times: list[float],
    *,
    label_a: str,
    label_b: str,
    title: str | None = None,
) -> None:
    """Side-by-side radial profiles: solid = A, dashed = B, colour = time."""
    comp_names = [c[0] for c in COMPONENTS]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), squeeze=False)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, max(len(times), 2)))
    for i, name in enumerate(comp_names):
        ax = axes[0, i]
        kind = snaps_a[0]["profiles"][name]["kind"]
        for j, t in enumerate(times):
            color = cmap[j]
            for snaps, ls, lab in (
                (snaps_a, "-", label_a if j == 0 else None),
                (snaps_b, "--", label_b if j == 0 else None),
            ):
                if j >= len(snaps):
                    continue
                prof = snaps[j]["profiles"][name]
                r = np.asarray(prof["r_mid"], dtype=float)
                y = np.asarray(prof["y"], dtype=float)
                ok = np.isfinite(y) & (y > 0)
                if not np.any(ok):
                    continue
                ax.plot(
                    r[ok],
                    y[ok],
                    color=color,
                    ls=ls,
                    lw=1.6,
                    label=lab if lab else (rf"$t={t:.3g}$" if ls == "-" else None),
                )
        # Prefer compact: time colours + style for A/B.
        time_handles = [
            plt.Line2D([0], [0], color=cmap[j], lw=1.6, label=rf"$t={t:.3g}$")
            for j, t in enumerate(times)
        ]
        style_handles = [
            plt.Line2D([0], [0], color="0.2", lw=1.6, ls="-", label=label_a),
            plt.Line2D([0], [0], color="0.2", lw=1.6, ls="--", label=label_b),
        ]
        ax.legend(
            handles=time_handles + style_handles,
            frameon=False,
            fontsize=6.5,
            loc="best",
            ncol=1,
        )
        ax.set_yscale("log")
        if kind == "rho":
            ax.set_xscale("log")
        ax.set_xlabel(snaps_a[0]["profiles"][name].get("xlabel", r"$R$ [kpc]"))
        ax.set_ylabel(snaps_a[0]["profiles"][name].get("ylabel", r"dens"))
        ax.set_title(f"{name} ({'ρ(r)' if kind == 'rho' else 'Σ(R)'})", fontsize=10)
    fig.suptitle(
        title or rf"{label_a} vs {label_b}: density profiles",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("runs/ml/field_maps/evolve_component_slices_2026-07-26"),
    )
    p.add_argument("--teacher", type=Path, default=None)
    p.add_argument("--rank", type=Path, default=RANK)
    p.add_argument("--n-bar", type=int, default=12)
    p.add_argument("--n-quiet", type=int, default=8)
    p.add_argument("--n-mid", type=int, default=4)
    p.add_argument("--n-rot-bar", type=int, default=2)
    p.add_argument("--bar-floor", type=float, default=0.25)
    p.add_argument("--quiet-ceil", type=float, default=0.05)
    p.add_argument("--enc-grid", type=int, default=4)
    p.add_argument("--n-pc", type=int, default=64)
    p.add_argument("--n-disk", type=int, default=1_000_000)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--evolve-gyr", type=float, default=0.50)
    p.add_argument("--omp", type=int, default=2)
    p.add_argument(
        "--force",
        type=str,
        default=None,
        choices=("gpu_bh", "bh_c", "bh"),
        help=(
            "N-body force backend (default: gpu_bh when CUDA/CuPy BH available, "
            "else bh_c OpenMP)."
        ),
    )
    p.add_argument("--timeout-s", type=float, default=7200.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--save-final-particles",
        action="store_true",
        help="Write particles_{arm}_t0.npz and particles_{arm}_final.npz for kinetic scoring.",
    )
    p.add_argument("--data-path", type=Path, default=None)
    p.add_argument(
        "--data-arm-label",
        type=str,
        default=None,
        help=(
            "Legend/caption for the data arm (default: 'data dump'). "
            "Use e.g. 'quiet IC (control)' when the data arm is a quiet "
            "GalactICS IC rather than a barred evolved dump."
        ),
    )
    p.add_argument(
        "--methods",
        type=str,
        default="amplify_knn_hybrid",
        help="Comma list of generative methods (data always included). Empty = none.",
    )
    p.add_argument("--ic-a2-min", type=float, default=0.30)
    p.add_argument("--ic-a2-tries", type=int, default=8)
    p.add_argument("--n-track", type=int, default=11)
    p.add_argument(
        "--faceon-times",
        type=str,
        default="0,0.12,0.25,0.38,0.5",
    )
    p.add_argument("--faceon-bins", type=int, default=128)
    p.add_argument(
        "--a2-r-eval",
        type=float,
        default=None,
        help=(
            "If set, track particle A₂ at this cylindrical radius [kpc] "
            "(linear interp of ring A₂; e.g. disk scale length R_d) "
            "instead of median A₂ over rings. Default: median (legacy)."
        ),
    )
    p.add_argument("--paper-figures", type=Path, default=None)
    p.add_argument(
        "--paper-prefix",
        type=str,
        default="fig_evolve_component",
        help="Paper figure basename prefix (e.g. fig_component_nonic).",
    )
    p.add_argument("--skip-recon", action="store_true", default=True)
    p.add_argument(
        "--with-recon",
        action="store_true",
        help="Also evolve FFT teacher recon arm (encode→decode→resample).",
    )
    p.add_argument(
        "--with-hybrid-bulge",
        action="store_true",
        help=(
            "Also evolve hybrid IC: FFT recon disk+halo, retain GalactICS "
            "bulge particles (cusp fidelity fallback)."
        ),
    )
    p.add_argument(
        "--with-shell-bulge",
        action="store_true",
        help=(
            "Also evolve FFT recon disk/halo + spherical-shell bulge resample "
            "(cusp-preserving without retaining exact particles)."
        ),
    )
    p.add_argument(
        "--with-input-dens-shell",
        action="store_true",
        help=(
            "Also evolve deposited dens (disk/halo) + teacher moments + "
            "spherical-shell bulge. Closes IC A₂ gap from soft AE dens."
        ),
    )
    p.add_argument(
        "--with-deposit-shell",
        action="store_true",
        help=(
            "Also evolve fully deposited dens+moments + spherical-shell bulge "
            "(oracle recon upper bound)."
        ),
    )
    p.add_argument(
        "--with-dens-amp-shell",
        action="store_true",
        help=(
            "Also evolve AE dens with axisym-residual amplify (generative) + "
            "shell bulge. Uses --dens-resid-alpha."
        ),
    )
    p.add_argument(
        "--with-residual-f0",
        action="store_true",
        help=(
            "Residual around GalactICS f0 (paint or full_dyn; see --residual-recipe). "
            "Requires --ic-path."
        ),
    )
    p.add_argument(
        "--residual-recipe",
        choices=("paint", "full_dyn_replace", "full_dyn_ot"),
        default="paint",
        help=(
            "paint=m2 on f0 axisym; full_dyn_replace=data disk⊕f0 halo/bulge; "
            "full_dyn_ot=OT-lite f0→data disk⊕f0 halo/bulge."
        ),
    )
    p.add_argument(
        "--ic-path",
        type=Path,
        default=None,
        help="GalactICS f0 IC (ic_state.npz) for --with-residual-f0.",
    )
    p.add_argument(
        "--residual-morph-source",
        choices=("teacher_recon", "deposit", "phase_b", "blend"),
        default="teacher_recon",
        help="Morph chart for residual-f0: AE recon, deposit, Phase-B δ, or blend.",
    )
    p.add_argument(
        "--phase-b-ckpt",
        type=Path,
        default=None,
        help="Phase-B delta_midplane.pt (required for --residual-morph-source phase_b).",
    )
    p.add_argument(
        "--phase-b-hint",
        choices=("deposit", "teacher", "blend"),
        default="blend",
        help="Morph dens chart for CondDelta hint channel.",
    )
    p.add_argument(
        "--residual-blend-weight",
        type=float,
        default=0.5,
        help="Deposit weight in teacher↔deposit blend (morph_source=blend / phase_b hint).",
    )
    p.add_argument(
        "--residual-morph-path",
        type=Path,
        default=None,
        help=(
            "Dump used as morph chart for --with-residual-f0 "
            "(default: --data-path). Set separately when data arm is a quiet "
            "control IC (e.g. no-bulge) but morph comes from a barred dump."
        ),
    )
    p.add_argument(
        "--residual-velocity-mode",
        choices=(
            "transplant",
            "moments",
            "morph_transplant",
            "morph_blend",
            "morph_hybrid",
            "f0",
            "morph",
            "blend",
            "hybrid",
        ),
        default="transplant",
        help=(
            "Disk velocities for residual-f0: f0/transplant=GalactICS kNN; "
            "morph/morph_transplant=barred morph dump kNN; blend=mix; "
            "hybrid=morph R<=--residual-vel-hybrid-r-max else f0; "
            "moments=f0 dens-map moments."
        ),
    )
    p.add_argument(
        "--residual-morph-vel-blend-weight",
        type=float,
        default=0.5,
        help="Morph weight for residual-velocity-mode=morph_blend (0=f0, 1=morph).",
    )
    p.add_argument(
        "--residual-vel-hybrid-r-max",
        type=float,
        default=5.0,
        help="For residual-velocity-mode=morph_hybrid: morph vel for R<=this [kpc].",
    )
    p.add_argument(
        "--residual-scale",
        choices=("multiplicative", "additive"),
        default="multiplicative",
        help="How morph residual contrast is applied onto GalactICS f0 dens.",
    )
    p.add_argument(
        "--residual-r-weight-peak",
        type=float,
        default=None,
        help="Gaussian radial weight peak [kpc] on morph contrast (e.g. R_d=2).",
    )
    p.add_argument(
        "--residual-r-weight-sigma",
        type=float,
        default=None,
        help="Gaussian radial weight σ [kpc] (requires --residual-r-weight-peak).",
    )
    p.add_argument(
        "--residual-r-weight-floor",
        type=float,
        default=0.0,
        help="Min radial weight outside peak (0=hard Gaussian).",
    )
    p.add_argument(
        "--residual-contrast-sharpen",
        type=float,
        default=0.0,
        help="Unsharp-mask amount on morph contrast (0=off; ~0.75–1.5 for soft teacher).",
    )
    p.add_argument(
        "--residual-contrast-smooth-kpc",
        type=float,
        default=1.5,
        help="Unsharp smooth scale [kpc] for --residual-contrast-sharpen.",
    )
    p.add_argument(
        "--residual-alpha-mode",
        choices=("fixed", "match_a2_rd"),
        default="fixed",
        help="fixed: --dens-resid-alpha; match_a2_rd: search α for map A₂(R_d).",
    )
    p.add_argument(
        "--residual-target-a2-rd",
        type=float,
        default=None,
        help="Target A₂(R_d) for --residual-alpha-mode match_a2_rd (default 0.49).",
    )
    p.add_argument(
        "--dens-resid-alpha",
        type=float,
        default=2.5,
        help="Non-axisym dens residual amplify for --with-dens-amp-shell (α=1 off).",
    )
    p.add_argument(
        "--dens-resid-mode",
        choices=("fixed", "residual_power", "map_a2"),
        default="fixed",
        help=(
            "fixed: --dens-resid-alpha; residual_power: match dens residual RMS; "
            "map_a2: search α to match midplane dens map A₂ (see --dens-resid-target-a2)."
        ),
    )
    p.add_argument(
        "--dens-resid-target-a2",
        type=float,
        default=None,
        help="Target midplane dens map A₂ for --dens-resid-mode map_a2 (default: deposit map A₂).",
    )
    p.add_argument(
        "--dens-resid-midplane-only",
        action="store_true",
        help="Amplify only central disk dens slabs (fragile-system safeguard).",
    )
    p.add_argument(
        "--dens-resid-kind",
        choices=("full", "m2"),
        default="full",
        help=(
            "full: amplify entire dens residual; m2: amplify only m=2 Fourier "
            "projection (safer when AE residual cosine vs deposit is low)."
        ),
    )
    p.add_argument(
        "--dens-resid-other-alpha",
        type=float,
        default=1.0,
        help="Scale for non-m2 residual when --dens-resid-kind m2 (1=unchanged).",
    )
    p.add_argument(
        "--moments-source",
        choices=("auto", "recon", "deposit"),
        default="auto",
        help="Moment channel source for dens-amp arm (deposit=AE dens+deposit mom ablation).",
    )
    p.add_argument(
        "--velocity-frame",
        choices=("cartesian", "cylindrical"),
        default="cartesian",
        help="Resample velocity draw frame (cylindrical = DF option B1).",
    )
    p.add_argument(
        "--match-cell-moments",
        action="store_true",
        help="Affine-correct per-cell mean/σ after resample (DF option B2).",
    )
    p.add_argument(
        "--reuse-data-from",
        type=Path,
        default=None,
        help=(
            "Reuse prior data-arm maps/profiles from this run dir "
            "(needs component_maps_data.npz); skip data evolve."
        ),
    )
    p.add_argument(
        "--skip-data",
        action="store_true",
        help="Do not build/evolve the data arm (use with --reuse-data-from).",
    )
    p.add_argument(
        "--extra-arm-npz",
        type=Path,
        default=None,
        help=(
            "Optional pre-built particle NPZ arm (pos/vel/mass/component_id[/eps]). "
            "Stratified to evolve N. Use for latent particle_retrieve ICs."
        ),
    )
    p.add_argument(
        "--extra-arm-name",
        type=str,
        default="latent_gen",
        help="Arm tag for --extra-arm-npz (default: latent_gen).",
    )
    p.add_argument(
        "--extra-arm-label",
        type=str,
        default=None,
        help="Legend label for --extra-arm-npz (default: arm name).",
    )
    args = p.parse_args()
    if args.with_recon:
        args.skip_recon = False
    if args.teacher is None:
        args.teacher = default_teacher()

    has_bulge = _ic_has_bulge(args.ic_path) if args.ic_path is not None else True
    n_tot = _n_total_for_disk(args.n_disk, has_bulge=has_bulge)
    n_ev = n_tot
    n_resample = n_tot
    if has_bulge:
        mix = f"halo≈{args.n_disk // 2:,}, bulge≈{args.n_disk // 4:,}"
    else:
        mix = f"halo≈{args.n_disk // 2:,}, bulge=0 (disk+halo mix 2:1)"
    print(
        f"n-disk={args.n_disk:,} → total N={n_tot:,} ({mix})",
        flush=True,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    # Keep CUDA visible for AE encode/decode and optional gpu_bh forces.
    # (Previously blanked CUDA_VISIBLE_DEVICES so corpus OpenMP jobs owned the GPU.)
    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] == "":
        del os.environ["CUDA_VISIBLE_DEVICES"]
    os.environ["OMP_NUM_THREADS"] = str(args.omp)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    ranked = json.loads(Path(args.rank).read_text())
    ranked.sort(key=lambda r: r["a2"], reverse=True)
    data_path = args.data_path or Path(ranked[0]["path"])
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    faceon_times = [float(x) for x in args.faceon_times.split(",") if x.strip()]
    if args.evolve_gyr not in faceon_times:
        faceon_times.append(float(args.evolve_gyr))
    faceon_times = sorted(set(faceon_times))

    print(f"=== teacher {args.teacher} ===", flush=True)
    teacher, cfg, stats = load_frozen_teacher_bundle(args.teacher)
    print(
        f"=== AE device={next(teacher.parameters()).device} "
        f"cuda={torch.cuda.is_available()} ===",
        flush=True,
    )

    lib = None
    if methods:
        print("=== build stratified library ===", flush=True)
        lib = build_library(args)

    arms: dict[str, dict] = {}
    reused_snaps: dict[str, tuple[list[dict], list[float]]] = {}
    data_arm_label = (
        str(args.data_arm_label).strip()
        if args.data_arm_label
        else "data dump"
    )
    labels = {
        "data": data_arm_label,
        "fft_recon": "FFT recon",
        "fft_recon_keep_bulge": "FFT recon + GalactICS bulge",
        "fft_recon_shell_bulge": "FFT recon + shell bulge",
        "fft_recon_input_dens_shell": "input dens + AE moments + shell bulge",
        "fft_recon_deposit_shell": "deposit dens+moments + shell bulge",
        "fft_recon_dens_amp_shell": "AE dens resid amplify + shell bulge",
        "residual_f0": "GalactICS f0 + morph dens residual",
        "amplify_knn_hybrid": "amplify_knn_hybrid",
    }
    if args.extra_arm_npz is not None:
        extra_tag = str(args.extra_arm_name).strip() or "latent_gen"
        labels[extra_tag] = (
            str(args.extra_arm_label).strip()
            if args.extra_arm_label
            else extra_tag
        )
    resample_kw = {
        "velocity_frame": str(args.velocity_frame),
        "match_cell_moments": bool(args.match_cell_moments),
    }

    reuse_dir = args.reuse_data_from
    if reuse_dir is not None:
        maps_npz = Path(reuse_dir) / "component_maps_data.npz"
        if not maps_npz.is_file():
            raise SystemExit(f"--reuse-data-from missing {maps_npz}")
        print(f"=== reuse data maps ← {maps_npz} ===", flush=True)
        snaps_d, times_d, _ = _load_maps_npz(maps_npz)
        reused_snaps["data"] = (snaps_d, times_d)
        args.skip_data = True
        # Prefer recorded snap times so FFT recon aligns with reused data.
        faceon_times = sorted(set(float(t) for t in times_d) | {float(args.evolve_gyr)})

    if not args.skip_data:
        print(f"=== arm data ← {data_path} (N={n_ev}) ===", flush=True)
        data_ev = _full_com(_subsample_dump(data_path, n_ev, rng))
        arms["data"] = {"parts": data_ev, "meta": {"path": str(data_path)}}

    if not args.skip_recon:
        print("=== arm fft_recon (teacher encode→decode→resample) ===", flush=True)
        recon = _recon_particles(
            data_path, teacher, cfg, stats, n_resample=n_resample, rng=rng, **resample_kw
        )
        arms["fft_recon"] = {
            "parts": _vcom_only(_stratified_down(recon, n_ev, rng)),
            "meta": {"teacher": str(args.teacher), "recipe": "fft_recon"},
        }

    if args.with_hybrid_bulge:
        print(
            "=== arm fft_recon_keep_bulge (FFT disk/halo + GalactICS bulge) ===",
            flush=True,
        )
        hybrid = _recon_particles(
            data_path,
            teacher,
            cfg,
            stats,
            n_resample=n_resample,
            rng=rng,
            retain_components=("bulge",),
            **resample_kw,
        )
        arms["fft_recon_keep_bulge"] = {
            "parts": _vcom_only(_stratified_down(hybrid, n_ev, rng)),
            "meta": {
                "teacher": str(args.teacher),
                "recipe": "fft_recon_keep_bulge",
                "retained": ["bulge"],
            },
        }

    if args.with_shell_bulge:
        print(
            "=== arm fft_recon_shell_bulge (FFT disk/halo + spherical-shell bulge) ===",
            flush=True,
        )
        shell = _recon_particles(
            data_path,
            teacher,
            cfg,
            stats,
            n_resample=n_resample,
            rng=rng,
            bulge_method="spherical_shells",
            **resample_kw,
        )
        arms["fft_recon_shell_bulge"] = {
            "parts": _vcom_only(_stratified_down(shell, n_ev, rng)),
            "meta": {
                "teacher": str(args.teacher),
                "recipe": "fft_recon_shell_bulge",
                "bulge_method": "spherical_shells",
            },
        }

    if args.with_input_dens_shell:
        print(
            "=== arm fft_recon_input_dens_shell "
            "(deposited dens + AE moments + shell bulge) ===",
            flush=True,
        )
        inj = _recon_particles(
            data_path,
            teacher,
            cfg,
            stats,
            n_resample=n_resample,
            rng=rng,
            bulge_method="spherical_shells",
            dens_source="input",
            dens_components=("disk", "halo"),
            **resample_kw,
        )
        arms["fft_recon_input_dens_shell"] = {
            "parts": _vcom_only(_stratified_down(inj, n_ev, rng)),
            "meta": {
                "teacher": str(args.teacher),
                "recipe": "fft_recon_input_dens_shell",
                "bulge_method": "spherical_shells",
                "dens_source": "input",
                "dens_components": ["disk", "halo"],
            },
        }

    if args.with_deposit_shell:
        print(
            "=== arm fft_recon_deposit_shell "
            "(deposited dens+moments + shell bulge) ===",
            flush=True,
        )
        dep = _recon_particles(
            data_path,
            teacher,
            cfg,
            stats,
            n_resample=n_resample,
            rng=rng,
            bulge_method="spherical_shells",
            dens_source="deposit",
            **resample_kw,
        )
        arms["fft_recon_deposit_shell"] = {
            "parts": _vcom_only(_stratified_down(dep, n_ev, rng)),
            "meta": {
                "teacher": str(args.teacher),
                "recipe": "fft_recon_deposit_shell",
                "bulge_method": "spherical_shells",
                "dens_source": "deposit",
            },
        }

    if getattr(args, "with_dens_amp_shell", False):
        alpha = float(args.dens_resid_alpha)
        mode = str(args.dens_resid_mode)
        mid_only = bool(args.dens_resid_midplane_only)
        mom_src = str(args.moments_source)
        print(
            f"=== arm fft_recon_dens_amp_shell "
            f"(AE dens resid mode={mode} α={alpha} midplane_only={mid_only} "
            f"moments={mom_src} + shell bulge) ===",
            flush=True,
        )
        amp = _recon_particles(
            data_path,
            teacher,
            cfg,
            stats,
            n_resample=n_resample,
            rng=rng,
            bulge_method="spherical_shells",
            dens_source="recon",
            dens_resid_alpha=alpha,
            dens_resid_components=("disk",),
            dens_resid_mode=mode,
            dens_resid_midplane_only=mid_only,
            dens_resid_target_a2=args.dens_resid_target_a2,
            dens_resid_kind=str(args.dens_resid_kind),
            dens_resid_other_alpha=float(args.dens_resid_other_alpha),
            moments_source=mom_src,
            **resample_kw,
        )
        alpha_used = float(amp.get("dens_resid_alpha_used", alpha))
        print(f"  dens_resid_alpha_used={alpha_used:.3f}", flush=True)
        arms["fft_recon_dens_amp_shell"] = {
            "parts": _vcom_only(_stratified_down(amp, n_ev, rng)),
            "meta": {
                "teacher": str(args.teacher),
                "recipe": "fft_recon_dens_amp_shell",
                "bulge_method": "spherical_shells",
                "dens_source": "recon",
                "dens_resid_alpha": alpha,
                "dens_resid_alpha_used": alpha_used,
                "dens_resid_mode": mode,
                "dens_resid_kind": str(args.dens_resid_kind),
                "dens_resid_other_alpha": float(args.dens_resid_other_alpha),
                "dens_resid_midplane_only": mid_only,
                "moments_source": mom_src,
                "dens_resid_components": ["disk"],
            },
        }

    if getattr(args, "with_residual_f0", False):
        if args.ic_path is None:
            raise SystemExit("--with-residual-f0 requires --ic-path (GalactICS f0)")
        alpha = float(args.dens_resid_alpha)
        morph_path = args.residual_morph_path or data_path
        recipe = str(getattr(args, "residual_recipe", "paint")).lower().strip()
        print(
            f"=== arm residual_f0 "
            f"(recipe={recipe} f0={args.ic_path} morph={args.residual_morph_source} "
            f"morph_path={morph_path} "
            f"α={alpha} kind={args.dens_resid_kind} "
            f"vel={args.residual_velocity_mode}) ===",
            flush=True,
        )
        if recipe in ("full_dyn_replace", "full_dyn_ot", "replace", "ot_lite"):
            mode = (
                "replace"
                if recipe in ("full_dyn_replace", "replace")
                else "ot_lite"
            )
            res = _full_dyn_residual_particles(
                args.ic_path,
                morph_path,
                n_resample=n_resample,
                rng=rng,
                mode=mode,
            )
        else:
            res = _residual_f0_particles(
                args.ic_path,
                morph_path,
                teacher,
                cfg,
                stats,
                n_resample=n_resample,
                rng=rng,
                morph_source=str(args.residual_morph_source),
                alpha=alpha,
                dens_resid_kind=str(args.dens_resid_kind),
                dens_resid_other_alpha=float(args.dens_resid_other_alpha),
                dens_resid_midplane_only=bool(args.dens_resid_midplane_only),
                residual_scale=str(args.residual_scale),
                velocity_mode=str(args.residual_velocity_mode),
                phase_b_ckpt=getattr(args, "phase_b_ckpt", None),
                phase_b_hint=str(getattr(args, "phase_b_hint", "blend")),
                blend_weight=float(getattr(args, "residual_blend_weight", 0.5)),
                morph_vel_blend_weight=float(
                    getattr(args, "residual_morph_vel_blend_weight", 0.5)
                ),
                vel_hybrid_r_max=float(
                    getattr(args, "residual_vel_hybrid_r_max", 5.0)
                ),
                r_weight_peak=getattr(args, "residual_r_weight_peak", None),
                r_weight_sigma=getattr(args, "residual_r_weight_sigma", None),
                contrast_sharpen=float(getattr(args, "residual_contrast_sharpen", 0.0)),
                contrast_smooth_kpc=float(
                    getattr(args, "residual_contrast_smooth_kpc", 1.5)
                ),
                r_weight_floor=float(getattr(args, "residual_r_weight_floor", 0.0)),
                alpha_mode=str(getattr(args, "residual_alpha_mode", "fixed")),
                target_a2_rd=getattr(args, "residual_target_a2_rd", None),
                a2_r_eval=getattr(args, "a2_r_eval", None),
                **resample_kw,
            )
        arms["residual_f0"] = {
            "parts": _vcom_only(_stratified_down(res, n_ev, rng)),
            "meta": {
                "teacher": str(args.teacher) if args.teacher else None,
                "recipe": recipe,
                "phase": res.get("phase"),
                "ic_path": str(args.ic_path),
                "morph_path": str(morph_path),
                "morph_source": str(args.residual_morph_source),
                "phase_b_ckpt": str(args.phase_b_ckpt) if args.phase_b_ckpt else None,
                "dens_resid_alpha": float(res.get("dens_resid_alpha_used", alpha)),
                "dens_resid_kind": str(args.dens_resid_kind),
                "dens_resid_other_alpha": float(args.dens_resid_other_alpha),
                "residual_scale": str(args.residual_scale),
                "velocity_mode": str(
                    res.get("velocity_meta", {}).get(
                        "velocity_mode", args.residual_velocity_mode
                    )
                ),
                "r_weight_peak": res.get("r_weight_peak"),
                "r_weight_sigma": res.get("r_weight_sigma"),
                "contrast_sharpen": res.get("contrast_sharpen"),
                "alpha_mode": res.get("alpha_mode"),
                "target_a2_rd": res.get("target_a2_rd"),
                "retain_components": res.get("retain_components"),
            },
        }

    for method in methods:
        print(f"=== arm {method} ===", flush=True)
        assert lib is not None
        best = None
        for attempt in range(max(1, args.ic_a2_tries)):
            gen, meta = _sample_gen(lib, method, rng, n_resample)
            gen_ev = _vcom_only(_stratified_down(gen, n_ev, rng))
            a2 = float(_a2_disk(gen_ev))
            print(f"  try#{attempt} IC A2={a2:.3f}", flush=True)
            if best is None or a2 > best[0]:
                best = (a2, gen_ev, meta)
            if a2 >= args.ic_a2_min:
                break
        assert best is not None
        arms[method] = {
            "parts": best[1],
            "meta": {k: v for k, v in best[2].items() if k != "z"},
        }

    if args.extra_arm_npz is not None:
        extra_tag = str(args.extra_arm_name).strip() or "latent_gen"
        extra_path = Path(args.extra_arm_npz)
        if not extra_path.is_file():
            raise SystemExit(f"--extra-arm-npz missing: {extra_path}")
        print(
            f"=== arm {extra_tag} ← {extra_path} (stratified to N={n_ev}) ===",
            flush=True,
        )
        extra = _full_com(_subsample_dump(extra_path, n_ev, rng))
        arms[extra_tag] = {
            "parts": extra,
            "meta": {
                "recipe": "extra_arm_npz",
                "path": str(extra_path),
                "label": labels.get(extra_tag, extra_tag),
            },
        }

    report = {
        "dt": args.dt,
        "evolve_gyr": args.evolve_gyr,
        "n_steps": int(np.ceil(args.evolve_gyr / args.dt)),
        "n_disk": int(args.n_disk),
        "n_evolve": n_ev,
        "omp": args.omp,
        "force": args.force or _default_force_backend(),
        "force_requested": args.force,
        "teacher": str(args.teacher),
        "data_path": str(data_path),
        "snap_times_gyr": faceon_times,
        "faceon_bins": args.faceon_bins,
        "a2_r_eval": args.a2_r_eval,
        "a2_metric": (
            f"A2(R={args.a2_r_eval:g} kpc) interp"
            if args.a2_r_eval is not None
            else "median_ring_A2"
        ),
        "component_fov_half_kpc": {c[0]: c[3] for c in COMPONENTS},
        "profile_rmax_kpc": {c[0]: c[4] for c in COMPONENTS},
        "count_mix": "disk:halo:bulge ≈ 4:2:1",
        "shared_centering": (
            "Evolve frame: data uses full soft COM; dens-resampled fft_recon/gen "
            "keep morphological origin + VCOM-only. Maps/profiles recenter each "
            "component on its own mass-weighted COM (halo dominates global COM)."
        ),
        "profile_convention": (
            "disk: Σ(R) midplane cylinders; bulge/halo: spherical ρ(r); "
            "all about per-component COM."
        ),
        "note": (
            "Per-component face-on Σ + radial profiles at evolve snapshots. "
            "Maps/profiles computed on-the-fly (no full particle archive). "
            "Primary generative dynamical-consistency arm: fft_recon."
        ),
        "arm_labels": dict(labels),
        "arms": {},
        "figures": {},
    }
    # Preserve prior arms in the same out dir (e.g. hybrid) when appending fft_recon.
    prior_json = args.out / "verdict.json"
    if prior_json.is_file():
        try:
            prior = json.loads(prior_json.read_text())
            report["arms"].update(prior.get("arms") or {})
            report["figures"].update(prior.get("figures") or {})
        except Exception:  # noqa: BLE001
            pass

    kept_snaps: dict[str, tuple[list[dict], list[float]]] = {}

    for tag, (snaps, snap_t) in reused_snaps.items():
        face_png = args.out / f"faceon_components_{tag}.png"
        prof_png = args.out / f"profiles_components_{tag}.png"
        if not face_png.is_file() or not prof_png.is_file():
            _plot_faceon_grid(
                face_png,
                snaps,
                snap_t,
                arm=labels.get(tag, tag),
                title=(
                    rf"{labels.get(tag, tag)}: face-on $\Sigma$ by component "
                    rf"($N_{{\rm disk}}={args.n_disk:,}$, $N_{{\rm tot}}={n_ev:,}$)"
                ),
            )
            _plot_profiles(
                prof_png,
                snaps,
                snap_t,
                arm=labels.get(tag, tag),
                title=(
                    rf"{labels.get(tag, tag)}: density profiles "
                    rf"($t_{{\rm end}}={args.evolve_gyr}\,\mathrm{{Gyr}}$)"
                ),
            )
        maps_npz = args.out / f"component_maps_{tag}.npz"
        if maps_npz.resolve() != (Path(reuse_dir) / f"component_maps_{tag}.npz").resolve():
            _save_maps_npz(maps_npz, snaps, snap_t, tag)
        elif not maps_npz.is_file():
            shutil.copy2(Path(reuse_dir) / f"component_maps_{tag}.npz", maps_npz)

        prior_row = report["arms"].get(tag, {})
        row = {
            "ok": True,
            "reused": True,
            "reuse_from": str(reuse_dir),
            "pre": prior_row.get("pre"),
            "a2_pre": prior_row.get("a2_pre"),
            "a2_post": prior_row.get("a2_post"),
            "com_drift_kpc": prior_row.get("com_drift_kpc", 0.0),
            "n_steps": prior_row.get("n_steps", report["n_steps"]),
            "force_method": prior_row.get("force_method", "bh_c"),
            "wall_s": prior_row.get("wall_s", 0.0),
            "t_gyr": prior_row.get("t_gyr"),
            "a2_t": prior_row.get("a2_t"),
            "com_norm_t": prior_row.get("com_norm_t"),
            "snap_t_gyr": snap_t,
            "snapshots": _serialize_snaps(snaps, snap_t),
            "meta": prior_row.get("meta", {"path": str(data_path)}),
            "figures": {
                "faceon": str(face_png),
                "profiles": str(prof_png),
                "maps_npz": str(maps_npz),
            },
        }
        report["arms"][tag] = row
        report["figures"][f"{tag}_faceon"] = str(face_png)
        report["figures"][f"{tag}_profiles"] = str(prof_png)
        kept_snaps[tag] = (snaps, snap_t)
        print(f"=== reused {tag} snaps={len(snap_t)} times={snap_t} ===", flush=True)

    for tag, pack in arms.items():
        parts = pack["parts"]
        pre = _metrics(parts)
        print(
            f"=== evolve {tag} A2={pre['a2']:.3f} N={parts['pos'].shape[0]} ===",
            flush=True,
        )
        evo = _evolve_component_tracked(
            parts,
            end_gyr=args.evolve_gyr,
            dt=args.dt,
            omp=args.omp,
            timeout_s=args.timeout_s,
            n_track=args.n_track,
            snap_times=faceon_times,
            faceon_bins=args.faceon_bins,
            force=args.force,
            a2_r_eval=args.a2_r_eval,
        )
        if not evo.get("ok"):
            report["arms"][tag] = {
                "ok": False,
                "pre": pre,
                "meta": pack["meta"],
                "evolve": {k: v for k, v in evo.items() if k != "snaps"},
            }
            print(f"  FAIL {evo.get('error')}", flush=True)
            continue

        if getattr(args, "save_final_particles", False):
            t0_path = args.out / f"particles_{tag}_t0.npz"
            tf_path = args.out / f"particles_{tag}_final.npz"
            np.savez_compressed(
                t0_path,
                pos=parts["pos"],
                vel=parts["vel"],
                mass=parts["mass"],
                eps=parts["eps"],
                component_id=parts["component_id"],
            )
            np.savez_compressed(
                tf_path,
                pos=evo["pos_final"],
                vel=evo["vel_final"],
                mass=evo["mass"],
                eps=evo["eps"],
                component_id=evo["component_id"],
            )
            print(f"  wrote {t0_path.name} + {tf_path.name}", flush=True)

        # Drop heavy arrays before JSON serialization.
        for _k in ("pos_final", "vel_final", "mass", "eps", "component_id"):
            evo.pop(_k, None)

        snaps = evo.pop("snaps")
        snap_t = evo.pop("snap_t_gyr")
        face_png = args.out / f"faceon_components_{tag}.png"
        prof_png = args.out / f"profiles_components_{tag}.png"
        _plot_faceon_grid(
            face_png,
            snaps,
            snap_t,
            arm=labels.get(tag, tag),
            title=(
                rf"{labels.get(tag, tag)}: face-on $\Sigma$ by component "
                rf"($N_{{\rm disk}}={args.n_disk:,}$, $N_{{\rm tot}}={n_ev:,}$)"
            ),
        )
        _plot_profiles(
            prof_png,
            snaps,
            snap_t,
            arm=labels.get(tag, tag),
            title=(
                rf"{labels.get(tag, tag)}: density profiles "
                rf"($t_{{\rm end}}={args.evolve_gyr}\,\mathrm{{Gyr}}$)"
            ),
        )
        maps_npz = args.out / f"component_maps_{tag}.npz"
        _save_maps_npz(maps_npz, snaps, snap_t, tag)

        row = {
            "ok": True,
            "pre": pre,
            "a2_pre": float(evo["a2_t"][0]) if evo.get("a2_t") else pre["a2"],
            "a2_post": float(evo["a2_t"][-1]) if evo.get("a2_t") else None,
            "com_drift_kpc": evo["com_drift_kpc"],
            "n_steps": evo["n_steps"],
            "force_method": evo["force_method"],
            "force_probe_s": evo.get("force_probe_s"),
            "s_per_step": evo.get("s_per_step"),
            "wall_s": evo["wall_s"],
            "t_gyr": evo["t_gyr"],
            "a2_t": evo["a2_t"],
            "com_norm_t": evo["com_norm_t"],
            "snap_t_gyr": snap_t,
            "snapshots": _serialize_snaps(snaps, snap_t),
            "meta": pack["meta"],
            "figures": {
                "faceon": str(face_png),
                "profiles": str(prof_png),
                "maps_npz": str(maps_npz),
            },
        }
        report["arms"][tag] = row
        report["figures"][f"{tag}_faceon"] = str(face_png)
        report["figures"][f"{tag}_profiles"] = str(prof_png)
        kept_snaps[tag] = (snaps, snap_t)
        print(
            f"  A2 {row['a2_pre']:.3f}→{row['a2_post']:.3f}  "
            f"COM={evo['com_drift_kpc']:.4f}  wall={evo['wall_s']:.1f}s  "
            f"force={evo['force_method']}  s/step={evo.get('s_per_step', float('nan')):.2f}  "
            f"snaps={len(snap_t)}",
            flush=True,
        )

    if "data" in kept_snaps and "fft_recon" in kept_snaps:
        snaps_d, times_d = kept_snaps["data"]
        snaps_r, times_r = kept_snaps["fft_recon"]
        # Align on overlapping times (should match when reusing data snap schedule).
        n = min(len(times_d), len(times_r), len(snaps_d), len(snaps_r))
        cmp_png = args.out / "profiles_components_data_vs_fft.png"
        _plot_profiles_compare(
            cmp_png,
            snaps_d[:n],
            snaps_r[:n],
            times_d[:n],
            label_a=labels.get("data", "data"),
            label_b=labels.get("fft_recon", "FFT recon"),
            title=(
                rf"{labels.get('data', 'data')} vs "
                rf"{labels.get('fft_recon', 'FFT recon')}: density profiles "
                rf"($N_{{\rm disk}}={args.n_disk:,}$, "
                rf"$t_{{\rm end}}={args.evolve_gyr}\,\mathrm{{Gyr}}$)"
            ),
        )
        report["figures"]["data_vs_fft_profiles"] = str(cmp_png)
        print(f"=== compare profiles → {cmp_png} ===", flush=True)

    # Also compare data vs any extra / residual generative arm when present.
    for cmp_tag in (
        str(args.extra_arm_name).strip() if args.extra_arm_npz is not None else "",
        "residual_f0",
        "latent_gen",
    ):
        if not cmp_tag or cmp_tag not in kept_snaps or "data" not in kept_snaps:
            continue
        if cmp_tag == "fft_recon":
            continue
        snaps_d, times_d = kept_snaps["data"]
        snaps_g, times_g = kept_snaps[cmp_tag]
        n = min(len(times_d), len(times_g), len(snaps_d), len(snaps_g))
        cmp_png = args.out / f"profiles_components_data_vs_{cmp_tag}.png"
        _plot_profiles_compare(
            cmp_png,
            snaps_d[:n],
            snaps_g[:n],
            times_d[:n],
            label_a=labels.get("data", "data"),
            label_b=labels.get(cmp_tag, cmp_tag),
            title=(
                rf"{labels.get('data', 'data')} vs "
                rf"{labels.get(cmp_tag, cmp_tag)}: density profiles "
                rf"($N_{{\rm disk}}={args.n_disk:,}$, "
                rf"$t_{{\rm end}}={args.evolve_gyr}\,\mathrm{{Gyr}}$)"
            ),
        )
        report["figures"][f"data_vs_{cmp_tag}_profiles"] = str(cmp_png)
        print(f"=== compare profiles → {cmp_png} ===", flush=True)
        break

    a2_png = args.out / "a2_t.png"
    _plot_a2_t(a2_png, report)
    report["figures"]["a2_t"] = str(a2_png)

    out_json = args.out / "verdict.json"
    out_json.write_text(json.dumps(report, indent=2))
    lines = [
        "# Component face-on / density-profile evolve slices",
        "",
        f"Teacher: `{args.teacher}`",
        f"Settings: dt={args.dt}, end_gyr={args.evolve_gyr} → **{report['n_steps']}** steps; "
        f"**disk = {args.n_disk:,}**, total N={n_ev:,} (mix 4:2:1); "
        f"OpenMP={args.omp}; face-on bins={args.faceon_bins}.",
        f"Snapshot times [Gyr]: {faceon_times}",
        f"FOV half-widths [kpc]: disk={COMPONENTS[0][3]}, bulge={COMPONENTS[1][3]}, "
        f"halo={COMPONENTS[2][3]}",
        "",
        "Primary dynamical-consistency comparison: **data vs `fft_recon`** "
        "(FFT-long teacher encode→decode→resample).",
        "",
        "| Arm | A₂ pre→post | COM drift | wall [s] | face-on fig | profiles fig |",
        "|-----|-------------|-----------|----------|-------------|--------------|",
    ]
    for tag, row in report["arms"].items():
        if not row.get("ok"):
            lines.append(f"| `{tag}` | FAIL | — | — | — | — |")
            continue
        a2p = row.get("a2_pre")
        a2q = row.get("a2_post")
        a2_s = (
            f"{a2p:.3f}→{a2q:.3f}"
            if isinstance(a2p, (int, float)) and isinstance(a2q, (int, float))
            else "reused"
        )
        wall = row.get("wall_s")
        wall_s = f"{wall:.1f}" if isinstance(wall, (int, float)) else "—"
        drift = row.get("com_drift_kpc")
        drift_s = f"{drift:.4f}" if isinstance(drift, (int, float)) else "—"
        lines.append(
            f"| `{tag}` | {a2_s} | {drift_s} | {wall_s} | "
            f"`{Path(row['figures']['faceon']).name}` | "
            f"`{Path(row['figures']['profiles']).name}` |"
        )
    if report["figures"].get("data_vs_fft_profiles"):
        lines += [
            "",
            f"Side-by-side profiles: `{Path(report['figures']['data_vs_fft_profiles']).name}`",
        ]
    (args.out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print((args.out / "SUMMARY.md").read_text(), flush=True)

    if args.paper_figures is not None:
        args.paper_figures.mkdir(parents=True, exist_ok=True)
        prefix = str(args.paper_prefix).rstrip("_")
        for tag, row in report["arms"].items():
            if not row.get("ok"):
                continue
            # Prefer fft_recon + data for paper; still copy others if present.
            for key, suffix in (
                ("faceon", f"{prefix}_faceon_{tag}.png"),
                ("profiles", f"{prefix}_profiles_{tag}.png"),
            ):
                src = Path(row["figures"][key])
                if src.is_file():
                    dst = args.paper_figures / suffix
                    shutil.copy2(src, dst)
                    print(f"  paper ← {suffix}", flush=True)
        cmp_src = report["figures"].get("data_vs_fft_profiles")
        if cmp_src and Path(cmp_src).is_file():
            dst = args.paper_figures / f"{prefix}_profiles_data_vs_fft.png"
            shutil.copy2(cmp_src, dst)
            print(f"  paper ← {dst.name}", flush=True)
        a2_src = report["figures"].get("a2_t")
        if a2_src and Path(a2_src).is_file():
            dst = args.paper_figures / f"{prefix}_a2_t.png"
            shutil.copy2(a2_src, dst)
            print(f"  paper ← {dst.name}", flush=True)
        print(f"copied panels → {args.paper_figures}", flush=True)

    print(f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
