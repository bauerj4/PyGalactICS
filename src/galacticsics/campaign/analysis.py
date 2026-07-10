"""Helpers for loading campaign outputs and building density / rotation plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from galacticsics.diagnostics.profiles import nfw_density_profile
from galacticsics.models import GalaxyModel
from galacticsics.sampling.particles import ParticleSet
from ntropy.analysis.density import DensityProfile, bin_spherical_density, compare_density_profiles
from ntropy.analysis.disk_density import (
    bin_midplane_surface_density,
    bin_plane_density,
    compare_surface_profiles,
)
from ntropy.analysis.tiered_diagnostics import diagnostics_dataframe
from ntropy.integrations.galacticsics import merge_galacticsics_components
from ntropy.io.particles import read_particles_ascii
from ntropy.particle_types import TypeRegistry
from ntropy.particles import ParticleState
from ntropy.softening import kinetic_energy, total_energy


def load_ic_state(path: Path) -> ParticleState:
    """Load a campaign ``ic_state.npz``."""
    with np.load(path, allow_pickle=True) as data:
        return ParticleState.from_arrays(
            data["pos"],
            data["vel"],
            data["mass"],
            data["eps"],
            type_id=data["type_id"] if "type_id" in data else None,
            timestep_bin=data["timestep_bin"] if "timestep_bin" in data else None,
            tags=data["tags"] if "tags" in data else None,
        )


def default_registry() -> TypeRegistry:
    return TypeRegistry.default_galaxy()


def load_merged_state(model_dir: Path, *, registry: TypeRegistry | None = None) -> ParticleState:
    """Load merged IC state (prefers ``ic_state.npz``, else legacy component files)."""
    ic_path = model_dir / "ic_state.npz"
    if ic_path.is_file():
        return load_ic_state(ic_path)
    registry = registry or default_registry()
    from galacticsics.campaign.runner import _load_ic_particles

    particles = _load_ic_particles(model_dir)
    return merge_galacticsics_components(particles, type_registry=registry)


def load_evolved_state(model_dir: Path, template: ParticleState) -> ParticleState | None:
    final_path = model_dir / "evolution" / "final.dat"
    if not final_path.exists():
        return None
    data = read_particles_ascii(final_path)
    pos = np.column_stack([data["x"], data["y"], data["z"]])
    vel = np.column_stack([data["vx"], data["vy"], data["vz"]])
    return ParticleState.from_arrays(
        pos, vel, data["mass"], template.eps,
        type_id=template.type_id,
        tags=template.tags,
    )


def load_model_json(model_dir: Path) -> GalaxyModel:
    from galacticsics.campaign.serialize import model_from_dict

    return model_from_dict(json.loads((model_dir / "model.json").read_text()))


def component_mask(state: ParticleState, name: str) -> np.ndarray:
    return state.tags == name


# Minimum shell occupancy for ρ(r) overlays (log shells are noisy below this).
HALO_PROFILE_MIN_COUNT = 50


def halo_profile_bin_edges(
    *,
    n_bins: int = 24,
    r_min: float = 1.0,
    r_max: float = 40.0,
    r_log_max: float = 10.0,
    n_log: int | None = None,
) -> np.ndarray:
    """
    Hybrid log (inner) + linear (outer) spherical shell edges.

    Log bins resolve the cusp; linear outer bins keep roughly uniform shell
    volume and particle counts after evolution.  Shared edges make t=0 and
    t>0 profiles directly comparable.
    """
    if n_log is None:
        n_log = max(n_bins // 2, 1)
    n_lin = max(n_bins - n_log, 1)
    r_lo = min(r_min, r_max * 0.99)
    r_transition = min(max(r_log_max, r_lo * 1.01), r_max * 0.99)
    log_edges = np.logspace(np.log10(r_lo), np.log10(r_transition), n_log + 1)
    lin_edges = np.linspace(r_transition, r_max, n_lin + 1)[1:]
    return np.concatenate([log_edges, lin_edges])


def _bin_spherical_density_edges(
    pos: np.ndarray,
    mass: np.ndarray,
    edges: np.ndarray,
) -> DensityProfile:
    """Bin particle mass into fixed spherical shells."""
    r = np.sqrt(np.sum(pos * pos, axis=1))
    n_bins = len(edges) - 1
    shell_mass = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        shell_mask = (r >= edges[i]) & (r < edges[i + 1])
        shell_mass[i] = mass[shell_mask].sum()
        counts[i] = int(shell_mask.sum())
    volumes = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    volumes = np.maximum(volumes, 1e-30)
    rho = shell_mass / volumes
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    return DensityProfile(r_mid=r_mid, rho=rho, counts=counts)


def halo_spherical_profile(
    state: ParticleState,
    *,
    n_bins: int = 24,
    r_max: float = 40.0,
    log_bins: bool = True,
    r_min: float = 1.0,
    bin_edges: np.ndarray | None = None,
    r_log_max: float = 10.0,
):
    """Spherical halo density profile with a stable inner radius.

    MW-style halos sampled to ``r_outer`` ~ 200 kpc rarely place particles
    inside ~1 kpc; log-spaced shells below that are empty or single-particle
    spikes that dominate misleading ρ(r) overlays.

    By default uses hybrid log+linear shell edges (:func:`halo_profile_bin_edges`).
    Pass explicit ``bin_edges`` so initial and evolved snapshots share identical
    binning.  Set ``log_bins=False`` to recover legacy pure-linear shells.
    """
    mask = component_mask(state, "halo")
    pos, mass = state.pos[mask], state.mass[mask]
    if bin_edges is not None:
        return _bin_spherical_density_edges(pos, mass, bin_edges)
    if log_bins:
        edges = halo_profile_bin_edges(
            n_bins=n_bins, r_min=r_min, r_max=r_max, r_log_max=r_log_max
        )
        return _bin_spherical_density_edges(pos, mass, edges)
    return bin_spherical_density(
        pos, mass, n_bins=n_bins, r_max=r_max, log_bins=False, r_min=r_min
    )


def profile_plot_series(
    profile: DensityProfile,
    *,
    min_count: int = HALO_PROFILE_MIN_COUNT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Radii and ρ for plotting; invalid shells are NaN so lines do not bridge gaps.
    """
    valid = profile.counts >= min_count
    r_mid = np.where(valid, profile.r_mid, np.nan)
    rho = np.where(valid, profile.rho, np.nan)
    return r_mid, rho, valid


def plot_halo_spherical_profile(
    ax,
    initial: DensityProfile,
    final: DensityProfile,
    *,
    model: GalaxyModel | None = None,
    end_time_gyr: float | None = None,
    min_count: int = HALO_PROFILE_MIN_COUNT,
    show_counts: bool = True,
    show_uncertainty: bool = True,
) -> Any:
    """
    Halo ρ(r) with NFW overlay, optional count axis, and √N uncertainty bands.

    Shells below ``min_count`` are masked (NaN) so evolved snapshots with sparse
    log bins do not produce jagged connected segments.
    """
    r_i, rho_i, valid_i = profile_plot_series(initial, min_count=min_count)
    r_f, rho_f, valid_f = profile_plot_series(final, min_count=min_count)
    end_label = f"t={end_time_gyr} Gyr" if end_time_gyr is not None else "final"

    line_i = ax.loglog(r_i, rho_i, "C0-", label="t=0", ms=3)[0]
    line_f = ax.loglog(r_f, rho_f, "C1--", label=end_label, ms=3)[0]

    if show_uncertainty:
        for prof, valid, color in (
            (initial, valid_i, line_i.get_color()),
            (final, valid_f, line_f.get_color()),
        ):
            idx = np.where(valid)[0]
            if idx.size == 0:
                continue
            rel = 1.0 / np.sqrt(np.maximum(prof.counts[idx], 1))
            ax.fill_between(
                prof.r_mid[idx],
                prof.rho[idx] / (1.0 + rel),
                prof.rho[idx] * (1.0 + rel),
                color=color,
                alpha=0.12,
                linewidth=0,
            )

    overlay_r = initial.r_mid[valid_i]
    if overlay_r.size == 0:
        overlay_r = initial.r_mid[initial.counts >= max(min_count // 5, 1)]
    if model is not None and overlay_r.size > 0:
        rho_ana = analytic_halo_overlay(model, overlay_r)
        if rho_ana is not None:
            ax.loglog(overlay_r, rho_ana, "k:", alpha=0.45, lw=1, label="NFW model")

    ax.set_xlabel("r [kpc]")
    ax.set_ylabel("ρ [M_unit / kpc³]")
    ax.set_title("Halo ρ(r)")
    ax.legend(fontsize=8, loc="upper right")

    if show_counts:
        ax2 = ax.twinx()
        count_r = np.where(valid_f, final.r_mid, np.nan)
        count_n = np.where(valid_f, final.counts, np.nan)
        ax2.plot(count_r, count_n, color="0.55", lw=0.8, alpha=0.55, label="counts (final)")
        ax2.set_ylabel("particles / shell", color="0.45", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="0.45", labelsize=7)
        ax2.set_yscale("log")
    return ax


def disk_surface_profile(
    state: ParticleState,
    *,
    n_bins: int = 20,
    r_max: float = 22.0,
):
    mask = component_mask(state, "disk")
    return bin_midplane_surface_density(state.pos[mask], state.mass[mask], n_bins=n_bins, r_max=r_max)


def disk_projections(
    state: ParticleState,
    *,
    half_extent: float = 20.0,
    n_bins: int = 96,
    z_slice: float | None = 0.3,
):
    mask = component_mask(state, "disk")
    pos, mass = state.pos[mask], state.mass[mask]
    face_on = bin_plane_density(pos, mass, axes=(0, 1), n_bins=n_bins, half_extent=half_extent)
    z_filter = np.abs(pos[:, 2]) < z_slice if z_slice is not None else None
    edge_on = bin_plane_density(
        pos, mass, axes=(0, 2), n_bins=n_bins, half_extent=half_extent, z_filter=z_filter
    )
    return face_on, edge_on


def halo_projections(
    state: ParticleState,
    *,
    half_extent: float = 40.0,
    n_bins: int = 96,
    z_slice: float | None = 0.3,
):
    """Face-on (x–y) and edge-on (x–z) volumetric-density maps for halo particles."""
    mask = component_mask(state, "halo")
    pos, mass = state.pos[mask], state.mass[mask]
    face_on = bin_plane_density(pos, mass, axes=(0, 1), n_bins=n_bins, half_extent=half_extent)
    z_filter = np.abs(pos[:, 2]) < z_slice if z_slice is not None else None
    edge_on = bin_plane_density(
        pos, mass, axes=(0, 2), n_bins=n_bins, half_extent=half_extent, z_filter=z_filter
    )
    return face_on, edge_on


def load_run_diagnostics(work_dir: Path):
    evo_dir = work_dir / "evolution"
    return diagnostics_dataframe(evo_dir), evo_dir


def load_rotation_curve(work_dir: Path, label: str = "ic") -> dict:
    path = work_dir / f"rotation_curve_{label}.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def model_summary_table(model: GalaxyModel) -> pd.DataFrame:
    """Key physical parameters (see :func:`~galacticsics.campaign.spec.model_component_parameter_table` for all fields)."""
    from galacticsics.campaign.spec import model_component_parameter_table

    table = model_component_parameter_table(model)
    key_params = {
        "halo.v0",
        "halo.a",
        "halo.r_outer",
        "disk.mass",
        "disk.scale_length",
        "disk.scale_height",
        "disk_kinematics.sigma_r0",
        "disk_kinematics.toomre_q_target",
    }
    subset = table[table["parameter"].isin(key_params)]
    grid_rows = pd.DataFrame(
        [
            {"parameter": "grid.nr", "value": model.grid.nr, "enabled": True},
            {"parameter": "grid.lmax", "value": model.grid.lmax, "enabled": True},
            {"parameter": "grid.dr", "value": model.grid.dr, "enabled": True},
        ]
    )
    return pd.concat([subset, grid_rows], ignore_index=True)[["parameter", "value"]]


def density_drift_metrics(initial: ParticleState, final: ParticleState) -> dict[str, float]:
    edges = halo_profile_bin_edges()
    halo_i = halo_spherical_profile(initial, bin_edges=edges)
    halo_f = halo_spherical_profile(final, bin_edges=edges)
    disk_i = disk_surface_profile(initial)
    disk_f = disk_surface_profile(final)
    return {
        "halo_rho_drift": compare_density_profiles(halo_i, halo_f, min_count=HALO_PROFILE_MIN_COUNT),
        "disk_sigma_drift": compare_surface_profiles(disk_i, disk_f, min_count=20),
    }


def analytic_halo_overlay(model: GalaxyModel, r_mid: np.ndarray) -> np.ndarray | None:
    if model.halo is None or not model.halo.enabled:
        return None
    return nfw_density_profile(r_mid, model.halo)


def density_map_log10(density_map, *, log_floor: float = 1e-30) -> np.ndarray:
    """log10 surface density for imshow (shape matches ``density_map.density.T``)."""
    return np.log10(np.maximum(density_map.density.T, log_floor))


def _occupied_map_mask(density_map, *, log_floor: float = 1e-30) -> np.ndarray:
    counts = getattr(density_map, "counts", None)
    if counts is not None:
        return counts.T > 0
    data = density_map_log10(density_map, log_floor=log_floor)
    return np.isfinite(data) & (data > np.log10(log_floor) + 0.5)


def density_map_color_limits(
    *density_maps,
    log_floor: float = 1e-30,
    vmin_pct: float = 2.0,
    vmax_pct: float = 99.5,
) -> tuple[float, float]:
    """
    Shared log10 Σ color limits from occupied bins across one or more maps.

    Empty bins are excluded so sparse N-body projections are not washed out
    by the log floor when comparing snapshots side by side.
    """
    samples: list[np.ndarray] = []
    for density_map in density_maps:
        data = density_map_log10(density_map, log_floor=log_floor)
        occupied = _occupied_map_mask(density_map, log_floor=log_floor)
        if occupied.any():
            samples.append(data[occupied])
    if not samples:
        return -5.0, 0.0
    stacked = np.concatenate(samples)
    vmin = float(np.percentile(stacked, vmin_pct))
    vmax = float(np.percentile(stacked, vmax_pct))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def plot_density_map(
    ax,
    density_map,
    *,
    title: str,
    cmap: str = "inferno",
    log_floor: float = 1e-30,
    vmin_pct: float = 2.0,
    vmax_pct: float = 99.5,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Any:
    """
    Face-on / edge-on surface-density map with percentile-based contrast.

    Uses the 2nd–99.5th percentile of log density in **occupied** bins. Pass
    explicit ``vmin`` / ``vmax`` (or :func:`density_map_color_limits`) to keep
    the same color scale across time slices.
    """
    extent = [
        density_map.x_edges[0],
        density_map.x_edges[-1],
        density_map.y_edges[0],
        density_map.y_edges[-1],
    ]
    data = density_map_log10(density_map, log_floor=log_floor)
    if vmin is None or vmax is None:
        auto_vmin, auto_vmax = density_map_color_limits(
            density_map,
            log_floor=log_floor,
            vmin_pct=vmin_pct,
            vmax_pct=vmax_pct,
        )
        if vmin is None:
            vmin = auto_vmin
        if vmax is None:
            vmax = auto_vmax
    im = ax.imshow(
        data,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title)
    return im


def add_density_colorbar(
    fig,
    mappable,
    axes,
    *,
    label: str = "log₁₀ density",
    shrink: float = 0.85,
    pad: float = 0.02,
) -> Any:
    """Attach a single colorbar without stealing a vertical strip from image axes."""
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    cbar = fig.colorbar(
        mappable,
        ax=list(axes),
        location="right",
        shrink=shrink,
        pad=pad,
    )
    cbar.set_label(label)
    return cbar
