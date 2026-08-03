"""Helpers for loading campaign outputs and building density / rotation plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from galacticsics.diagnostics.df_validation import (
    validate_bulge_df,
    validate_disk_df,
    validate_halo_df,
    validate_ic_distribution_functions,
)
from galacticsics.diagnostics.profiles import nfw_density_profile
from galacticsics.models import GalaxyModel
from galacticsics.sampling.particles import ParticleSet
from ntropy.analysis.density import DensityProfile, bin_spherical_density, compare_density_profiles
from ntropy.analysis.disk_density import (
    bin_midplane_surface_density,
    bin_plane_density,
    compare_surface_profiles,
    disk_azimuthal_fourier,
)
from ntropy.analysis.tiered_diagnostics import diagnostics_dataframe
from ntropy.integrations.galacticsics import merge_galacticsics_components
from ntropy.io.particles import read_particles_ascii
from ntropy.particle_types import TypeRegistry
from ntropy.particles import ParticleState
from ntropy.softening import kinetic_energy, total_energy, virial_diagnostic


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


def _load_particle_dump(path: Path, template: ParticleState) -> ParticleState:
    """Reconstruct :class:`ParticleState` from an ntropy tiered ``step_*.npz`` dump."""
    with np.load(path, allow_pickle=True) as data:
        kwargs: dict[str, Any] = {
            "pos": data["pos"],
            "vel": data["vel"],
            "mass": data["mass"],
            "eps": data["eps"],
        }
        if "type_id" in data:
            kwargs["type_id"] = data["type_id"]
        if "timestep_bin" in data:
            kwargs["timestep_bin"] = data["timestep_bin"]
        state = ParticleState.from_arrays(**kwargs)
    if template.tags is not None:
        if state.n != template.n:
            raise ValueError(
                f"particle dump N={state.n} != template N={template.n} ({path.name})"
            )
        state.tags = template.tags.copy()
    return state


def evolution_checkpoint_times(
    checkpoint_gyr: list[float],
    *,
    end_time_gyr: float,
    tolerance_gyr: float = 1e-6,
) -> list[float]:
    """
    Sorted unique checkpoint times, always including t=0 and the run end.

    Intermediate entries come from ``checkpoint_gyr`` (e.g. CONFIG
    ``checkpoints_gyr``).  The final time is appended when absent so notebooks
    can plot t=0, mid-evolution dumps, and ``final.dat`` in one grid.
    """
    times = sorted({float(t) for t in checkpoint_gyr})
    if not times or times[0] > tolerance_gyr:
        times = [0.0, *times]
    end = float(end_time_gyr)
    if not any(abs(t - end) <= tolerance_gyr for t in times):
        times.append(end)
    return sorted(times)


def load_evolution_checkpoints(
    model_dir: Path,
    checkpoint_gyr: list[float],
    *,
    template: ParticleState | None = None,
    end_time_gyr: float | None = None,
    tolerance_gyr: float = 0.02,
) -> list[tuple[float, ParticleState]]:
    """
    Load particle states at campaign checkpoint times.

    Uses ``ic_state.npz`` (or merged ICs) at t=0, ``evolution/final.dat`` at
    the run end, and ``evolution/particles/step_*.npz`` at intermediate times
    (matched via ``diagnostics.csv`` ``step`` / ``t_gyr``).  Times with no dump
    within ``tolerance_gyr`` are skipped (warnings via the ``warnings`` module).
    """
    import warnings

    model_dir = Path(model_dir)
    evo_dir = model_dir / "evolution"
    template = template or load_merged_state(model_dir)

    if end_time_gyr is None:
        end_time_gyr = float(checkpoint_gyr[-1]) if checkpoint_gyr else 0.0
        diag_csv = evo_dir / "diagnostics.csv"
        if diag_csv.is_file():
            df_end = pd.read_csv(diag_csv)
            if len(df_end) and "t_gyr" in df_end.columns:
                end_time_gyr = float(df_end["t_gyr"].iloc[-1])

    times = evolution_checkpoint_times(checkpoint_gyr, end_time_gyr=float(end_time_gyr))
    diag_df = None
    diag_csv = evo_dir / "diagnostics.csv"
    if diag_csv.is_file():
        diag_df = pd.read_csv(diag_csv)

    particles_dir = evo_dir / "particles"
    out: list[tuple[float, ParticleState]] = []

    for t_target in times:
        if abs(t_target) <= tolerance_gyr:
            out.append((0.0, template.copy()))
            continue

        if abs(t_target - float(end_time_gyr)) <= tolerance_gyr:
            final = load_evolved_state(model_dir, template)
            if final is None:
                warnings.warn(f"No evolution/final.dat for t={t_target:.3f} Gyr")
                continue
            out.append((float(end_time_gyr), final))
            continue

        if diag_df is None or "step" not in diag_df.columns or "t_gyr" not in diag_df.columns:
            warnings.warn(f"No diagnostics.csv — cannot resolve t={t_target:.3f} Gyr")
            continue

        idx = int((diag_df["t_gyr"] - t_target).abs().idxmin())
        t_actual = float(diag_df.loc[idx, "t_gyr"])
        if abs(t_actual - t_target) > tolerance_gyr:
            warnings.warn(
                f"No evolution dump within {tolerance_gyr} Gyr of t={t_target:.3f} "
                f"(nearest diagnostics t={t_actual:.3f})"
            )
            continue
        step = int(diag_df.loc[idx, "step"])
        dump_path = particles_dir / f"step_{step:06d}.npz"
        if not dump_path.is_file():
            warnings.warn(f"Missing {dump_path.relative_to(model_dir)} for t={t_actual:.3f} Gyr")
            continue
        out.append((t_actual, _load_particle_dump(dump_path, template)))

    return out


def load_model_json(model_dir: Path) -> GalaxyModel:
    from galacticsics.campaign.serialize import model_from_dict

    return model_from_dict(json.loads((model_dir / "model.json").read_text()))


def component_mask(state: ParticleState, name: str) -> np.ndarray:
    return state.tags == name


# Minimum shell occupancy for ρ(r) drift metrics (log shells are noisy below this).
HALO_PROFILE_MIN_COUNT = 50
# Looser threshold for evolved ρ(r) overlays — inner shells empty after heating.
HALO_PROFILE_PLOT_MIN_COUNT = 25


def halo_profile_min_count(
    n_halo: int,
    *,
    n_bins: int = 24,
    purpose: str = "plot",
) -> int:
    """
    Occupancy threshold for halo ρ(r) shells.

    ``purpose='drift'`` keeps a fixed high bar for quantitative comparison;
    ``purpose='plot'`` scales down for evolved snapshots whose inner shells
    lose particles to outward diffusion.
    """
    if purpose == "drift":
        return HALO_PROFILE_MIN_COUNT
    target = max(HALO_PROFILE_PLOT_MIN_COUNT, n_halo // max(5 * n_bins, 1))
    return min(HALO_PROFILE_MIN_COUNT, target)


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
    min_count: int | None = None,
    min_count_final: int | None = None,
    show_counts: bool = True,
    show_uncertainty: bool = True,
) -> Any:
    """
    Halo ρ(r) with NFW overlay, optional count axis, and √N uncertainty bands.

    Shells below ``min_count`` are masked (NaN) so evolved snapshots with sparse
    log bins do not produce jagged connected segments.

    Pass ``min_count_final`` when the evolved halo has depleted inner shells
    (common after heating); it defaults to :data:`HALO_PROFILE_PLOT_MIN_COUNT`.
    """
    if min_count is None:
        min_count = HALO_PROFILE_MIN_COUNT
    if min_count_final is None:
        min_count_final = HALO_PROFILE_PLOT_MIN_COUNT
    r_i, rho_i, valid_i = profile_plot_series(initial, min_count=min_count)
    r_f, rho_f, valid_f = profile_plot_series(final, min_count=min_count_final)
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
    n_bins: int = 128,
    z_slice: float | None = 0.3,
):
    return component_projections(
        state,
        "disk",
        half_extent=half_extent,
        n_bins=n_bins,
        z_slice=z_slice,
    )


def halo_projections(
    state: ParticleState,
    *,
    half_extent: float = 40.0,
    n_bins: int = 96,
    z_slice: float | None = None,
):
    """Face-on (x–y) and side-on (x–z) surface-density maps for halo particles.

    Side-on uses the full line of sight by default (no midplane cut) so the
    halo stays roughly round rather than looking disk-like.
    """
    return component_projections(
        state,
        "halo",
        half_extent=half_extent,
        n_bins=n_bins,
        z_slice=z_slice,
    )


_COMPONENT_PROJ_DEFAULTS: dict[str, dict[str, float | int | None]] = {
    "disk": {"half_extent": 20.0, "n_bins": 128, "z_slice": 0.3},
    # Full column for halo/bulge side-on — a thin |z| cut makes them look disk-like.
    "halo": {"half_extent": 40.0, "n_bins": 96, "z_slice": None},
    "bulge": {"half_extent": 5.0, "n_bins": 96, "z_slice": None},
}


def component_projections(
    state: ParticleState,
    component: str,
    *,
    half_extent: float | None = None,
    n_bins: int | None = None,
    z_slice: float | None | str = "default",
):
    """
    Face-on (x–y) and side-on (x–z) projected surface-density maps for one component.

    Parameters
    ----------
    state : ParticleState
        Particle snapshot (uses ``tags`` or ``type_id`` via :func:`component_mask`).
    component : str
        ``disk``, ``halo``, or ``bulge``.
    half_extent, n_bins, z_slice
        Override per-component defaults.  Pass ``z_slice=None`` to project all
        particles in the side-on map (no midplane cut).  Leave ``z_slice`` as
        ``"default"`` to use the component-specific midplane cut.
    """
    defaults = _COMPONENT_PROJ_DEFAULTS.get(
        component, {"half_extent": 20.0, "n_bins": 96, "z_slice": 0.3}
    )
    if half_extent is None:
        half_extent = float(defaults["half_extent"])  # type: ignore[arg-type]
    if n_bins is None:
        n_bins = int(defaults["n_bins"])  # type: ignore[arg-type]
    if z_slice == "default":
        z_slice = defaults["z_slice"]  # type: ignore[assignment]
    z_cut: float | None = None if z_slice is None else float(z_slice)

    mask = component_mask(state, component)
    if not np.any(mask):
        raise ValueError(f"no particles tagged '{component}' in state (N={state.n})")
    pos, mass = state.pos[mask], state.mass[mask]
    face_on = bin_plane_density(
        pos, mass, axes=(0, 1), n_bins=n_bins, half_extent=half_extent
    )
    z_filter = np.abs(pos[:, 2]) < z_cut if z_cut is not None else None
    side_on = bin_plane_density(
        pos, mass, axes=(0, 2), n_bins=n_bins, half_extent=half_extent, z_filter=z_filter
    )
    return face_on, side_on


def present_components(state: ParticleState) -> list[str]:
    """Ordered component labels present in ``state`` (disk, bulge, halo)."""
    order = ("disk", "bulge", "halo")
    return [name for name in order if np.any(component_mask(state, name))]


def write_component_projection_pngs(
    state: ParticleState,
    out_dir: Path | str,
    *,
    components: list[str] | None = None,
    label: str = "",
    dpi: int = 120,
    cmap: str = "inferno",
) -> list[Path]:
    """
    Write face-on and side-on PNGs for each component into ``out_dir``.

    Filenames: ``{component}_face_on.png`` and ``{component}_side_on.png``
    (optional ``_{label}`` suffix before the extension when ``label`` is set).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comps = components or present_components(state)
    suffix = f"_{label}" if label else ""
    written: list[Path] = []

    for component in comps:
        face, side = component_projections(state, component)
        for view, dens_map, axes_label in (
            ("face_on", face, "x–y"),
            ("side_on", side, "x–z"),
        ):
            fig, ax = plt.subplots(figsize=(5.0, 4.5))
            im = plot_density_map(
                ax,
                dens_map,
                title=f"{component} {view.replace('_', '-')} ({axes_label})",
                cmap=cmap,
            )
            ax.set_xlabel("x [kpc]")
            ax.set_ylabel("y [kpc]" if view == "face_on" else "z [kpc]")
            add_density_colorbar(
                fig, im, ax, label=projection_colorbar_label(component)
            )
            path = out_dir / f"{component}_{view}{suffix}.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            written.append(path)
    return written


def write_campaign_projection_pngs(
    work_root: Path | str,
    *,
    snapshots: list[str] | None = None,
    components: list[str] | None = None,
    model_dirs: list[Path] | None = None,
    dpi: int = 120,
    step_stride: int = 1,
) -> list[Path]:
    """
    Write component projection PNGs for every model under a campaign work root.

    Parameters
    ----------
    work_root : path
        Campaign directory containing per-model subfolders.
    snapshots : list of str
        Which states to plot.  ``ic`` uses ``ic_state.npz``; ``final`` uses
        ``evolution/final.dat`` when present; ``latest`` uses the newest
        ``evolution/particles/step_*.npz``; ``steps`` writes every dump
        (optionally thinned by ``step_stride``) under
        ``projections/step_XXXXXX/``.  Default: ``ic``, ``steps`` when dumps
        exist, and ``final`` when available.
    components : list of str, optional
        Restrict to these components (default: all present in the state).
    model_dirs : list of path, optional
        Explicit model directories (default: every child with ``ic_state.npz``).
    step_stride : int
        When plotting ``steps``, keep every Nth dump (``1`` = all).
    """
    work_root = Path(work_root)
    if model_dirs is None:
        model_dirs = sorted(
            p for p in work_root.iterdir() if p.is_dir() and (p / "ic_state.npz").is_file()
        )
    if not model_dirs:
        raise FileNotFoundError(f"no model dirs with ic_state.npz under {work_root}")
    stride = max(1, int(step_stride))

    written: list[Path] = []
    for model_dir in model_dirs:
        ic = load_merged_state(model_dir)
        particles_dir = model_dir / "evolution" / "particles"
        dumps = (
            sorted(particles_dir.glob("step_*.npz")) if particles_dir.is_dir() else []
        )

        if snapshots is not None:
            wanted = list(snapshots)
        else:
            wanted = ["ic"]
            if dumps:
                wanted.append("steps")
            if load_evolved_state(model_dir, ic) is not None:
                wanted.append("final")

        # (subdir_name, state) pairs — steps expand to many entries
        to_write: list[tuple[str, ParticleState]] = []
        for snap in wanted:
            if snap == "ic":
                to_write.append(("ic", ic))
            elif snap == "final":
                final = load_evolved_state(model_dir, ic)
                if final is not None:
                    to_write.append(("final", final))
            elif snap == "latest":
                if dumps:
                    to_write.append(("latest", _load_particle_dump(dumps[-1], ic)))
            elif snap == "steps":
                selected = dumps[::stride]
                # Always include the last dump if stride skipped it.
                if dumps and dumps[-1] not in selected:
                    selected = [*selected, dumps[-1]]
                for dump_path in selected:
                    to_write.append(
                        (dump_path.stem, _load_particle_dump(dump_path, ic))
                    )
            else:
                raise ValueError(
                    f"unknown snapshot {snap!r}; use ic|final|latest|steps"
                )

        for snap_name, state in to_write:
            out_dir = model_dir / "projections" / snap_name
            written.extend(
                write_component_projection_pngs(
                    state,
                    out_dir,
                    components=components,
                    dpi=dpi,
                )
            )
    return written


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


def component_radial_stats(
    state: ParticleState,
    component: str,
    *,
    radii_kpc: tuple[float, ...] = (20.0, 40.0, 80.0, 120.0, 160.0, 200.0),
) -> dict[str, float]:
    """Per-component spherical radius summary and mass fractions beyond thresholds."""
    mask = component_mask(state, component)
    if not np.any(mask):
        return {"n": 0.0}
    pos = state.pos[mask]
    mass = state.mass[mask]
    r = np.sqrt(np.sum(pos * pos, axis=1))
    total_mass = float(mass.sum())
    out: dict[str, float] = {
        "n": float(mask.sum()),
        "r_min": float(r.min()),
        "r_median": float(np.median(r)),
        "r_max": float(r.max()),
    }
    for thr in radii_kpc:
        beyond = r > thr
        out[f"mass_frac_r_gt_{int(thr)}"] = float(mass[beyond].sum() / total_mass)
    return out


def conserved_quantity_drift(initial: ParticleState, final: ParticleState) -> dict[str, float]:
    """
    Relative drift of energy and bulk momentum between two snapshots.

    Energy uses :func:`~ntropy.softening.total_energy` (kinetic-only proxy when
    ``N > LARGE_N_ENERGY_THRESHOLD``).
    """
    e0 = total_energy(initial.pos, initial.vel, initial.mass, initial.eps)
    ef = total_energy(final.pos, final.vel, final.mass, final.eps)
    p0 = (initial.mass[:, None] * initial.vel).sum(axis=0)
    pf = (final.mass[:, None] * final.vel).sum(axis=0)
    p0_norm = float(np.linalg.norm(p0))
    dp_norm = float(np.linalg.norm(pf - p0))
    out = {
        "dE_over_E0": abs(ef - e0) / max(abs(e0), 1e-30),
        "dP_abs": dp_norm,
    }
    if p0_norm > 1e-6:
        out["dP_over_P0"] = dp_norm / p0_norm
    for label in ("disk", "halo"):
        m0 = component_mask(initial, label)
        mf = component_mask(final, label)
        if np.any(m0) and np.any(mf):
            ri = np.sqrt(np.sum(initial.pos[m0] ** 2, axis=1))
            rf = np.sqrt(np.sum(final.pos[mf] ** 2, axis=1))
            out[f"{label}_r_median_shift_kpc"] = float(np.median(rf) - np.median(ri))
    return out


def state_virial_summary(
    state: ParticleState,
    *,
    rtol: float = 0.3,
    max_particles: int | None = None,
    prefix: str = "",
) -> dict[str, float | bool | int]:
    """Virial-theorem diagnostic for one snapshot (see :func:`virial_diagnostic`)."""
    raw = virial_diagnostic(
        state.pos, state.vel, state.mass, state.eps,
        rtol=rtol, max_particles=max_particles,
    )
    if not prefix:
        return raw
    return {f"{prefix}{k}": v for k, v in raw.items()}


def disk_axisymmetry_diagnostic(
    state: ParticleState,
    *,
    m: int = 2,
    n_bins: int = 15,
    r_max: float = 22.0,
    z_max: float = 0.3,
    min_count: int = 20,
) -> dict[str, Any]:
    """
    Quantify disk departure from axisymmetry via azimuthal Fourier modes.

    At ``t=0``, GalactICS ICs are axisymmetric and ``A2/A0`` is shot-noise
    level (``∝ 1/√N`` per ring).  Growth over multi-Gyr evolution can signal
    bars or spirals from swing amplification (Bauer & Widrow 2018).
    """
    mask = component_mask(state, "disk")
    if not np.any(mask):
        return {"m": m, "a_m_over_a0_median": float("nan"), "n_disk": 0}
    fourier = disk_azimuthal_fourier(
        state.pos[mask], state.mass[mask],
        m=m, n_bins=n_bins, r_max=r_max, z_max=z_max, min_count=min_count,
    )
    return {
        "m": m,
        "n_disk": int(mask.sum()),
        "a_m_over_a0_median": fourier["a_m_over_a0_median"],
        "r_mid": fourier["r_mid"],
        "a_m_over_a0": fourier["a_m_over_a0"],
        "counts": fourier["counts"],
    }


def summarize_evolution_health(
    initial: ParticleState,
    final: ParticleState,
    *,
    model: GalaxyModel | None = None,
    work_dir: Path | None = None,
    diagnostics_df: pd.DataFrame | None = None,
    validate_df: bool = True,
) -> dict[str, Any]:
    """
    One-shot evolution sanity check for campaign walkthrough notebooks.

    Returns energy/momentum drift, density-profile drift, per-component radial
    stats, virial-theorem diagnostics on the IC snapshot, optional DF validation
    (when ``work_dir`` and ``model`` are set), and (when provided) the terminal
    row from ``diagnostics.csv``.

    Notes
    -----
    Pairwise ``2T/|W|`` is only a rough self-gravity check.  GalactICS ICs are
    equilibria in the fixed ``dbh`` potential, so expect ``~0.7–1.2`` rather than
    exact unity; values ``≫ 10`` usually mean a broken diagnostic (e.g. mismatched
    kinetic/potential particle sets), not a physical IC.
    """
    edges = halo_profile_bin_edges()
    halo_i = halo_spherical_profile(initial, bin_edges=edges)
    halo_f = halo_spherical_profile(final, bin_edges=edges)
    drift = density_drift_metrics(initial, final)
    disk_ax_i = disk_axisymmetry_diagnostic(initial)
    disk_ax_f = disk_axisymmetry_diagnostic(final)
    summary: dict[str, Any] = {
        **conserved_quantity_drift(initial, final),
        **drift,
        **state_virial_summary(initial, prefix="virial_ic_"),
        **state_virial_summary(final, prefix="virial_final_"),
        "disk_axisymmetry_ic": disk_ax_i,
        "disk_axisymmetry_final": disk_ax_f,
        "disk_m2_a0_median_ic": disk_ax_i["a_m_over_a0_median"],
        "disk_m2_a0_median_final": disk_ax_f["a_m_over_a0_median"],
        "halo_ic": component_radial_stats(initial, "halo"),
        "halo_final": component_radial_stats(final, "halo"),
        "disk_ic": component_radial_stats(initial, "disk", radii_kpc=(5.0, 10.0, 15.0, 20.0)),
        "disk_final": component_radial_stats(final, "disk", radii_kpc=(5.0, 10.0, 15.0, 20.0)),
    }
    if model is not None:
        valid = halo_i.counts >= HALO_PROFILE_MIN_COUNT
        rho_ana = analytic_halo_overlay(model, halo_i.r_mid[valid])
        if rho_ana is not None and valid.any():
            summary["halo_ic_nfw_median_ratio"] = float(
                np.median(halo_i.rho[valid] / np.maximum(rho_ana, 1e-30))
            )
        valid_f = halo_f.counts >= HALO_PROFILE_PLOT_MIN_COUNT
        rho_ana_f = analytic_halo_overlay(model, halo_f.r_mid[valid_f])
        if rho_ana_f is not None and valid_f.any():
            summary["halo_final_nfw_median_ratio"] = float(
                np.median(halo_f.rho[valid_f] / np.maximum(rho_ana_f, 1e-30))
            )
    if diagnostics_df is not None and len(diagnostics_df):
        last = diagnostics_df.iloc[-1]
        for col in ("dE_over_E0", "active_fraction", "disk_mean_bin", "halo_mean_bin"):
            if col in last:
                summary[f"diag_{col}"] = float(last[col])

    if validate_df and work_dir is not None and model is not None:
        df_report = validate_ic_distribution_functions(
            initial, model, work_dir, include_virial=False
        )
        summary["df_validation"] = df_report
        summary["df_validation_pass"] = bool(df_report.get("overall_pass", False))
        for comp in ("halo", "disk", "bulge"):
            comp_report = df_report.get(comp, {})
            if not isinstance(comp_report, dict):
                continue
            prefix = f"df_{comp}_"
            if "pass" in comp_report:
                summary[f"{prefix}pass"] = comp_report["pass"]
            for key in (
                "energy_df_max_log_rel",
                "speed_in_bounds_frac",
                "surface_density_max_rel",
                "sigma_r0_ratio",
                "a2_over_a0_median",
                "cordbh_f_d_min",
                "cordbh_f_sz_min",
                "df_positive_frac",
            ):
                if key in comp_report and isinstance(comp_report[key], (int, float)):
                    summary[f"{prefix}{key}"] = float(comp_report[key])

    return summary


def projection_colorbar_label(component: str) -> str:
    """Axis label for face-on / edge-on projected maps (surface density, not ρ)."""
    return "log₁₀ Σ"


def dens_array_log10(
    dens,
    *,
    floor: float | None = None,
    vmax_pct: float = 98.0,
    vmax: float | None = None,
) -> tuple[np.ndarray, float, float, float]:
    """log10(Σ + ε) display array plus imshow limits for a raw dens map.

    Returns ``(show, vmin, vmax_show, floor)`` where ``show = log10(max(Σ,0)+ε)``.
    Pass shared ``floor`` / ``vmax`` across a column or row for matched stretch.
    """
    x = np.asarray(dens, dtype=np.float64)
    pos = x[np.isfinite(x) & (x > 0)]
    if floor is None:
        if pos.size:
            floor = max(
                float(np.percentile(pos, 20)) * 1e-2,
                float(np.percentile(pos, 99.0)) * 1e-4,
                1e-12,
            )
        else:
            floor = 1e-12
    floor = float(max(floor, 1e-30))
    if vmax is None:
        vmax = float(np.percentile(pos, vmax_pct)) if pos.size else 1.0
    vmax = float(max(vmax, 10.0 * floor))
    show = np.log10(np.maximum(x, 0.0) + floor)
    return show, float(np.log10(floor)), float(np.log10(vmax + floor)), floor


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
    label: str = "log₁₀ Σ",
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
