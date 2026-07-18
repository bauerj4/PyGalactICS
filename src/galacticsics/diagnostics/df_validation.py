"""Distribution-function validation for sampled IC components."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import interp1d

from galacticsics.io import read_disk_correction, read_frequency_table, read_harmonic_potential
from galacticsics.io.formats import cordbh_is_valid
from galacticsics.models import GalaxyModel
from galacticsics.potential.evaluate import evaluate_potential
from ntropy.analysis.disk_density import (
    bin_midplane_surface_density,
    compare_surface_density,
    disk_azimuthal_fourier,
)
from ntropy.ics.disk import ExponentialDiskParams
from ntropy.particles import ParticleState
from ntropy.softening import virial_diagnostic


def _component_mask(state: ParticleState, name: str) -> np.ndarray:
    return state.tags == name


def _disk_surface_profile(state: ParticleState, *, n_bins: int = 20, r_max: float = 22.0):
    mask = _component_mask(state, "disk")
    return bin_midplane_surface_density(state.pos[mask], state.mass[mask], n_bins=n_bins, r_max=r_max)


# Default pass/fail thresholds (tuned for N ~ 10³–10⁵ collisionless tracers).
DEFAULT_THRESHOLDS: dict[str, float] = {
    "halo_energy_df_max_log_rel": 0.45,
    "halo_speed_in_bounds_min_frac": 0.995,
    "disk_surface_density_max_rel": 0.45,
    "disk_sigma_r0_ratio_min": 0.55,
    "disk_sigma_r0_ratio_max": 1.45,
    "disk_a2_over_a0_median_max": 0.12,
    "disk_cordbh_f_d_min": 0.1,
    "disk_cordbh_f_sz_min": 0.1,
    "disk_df_positive_min_frac": 0.999,
    "bulge_energy_df_max_log_rel": 0.5,
    "virial_residual_rel_max": 0.55,
}


def _read_log_df_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    energies: list[float] = []
    log_df: list[float] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            energies.append(float(parts[0]))
            log_df.append(float(parts[1]))
    if not energies:
        raise ValueError(f"empty DF table: {path}")
    energies_arr = np.asarray(energies, dtype=float)
    log_df_arr = np.asarray(log_df, dtype=float)
    # DF tables are written with energies descending (psi0 -> psic); return
    # them ascending so interp1d(assume_sorted=True) callers are correct.
    order = np.argsort(energies_arr)
    return energies_arr[order], log_df_arr[order]


def _subsample_indices(n: int, *, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    if n <= max_samples:
        return np.arange(n, dtype=int)
    return np.sort(rng.choice(n, size=max_samples, replace=False))


def _normalized_histogram(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(values, bins=edges)
    total = hist.sum()
    if total <= 0:
        return np.zeros(len(edges) - 1, dtype=float)
    return hist.astype(float) / total


def _max_log_histogram_rel(
    sample_hist: np.ndarray,
    ref_hist: np.ndarray,
    *,
    min_ref: float = 1e-8,
) -> float:
    """Maximum |log(sample/ref)| over bins where both histograms are populated."""
    max_rel = 0.0
    for s, r in zip(sample_hist, ref_hist):
        if s <= 0 or r <= min_ref:
            continue
        max_rel = max(max_rel, abs(math.log(max(s, 1e-30) / r)))
    return max_rel


def _disk_params_from_model(model: GalaxyModel) -> ExponentialDiskParams | None:
    disk = model.disk
    if disk is None or not disk.enabled:
        return None
    return ExponentialDiskParams(
        mass=disk.mass,
        scale_length=disk.scale_length,
        outer_radius=disk.outer_radius,
        scale_height=disk.scale_height,
        trunc_width=disk.trunc_width,
    )


def _cylindrical_kinematics(pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r_cyl = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    cph = np.where(r_cyl > 0, pos[:, 0] / r_cyl, 1.0)
    sph = np.where(r_cyl > 0, pos[:, 1] / r_cyl, 0.0)
    v_r = vel[:, 0] * cph + vel[:, 1] * sph
    v_phi = -vel[:, 0] * sph + vel[:, 1] * cph
    return r_cyl, pos[:, 2], v_r, v_phi, vel[:, 2]


def _monopole_density_of_states(
    pot,
    e_values: np.ndarray,
    *,
    radial_weight: np.ndarray | None = None,
    n_radii: int = 400,
) -> np.ndarray:
    """
    (Optionally weighted) density of states ``g(E)`` for the monopole potential.

    For an ergodic DF sampled self-consistently the differential energy
    distribution is ``N(E) dE = f(E) g(E) dE`` with

    .. math::

       g(E) = 16 \\pi^2 \\int_0^{r_{max}(E)} r^2
              \\sqrt{2\\,[\\Psi(r) - E]}\\; dr,

    so a histogram comparison against a DF table must weight by ``g``;
    comparing against ``f`` alone fails even for perfect ICs.

    ``radial_weight`` (evaluated on the same radial grid, see
    :func:`_monopole_radial_grid`) generalizes this to samplers that draw
    positions from a density different from the DF's self-consistent one.
    """
    radii, psi_r = _monopole_radial_grid(pot, n_radii=n_radii)
    weight = np.ones_like(radii) if radial_weight is None else radial_weight
    g = np.zeros(len(e_values), dtype=float)
    for i, e in enumerate(e_values):
        arg = 2.0 * (psi_r - e)
        integrand = np.where(
            arg > 0.0,
            weight * radii * radii * np.sqrt(np.maximum(arg, 0.0)),
            0.0,
        )
        g[i] = 16.0 * math.pi**2 * float(np.trapezoid(integrand, radii))
    return g


def _monopole_radial_grid(pot, *, n_radii: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Radial grid and midplane monopole potential used by the g(E) integrals."""
    r_max = pot.nr * pot.dr
    radii = np.linspace(r_max / n_radii, r_max, n_radii)
    psi_r = np.array([evaluate_potential(pot, float(r), 0.0) for r in radii])
    return radii, psi_r


def _lowered_df_density(
    psi_r: np.ndarray,
    log_interp,
    *,
    psic: float,
    fcut: float,
    n_speed: int = 96,
) -> np.ndarray:
    """Velocity-space integral of the lowered DF: rho_DF(r) = 4 pi ∫ f v² dv."""
    rho = np.zeros_like(psi_r)
    for i, psi in enumerate(psi_r):
        if psi <= psic:
            continue
        v = np.linspace(0.0, math.sqrt(2.0 * (psi - psic)), n_speed)
        f = np.maximum(np.exp(log_interp(psi - 0.5 * v * v)) - fcut, 0.0)
        rho[i] = 4.0 * math.pi * float(np.trapezoid(f * v * v, v))
    return rho


def _halo_binding_energies(
    pos: np.ndarray,
    vel: np.ndarray,
    pot,
    *,
    psic: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_cyl = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    psi = np.array([evaluate_potential(pot, float(r_cyl[i]), float(pos[i, 2])) for i in range(len(pos))])
    v2 = np.sum(vel * vel, axis=1)
    energy = psi - 0.5 * v2
    speed = np.sqrt(np.maximum(v2, 0.0))
    vmax = np.sqrt(np.maximum(2.0 * (psi - psic), 0.0))
    return energy, speed, vmax


def validate_halo_df(
    state: ParticleState,
    model: GalaxyModel,
    work_dir: Path,
    *,
    thresholds: dict[str, float] | None = None,
    max_samples: int = 8000,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """
    Compare halo IC velocities to the tabulated isotropic NFW DF (``dfnfw.dat``).

    Checks binding-energy histogram vs ``dfnfw`` and that speeds stay inside the
    accessible domain ``|v| <= sqrt(2[Psi - Psi_c])``.
    """
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    work_dir = Path(work_dir)
    mask = _component_mask(state, "halo")
    n = int(mask.sum())
    out: dict[str, Any] = {"component": "halo", "n": n, "enabled": n > 0, "pass": True}
    if n == 0 or model.halo is None or not model.halo.enabled:
        out["pass"] = not (model.halo is not None and model.halo.enabled)
        out["skipped"] = True
        return out

    dfnfw = work_dir / "dfnfw.dat"
    if not dfnfw.is_file():
        out.update({"pass": False, "error": f"missing {dfnfw.name}"})
        return out

    pot = read_harmonic_potential(work_dir / "dbh.dat")
    psic = pot.psic
    energies_tab, log_df_tab = _read_log_df_table(dfnfw)
    log_interp = interp1d(
        energies_tab,
        log_df_tab,
        kind="linear",
        bounds_error=False,
        fill_value=(float(log_df_tab[0]), float(log_df_tab[-1])),
        assume_sorted=True,
    )
    log_fcut = float(log_interp(psic))
    fcut = math.exp(log_fcut)

    rng = rng or np.random.default_rng(0)
    idx = _subsample_indices(n, max_samples=max_samples, rng=rng)
    pos = state.pos[mask][idx]
    vel = state.vel[mask][idx]

    energy, speed, vmax = _halo_binding_energies(pos, vel, pot, psic=psic)
    valid = energy > psic
    if not np.any(valid):
        out.update({"pass": False, "error": "no halo particles above Psi_c"})
        return out

    e_valid = energy[valid]
    e_lo = max(float(np.percentile(e_valid, 1.0)), psic + 1e-6)
    e_hi = float(np.percentile(e_valid, 99.0))
    n_bins = min(24, max(8, len(e_valid) // 80))
    edges = np.linspace(e_lo, e_hi, n_bins + 1)
    sample_hist = _normalized_histogram(e_valid, edges)
    e_mid = 0.5 * (edges[:-1] + edges[1:])
    # genhalo draws positions from the target NFW rho(r) and velocities from the
    # lowered DF at that position, so the expected differential energy
    # distribution is N(E) ∝ f(E) ∫ r² [rho_target/rho_DF] sqrt(2[Psi-E]) dr —
    # a density-of-states integral weighted by the position/velocity mismatch.
    from galacticsics.potential.poisson.densities import halo_density_spherical

    radii, psi_r = _monopole_radial_grid(pot)
    rho_df = _lowered_df_density(psi_r, log_interp, psic=psic, fcut=fcut)
    rho_target = np.array([halo_density_spherical(float(r), model.halo) for r in radii])
    weight = np.where(rho_df > 0.0, rho_target / np.maximum(rho_df, 1e-30), 0.0)
    f_lowered = np.maximum(np.exp(log_interp(e_mid)) - fcut, 0.0)
    ref_hist = f_lowered * _monopole_density_of_states(pot, e_mid, radial_weight=weight)
    if ref_hist.sum() > 0:
        ref_hist /= ref_hist.sum()

    energy_df_max_log_rel = _max_log_histogram_rel(sample_hist, ref_hist)
    # Loosen energy-histogram tolerance for small particle counts (Poisson noise).
    energy_thr = thr["halo_energy_df_max_log_rel"] * max(1.0, 120.0 / math.sqrt(max(n, 1)))
    in_bounds = speed[valid] <= (1.001 * vmax[valid] + 1e-9)
    speed_in_bounds_frac = float(in_bounds.mean())

    out.update(
        {
            "energy_df_max_log_rel": energy_df_max_log_rel,
            "speed_in_bounds_frac": speed_in_bounds_frac,
            "n_sampled": int(len(pos)),
            "psic": float(psic),
            "thresholds": {
                "energy_df_max_log_rel": energy_thr,
                "speed_in_bounds_frac": thr["halo_speed_in_bounds_min_frac"],
            },
        }
    )
    out["pass"] = bool(
        energy_df_max_log_rel <= energy_thr
        and speed_in_bounds_frac >= thr["halo_speed_in_bounds_min_frac"]
    )
    return out


def validate_disk_df(
    state: ParticleState,
    model: GalaxyModel,
    work_dir: Path,
    *,
    thresholds: dict[str, float] | None = None,
    max_samples: int = 4000,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """
    Validate disk ICs against the epicycle DF (``diskdf5ez`` + ``cordbh.dat``).

    Reports midplane ``Sigma(R)`` vs target, global ``sigma_R`` vs ``sigma_r0``,
    ``m=2`` axisymmetry (``A2/A0``), and ``cordbh`` correction health.
    """
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    work_dir = Path(work_dir)
    mask = _component_mask(state, "disk")
    n = int(mask.sum())
    out: dict[str, Any] = {"component": "disk", "n": n, "enabled": n > 0, "pass": True}
    disk_params = _disk_params_from_model(model)
    if n == 0 or disk_params is None:
        out["pass"] = disk_params is None
        out["skipped"] = True
        return out

    cordbh_path = work_dir / "cordbh.dat"
    if not cordbh_path.is_file():
        out.update({"pass": False, "error": "missing cordbh.dat"})
        return out

    corr = read_disk_correction(cordbh_path)
    f_d_min = float(corr.f_d.min())
    f_sz_min = float(corr.f_sz.min())
    cordbh_valid = cordbh_is_valid(cordbh_path)

    _, _, v_r, _, _ = _cylindrical_kinematics(state.pos[mask], state.vel[mask])
    sig_r = float(np.std(v_r))
    sig_r_ratio = sig_r / max(model.disk_kinematics.sigma_r0, 1e-30)

    surface = _disk_surface_profile(state)
    surface_max_rel = compare_surface_density(surface, disk_params, min_count=8, skip_edges=1)

    small_n_scale = max(1.0, 120.0 / math.sqrt(max(n, 1)))
    surface_thr = thr["disk_surface_density_max_rel"] * small_n_scale
    sigma_thr_scale = max(1.0, 60.0 / math.sqrt(max(n, 1)))
    sigma_r_min = max(0.2, thr["disk_sigma_r0_ratio_min"] / sigma_thr_scale)
    sigma_r_max = thr["disk_sigma_r0_ratio_max"] * sigma_thr_scale

    fourier = disk_azimuthal_fourier(
        state.pos[mask],
        state.mass[mask],
        m=2,
        n_bins=12,
        z_max=0.3,
        min_count=max(15, n // 200),
    )
    a2_median = float(fourier["a_m_over_a0_median"])
    a2_noise = 5.0 / math.sqrt(max(n, 1))
    a2_limit = max(thr["disk_a2_over_a0_median_max"], a2_noise)

    df_positive_frac = float("nan")
    if (work_dir / "dbh.dat").is_file() and (work_dir / "freqdbh.dat").is_file():
        from galacticsics.distribution.diskdf_solve import _RcircInterpolator, _diskdf5ez
        from galacticsics.numerics import natural_cubic_spline

        pot = read_harmonic_potential(work_dir / "dbh.dat")
        freq = read_frequency_table(work_dir / "freqdbh.dat")
        rcirc_fn = _RcircInterpolator(pot, freq)
        spline_d = natural_cubic_spline(corr.radius, corr.f_d)
        spline_sz = natural_cubic_spline(corr.radius, corr.f_sz)

        rng = rng or np.random.default_rng(1)
        idx = _subsample_indices(n, max_samples=max_samples, rng=rng)
        pos = state.pos[mask][idx]
        vel = state.vel[mask][idx]
        r_cyl, z, v_r, v_phi, v_z = _cylindrical_kinematics(pos, vel)
        f_vals = np.array(
            [
                _diskdf5ez(
                    float(v_r[i]),
                    float(v_phi[i]),
                    float(v_z[i]),
                    float(r_cyl[i]),
                    float(z[i]),
                    pot,
                    freq,
                    rcirc_fn,
                    spline_d,
                    spline_sz,
                )
                for i in range(len(pos))
            ]
        )
        df_positive_frac = float(np.mean(f_vals > 0.0))

    out.update(
        {
            "surface_density_max_rel": surface_max_rel,
            "sigma_r0_ratio": sig_r_ratio,
            "a2_over_a0_median": a2_median,
            "a2_over_a0_limit": a2_limit,
            "cordbh_valid": cordbh_valid,
            "cordbh_f_d_min": f_d_min,
            "cordbh_f_sz_min": f_sz_min,
            "df_positive_frac": df_positive_frac,
            "thresholds": {
                "surface_density_max_rel": surface_thr,
                "sigma_r0_ratio": (
                    sigma_r_min,
                    sigma_r_max,
                ),
                "a2_over_a0_median": a2_limit,
                "cordbh_f_d_min": thr["disk_cordbh_f_d_min"],
                "cordbh_f_sz_min": thr["disk_cordbh_f_sz_min"],
                "df_positive_frac": thr["disk_df_positive_min_frac"],
            },
        }
    )
    out["pass"] = bool(
        surface_max_rel <= surface_thr
        and sigma_r_min <= sig_r_ratio <= sigma_r_max
        and (math.isnan(a2_median) or a2_median <= a2_limit)
        and cordbh_valid
        and f_d_min >= thr["disk_cordbh_f_d_min"]
        and f_sz_min >= thr["disk_cordbh_f_sz_min"]
        and (math.isnan(df_positive_frac) or df_positive_frac >= thr["disk_df_positive_min_frac"])
    )
    return out


def validate_bulge_df(
    state: ParticleState,
    model: GalaxyModel,
    work_dir: Path,
    *,
    thresholds: dict[str, float] | None = None,
    max_samples: int = 4000,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Compare bulge binding energies to ``dfsersic.dat`` when a bulge is enabled."""
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    work_dir = Path(work_dir)
    mask = _component_mask(state, "bulge")
    n = int(mask.sum())
    bulge_enabled = model.bulge is not None and model.bulge.enabled
    out: dict[str, Any] = {"component": "bulge", "n": n, "enabled": bulge_enabled, "pass": True}
    if not bulge_enabled:
        out["skipped"] = True
        return out
    if n == 0:
        out.update({"pass": False, "skipped": False, "error": "bulge enabled but no particles"})
        return out

    dfsersic = work_dir / "dfsersic.dat"
    if not dfsersic.is_file():
        out.update({"pass": False, "error": "missing dfsersic.dat"})
        return out

    pot = read_harmonic_potential(work_dir / "dbh.dat")
    psic = pot.psic
    energies_tab, log_df_tab = _read_log_df_table(dfsersic)
    log_interp = interp1d(
        energies_tab,
        log_df_tab,
        kind="linear",
        bounds_error=False,
        fill_value=(float(log_df_tab[0]), float(log_df_tab[-1])),
        assume_sorted=True,
    )
    log_fcut = float(log_interp(psic))
    fcut = math.exp(log_fcut)

    rng = rng or np.random.default_rng(2)
    idx = _subsample_indices(n, max_samples=max_samples, rng=rng)
    pos = state.pos[mask][idx]
    vel = state.vel[mask][idx]
    energy, speed, vmax = _halo_binding_energies(pos, vel, pot, psic=psic)
    valid = energy > psic
    if not np.any(valid):
        out.update({"pass": False, "error": "no bulge particles above Psi_c"})
        return out

    e_valid = energy[valid]
    e_lo = max(float(np.percentile(e_valid, 1.0)), psic + 1e-6)
    e_hi = float(np.percentile(e_valid, 99.0))
    n_bins = min(20, max(6, len(e_valid) // 60))
    edges = np.linspace(e_lo, e_hi, n_bins + 1)
    sample_hist = _normalized_histogram(e_valid, edges)
    e_mid = 0.5 * (edges[:-1] + edges[1:])
    f_lowered = np.maximum(np.exp(log_interp(e_mid)) - fcut, 0.0)
    ref_hist = f_lowered * _monopole_density_of_states(pot, e_mid)
    if ref_hist.sum() > 0:
        ref_hist /= ref_hist.sum()

    energy_df_max_log_rel = _max_log_histogram_rel(sample_hist, ref_hist)
    energy_thr = thr["bulge_energy_df_max_log_rel"] * max(1.0, 120.0 / math.sqrt(max(n, 1)))
    in_bounds = speed[valid] <= (1.001 * vmax[valid] + 1e-9)
    out.update(
        {
            "energy_df_max_log_rel": energy_df_max_log_rel,
            "speed_in_bounds_frac": float(in_bounds.mean()),
            "n_sampled": int(len(pos)),
            "thresholds": {"energy_df_max_log_rel": energy_thr},
        }
    )
    out["pass"] = energy_df_max_log_rel <= energy_thr
    return out


def validate_ic_distribution_functions(
    state: ParticleState,
    model: GalaxyModel,
    work_dir: Path,
    *,
    thresholds: dict[str, float] | None = None,
    include_virial: bool = False,
    include_virial_in_pass: bool = False,
) -> dict[str, Any]:
    """
    Run halo, disk, and bulge DF checks on an IC snapshot.

    Returns per-component metrics plus ``overall_pass``.

    Notes
    -----
    Pairwise :func:`virial_diagnostic` is **not** meaningful for GalactICS ICs
    (equilibrium in the fixed ``dbh`` potential, not pure self-gravity).  Pass
    ``include_virial=True`` for informational output only; it does not affect
    ``overall_pass`` unless ``include_virial_in_pass=True``.
    """
    work_dir = Path(work_dir)
    halo = validate_halo_df(state, model, work_dir, thresholds=thresholds)
    disk = validate_disk_df(state, model, work_dir, thresholds=thresholds)
    bulge = validate_bulge_df(state, model, work_dir, thresholds=thresholds)

    summary: dict[str, Any] = {
        "halo": halo,
        "disk": disk,
        "bulge": bulge,
    }
    passes = [halo.get("pass", False), disk.get("pass", False)]
    if bulge.get("enabled") and not bulge.get("skipped"):
        passes.append(bulge.get("pass", False))

    if include_virial:
        thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        virial = virial_diagnostic(
            state.pos, state.vel, state.mass, state.eps, rtol=thr["virial_residual_rel_max"]
        )
        summary["virial"] = virial
        summary["virial_pass"] = bool(virial["is_virial_equilibrium"])
        if include_virial_in_pass:
            passes.append(summary["virial_pass"])

    summary["overall_pass"] = all(passes)
    return summary
