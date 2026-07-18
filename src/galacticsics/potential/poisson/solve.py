"""
Python self-consistent multipole Poisson solver (legacy ``dbh.f`` port).

The iteration alternates between:

1. **Density harmonics** — polar quadrature of ``rho(s, z)`` at each radial shell.
2. **Potential harmonics** — Binney & Tremaine eq. 2-208 radial synthesis.
3. **Convergence** — tidal-radius stability and gradual ``l_max`` ramp.

See :doc:`/docs/dbh_python_backend` for naming conventions and performance knobs.
"""

from __future__ import annotations

import math
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from galacticsics.io.formats import write_harmonic_potential
from galacticsics.models import GalaxyModel
from galacticsics.potential.harmonics import ComponentFlags, HarmonicPotential
from galacticsics.potential.poisson.df_tables import write_df_artifacts
from galacticsics.potential.poisson.densities import (
    bulge_density_on_grid,
    bulge_density_psi,
    halo_density_spherical,
    halo_density_spherical_array,
)
from galacticsics.potential.poisson.appdisk import disk_monopole_density_grid
from galacticsics.potential.poisson.integrals import (
    integrate_polar_density_spherical,
    monopole_estimate_from_spherical_density,
    poisson_harmonics_from_density,
)
from galacticsics.potential.poisson.potential import PoissonArrays, potential_at
from galacticsics.potential.poisson.sersic import bulge_mass_spherical
from galacticsics.numerics import quadrature_node_count


def _assert_python_supported(model: GalaxyModel) -> None:
    """
    Validate that the model uses only Python-supported components.

    Supported combinations include halo with any subset of disk and bulge.
    Gas, second disk, and black holes remain legacy-only.

    Parameters
    ----------
    model : GalaxyModel
        Galaxy configuration to validate.

    Raises
    ------
    NotImplementedError
        When an unsupported component is enabled.
    ValueError
        When halo-only models request ``lmax > 0``.
    """
    if model.disk2 and model.disk2.enabled:
        raise NotImplementedError("Python backend: disk2 not yet supported")
    if model.gas and model.gas.enabled:
        raise NotImplementedError("Python backend: gas not yet supported")
    if model.black_hole and model.black_hole.enabled:
        raise NotImplementedError("Python backend: black hole not yet supported")
    if not (model.halo and model.halo.enabled):
        raise ValueError("Python backend requires an enabled halo")
    has_disk = model.disk is not None and model.disk.enabled
    has_bulge = model.bulge is not None and model.bulge.enabled
    has_baryon = has_disk or has_bulge
    if not has_baryon and model.grid.lmax > 0:
        raise ValueError(
            "Python backend halo-only solves require lmax=0 "
            "(enable disk or bulge for multipole lmax > 0)"
        )


def _scaled_polar_node_count(max_harmonic_degree: int, n_radial_shells: int) -> int:
    """
    Polar quadrature node count scaled to grid resolution.

    Parameters
    ----------
    max_harmonic_degree : int
        Configured ``l_max`` on the radial grid.
    n_radial_shells : int
        Number of radial shells ``nr``.

    Returns
    -------
    ntheta : int
        Odd node count suitable for Simpson quadrature.
    """
    return quadrature_node_count(max(6, min(max_harmonic_degree * 4 + 2, 12 + n_radial_shells // 20)))


def _resolve_solve_workers(n_workers: int | None) -> int:
    """
    Resolve thread-pool size for parallel shell integration.

    Parameters
    ----------
    n_workers : int or None
        Explicit worker count.  When ``None``, reads ``GALACTICSICS_SOLVE_WORKERS``
        (default ``0`` = serial).

    Returns
    -------
    n_workers : int
        Non-negative worker count.
    """
    if n_workers is None:
        n_workers = int(os.environ.get("GALACTICSICS_SOLVE_WORKERS", "0"))
    return max(0, int(n_workers))


def _fill_density_harmonics_at_shells(
    density_harmonics: np.ndarray,
    *,
    arrays: PoissonArrays,
    model: GalaxyModel,
    radial_step: float,
    n_radial_shells: int,
    active_lmax: int,
    n_polar_nodes: int,
    dens_fn_halo,
    dens_fn_bulge,
    cutoff_potential: float,
    n_workers: int,
    halo_psi_tables: tuple[np.ndarray, np.ndarray, float] | None,
    bulge_psi_tables: tuple[np.ndarray, np.ndarray, float, float] | None,
) -> None:
    """
    Accumulate density harmonics for one Poisson iteration.

    Parameters
    ----------
    density_harmonics : ndarray, shape (n_harmonic_rows, nr + 1)
        Output buffer ``adens``; overwritten in place for shells ``ir >= 1``.
    arrays : PoissonArrays
        Current potential harmonic state.
    model : GalaxyModel
        Galaxy configuration.
    radial_step : float
        Radial grid spacing ``dr`` [kpc].
    n_radial_shells : int
        Number of radial shells ``nr``.
    active_lmax : int
        Active even harmonic degree during this iteration.
    n_polar_nodes : int
        Polar quadrature node count.
    dens_fn_halo, dens_fn_bulge : callable or None
        Scalar ``rho(psi)`` interpolators.
    cutoff_potential : float
        DF cutoff potential ``psic`` [100 km/s]\\ :sup:`2`].
    n_workers : int
        Thread-pool size (``0`` or ``1`` runs serially).
    halo_psi_tables, bulge_psi_tables : tuple, optional
        Pre-tabulated DF lookup tables for vectorized density evaluation.
    """
    from galacticsics.potential.poisson.fast import fill_density_harmonics

    fill_density_harmonics(
        density_harmonics,
        arrays=arrays,
        model=model,
        radial_step=radial_step,
        n_radial_shells=n_radial_shells,
        active_lmax=active_lmax,
        n_polar_nodes=n_polar_nodes,
        dens_fn_halo=dens_fn_halo,
        dens_fn_bulge=dens_fn_bulge,
        cutoff_potential=cutoff_potential,
        n_workers=n_workers,
        halo_psi_tables=halo_psi_tables,
        bulge_psi_tables=bulge_psi_tables,
    )


def _halo_density_normalization(halo) -> float:
    """Legacy ``haloconst`` amplitude for ``dbh.dat`` metadata."""
    return (2.0 ** (1.0 - halo.cusp)) * halo.v0**2 / (4.0 * math.pi * halo.a**2)


def _write_halo_only_harmonics(
    pot: HarmonicPotential,
    *,
    model: GalaxyModel,
    work_dir: Path,
    max_harmonic_degree: int,
    radial_step: float,
    n_radial_shells: int,
) -> None:
    """
    Build and write halo-only harmonics to ``h.dat``.

    Parameters
    ----------
    pot : HarmonicPotential
        Combined potential from the main solve.
    model : GalaxyModel
        Galaxy configuration.
    work_dir : Path
        Output directory.
    max_harmonic_degree : int
        Maximum configured ``l_max``.
    radial_step : float
        Radial grid spacing [kpc].
    n_radial_shells : int
        Number of radial shells ``nr``.
    """
    halo = model.halo
    assert halo is not None
    n_polar_nodes = _scaled_polar_node_count(max_harmonic_degree, n_radial_shells)
    halo_density_harmonics = np.zeros_like(pot.adens)

    def _halo_rho(radii: np.ndarray) -> np.ndarray:
        return halo_density_spherical_array(radii, halo)

    for harmonic_degree in range(0, max_harmonic_degree + 1, 2):
        harmonic_row = harmonic_degree // 2
        for shell_index in range(1, n_radial_shells + 1):
            shell_radius = shell_index * radial_step
            halo_density_harmonics[harmonic_row, shell_index] = integrate_polar_density_spherical(
                shell_radius,
                n_polar_nodes,
                harmonic_degree,
                lambda rad, h=halo: halo_density_spherical(rad, h),
                rho_array_fn=_halo_rho,
            )

    halo_potential_harmonics, halo_force_harmonics, halo_force_second_harmonics = (
        poisson_harmonics_from_density(
            halo_density_harmonics,
            dr=radial_step,
            nr=n_radial_shells,
            lmax=max_harmonic_degree,
            frac=0.0,
            lmax_old=-2,
        )
    )
    halo_pot = replace(
        pot,
        flags=ComponentFlags(halo=True),
        adens=halo_density_harmonics,
        apot=halo_potential_harmonics,
        fr=halo_force_harmonics,
        fr2=halo_force_second_harmonics,
    )
    write_harmonic_potential(halo_pot, work_dir / "h.dat")


def _solve_halo_monopole_python(
    model: GalaxyModel,
    work_dir: Path,
    *,
    npsi: int,
    nint: int,
) -> HarmonicPotential:
    """Fast spherical halo solve (``lmax=0``, no baryons)."""
    return _solve_spherical_monopole_python(
        model,
        work_dir,
        npsi=npsi,
        nint=nint,
        has_bulge=False,
    )


def _solve_spherical_monopole_python(
    model: GalaxyModel,
    work_dir: Path,
    *,
    npsi: int,
    nint: int,
    has_bulge: bool,
) -> HarmonicPotential:
    """
    Fast spherical solve for halo-only or bulge+halo (no disk).

    Disk-free layouts are spherically symmetric; direct ``rho(r)`` monopole
    synthesis avoids degenerate DF ``rho(psi)`` tables when ``psi0 ~ psid``.

    Parameters
    ----------
    model : GalaxyModel
        Galaxy with enabled halo and optional bulge.
    work_dir : Path
        Output directory for ``dbh.dat`` and auxiliary tables.
    npsi, nint : int
        DF table resolution.
    has_bulge : bool
        When ``True``, include bulge density in the monopole.

    Returns
    -------
    HarmonicPotential
        Converged monopole potential.
    """
    radial_step = model.grid.dr
    n_radial_shells = model.grid.nr
    max_harmonic_degree = model.grid.lmax
    n_harmonic_rows = max_harmonic_degree // 2 + 1
    halo = model.halo
    assert halo is not None
    bulge = model.bulge if has_bulge else None

    reference_potential, cutoff_potential, inner_potential, _, _, _ = write_df_artifacts(
        work_dir, model, npsi=npsi, nint=nint
    )
    if model.halo is None or not model.halo.enabled:
        raise ValueError("spherical monopole solve requires an enabled halo")
    if has_bulge and (model.bulge is None or not model.bulge.enabled):
        raise ValueError("bulge+halo spherical solve requires an enabled bulge")

    halo_normalization = _halo_density_normalization(halo)
    bulge_normalization = bulge.bulge_const if bulge is not None else 0.0

    grid_radii = np.arange(n_radial_shells + 1, dtype=float) * radial_step
    halo_density = halo_density_spherical_array(grid_radii, halo)
    if has_bulge and bulge is not None:
        bulge_density = bulge_density_on_grid(grid_radii, bulge)
        total_density = halo_density + bulge_density
    else:
        total_density = halo_density

    halo_monopole_potential, halo_monopole_force = monopole_estimate_from_spherical_density(
        total_density, dr=radial_step, nr=n_radial_shells
    )
    potential_harmonics = np.zeros((n_harmonic_rows, n_radial_shells + 1), dtype=float)
    force_harmonics = np.zeros((n_harmonic_rows, n_radial_shells + 1), dtype=float)
    for shell_index in range(n_radial_shells + 1):
        potential_harmonics[0, shell_index] = math.sqrt(4 * math.pi) * halo_monopole_potential[shell_index]
        force_harmonics[0, shell_index] = halo_monopole_force[shell_index]
    monopole_at_origin = potential_harmonics[0, 0]
    potential_harmonics[0, :] = (
        potential_harmonics[0, :]
        + reference_potential * math.sqrt(4 * math.pi)
        - monopole_at_origin
    )

    halo_mass = float(4 * math.pi * np.sum(halo_density[1:] * grid_radii[1:] ** 2) * radial_step)
    halo_outer_radius = halo.r_outer
    if has_bulge and bulge is not None:
        bulge_mass = bulge_mass_spherical(grid_radii, bulge, radial_step)
        bulge_outer_radius = 3.0 * bulge.a
    else:
        bulge_mass = 0.0
        bulge_outer_radius = 0.0

    tidal_radius = halo.r_outer
    for shell_index in range(1, n_radial_shells + 1):
        potential_inner = halo_monopole_potential[shell_index - 1]
        potential_outer = halo_monopole_potential[shell_index]
        if (potential_inner - cutoff_potential) * (potential_outer - cutoff_potential) <= 0 and (
            potential_outer != potential_inner
        ):
            tidal_radius = (
                shell_index - 1 - (potential_inner - cutoff_potential) / (potential_outer - potential_inner)
            ) * radial_step
            break

    (work_dir / "rtidal.dat").write_text(f"{tidal_radius}\n")
    (work_dir / "mr.dat").write_text(
        f"0.0 0.0\n{bulge_mass} {bulge_outer_radius}\n{halo_mass} {halo_outer_radius}\n"
    )

    density_harmonics = np.zeros((n_harmonic_rows, n_radial_shells + 1), dtype=float)
    for shell_index in range(1, n_radial_shells + 1):
        density_harmonics[0, shell_index] = (
            4 * math.pi * total_density[shell_index] * grid_radii[shell_index] ** 2
        )

    _, _, force_second_harmonics = poisson_harmonics_from_density(
        density_harmonics, dr=radial_step, nr=n_radial_shells, lmax=max_harmonic_degree, frac=0.0, lmax_old=-2
    )

    flags = ComponentFlags(bulge=has_bulge, halo=True)
    pot = HarmonicPotential(
        model=model,
        psi0=reference_potential,
        haloconst=halo_normalization,
        bulgeconst=bulge_normalization,
        psic=cutoff_potential,
        psid=inner_potential,
        flags=flags,
        radii=grid_radii,
        adens=density_harmonics,
        apot=potential_harmonics,
        fr=force_harmonics,
        fr2=force_second_harmonics,
    )
    write_harmonic_potential(pot, work_dir / "dbh.dat")
    _write_halo_only_harmonics(
        pot,
        model=model,
        work_dir=work_dir,
        max_harmonic_degree=max_harmonic_degree,
        radial_step=radial_step,
        n_radial_shells=n_radial_shells,
    )
    return pot


def _initial_monopole(
    model: GalaxyModel,
    halo_monopole_potential: np.ndarray,
    disk_monopole_potential: np.ndarray,
    bulge_monopole_potential: np.ndarray,
) -> np.ndarray:
    """
    Seed multipole coefficients from monopole potential estimates.

    Parameters
    ----------
    model : GalaxyModel
        Grid parameters.
    halo_monopole_potential, disk_monopole_potential, bulge_monopole_potential : ndarray, shape (nr + 1,)
        Component monopole potentials [100 km/s]\\ :sup:`2`].

    Returns
    -------
    potential_harmonics : ndarray, shape (n_harmonic_rows, nr + 1)
        Initial potential harmonics; only the monopole row is filled.
    """
    n_radial_shells = model.grid.nr
    potential_harmonics = np.zeros((model.grid.lmax // 2 + 1, n_radial_shells + 1), dtype=float)
    for shell_index in range(1, n_radial_shells + 1):
        total_monopole = (
            halo_monopole_potential[shell_index]
            + disk_monopole_potential[shell_index]
            + bulge_monopole_potential[shell_index]
        )
        potential_harmonics[0, shell_index] = math.sqrt(4 * math.pi) * total_monopole
    return potential_harmonics


def _scaled_df_resolution(model: GalaxyModel, npsi: int, nint: int) -> tuple[int, int]:
    """
    Reduce DF table cost on coarse grids (tests / campaigns).

    Parameters
    ----------
    model : GalaxyModel
        Galaxy whose ``grid.nr`` sets the cap.
    npsi, nint : int
        Requested table resolution.

    Returns
    -------
    npsi_eff, nint_eff : tuple of int
        Capped resolution values.
    """
    n_radial_shells = model.grid.nr
    return min(npsi, max(128, n_radial_shells * 2)), min(nint, max(8, n_radial_shells // 20 + 4))


def _scaled_max_iter(n_radial_shells: int, max_harmonic_degree: int, max_iter: int) -> int:
    """
    Cap Poisson iteration count for coarse grids.

    Parameters
    ----------
    n_radial_shells : int
        Radial shell count ``nr``.
    max_harmonic_degree : int
        Configured ``l_max``.
    max_iter : int
        User-requested upper bound.

    Returns
    -------
    max_iter_eff : int
        Effective iteration cap.
    """
    return min(max_iter, max(16, 12 + n_radial_shells // 6 + max_harmonic_degree * 3))


def _dens_psi_halo_factory(
    energies: np.ndarray,
    dens_psi: np.ndarray,
    reference_potential: float,
    inner_potential: float,
):
    """
    Build ``rho_halo(psi)`` interpolator from tabulated halo densities.

    Parameters
    ----------
    energies : ndarray, shape (npsi,)
        DF energy grid.
    dens_psi : ndarray, shape (npsi,)
        Tabulated halo density.
    reference_potential : float
        Potential at the origin ``psi0``.
    inner_potential : float
        Inner DF energy ``psid`` (unused in lookup; kept for API parity).

    Returns
    -------
    interp : callable
        Scalar ``rho(psi)`` evaluator.
    """
    from scipy.interpolate import interp1d

    order = np.argsort(energies)
    e_sorted = energies[order]
    d_sorted = dens_psi[order]
    dens_interp = interp1d(
        e_sorted,
        d_sorted,
        kind="linear",
        bounds_error=False,
        fill_value=(float(d_sorted[0]), float(d_sorted[-1])),
        assume_sorted=True,
    )

    def interp(psi: float) -> float:
        if psi >= reference_potential:
            return float(dens_psi[0])
        return float(dens_interp(psi))

    return interp


def _dens_psi_bulge_factory(
    energies: np.ndarray,
    dens_psi: np.ndarray,
    *,
    reference_potential: float,
    cutoff_potential: float,
    inner_potential: float,
):
    """
    Build ``rho_bulge(psi)`` interpolator from ``denspsibulge.dat`` rows.

    Parameters
    ----------
    energies : ndarray, shape (npsi,)
        DF energy grid.
    dens_psi : ndarray, shape (npsi,)
        Tabulated bulge density.
    reference_potential, cutoff_potential, inner_potential : float
        Legacy ``psi0``, ``psic``, ``psid`` energies.

    Returns
    -------
    interp : callable
        Scalar ``rho(psi)`` evaluator.
    """

    def interp(psi: float) -> float:
        return bulge_density_psi(
            psi,
            energies,
            dens_psi,
            psi0=reference_potential,
            psic=cutoff_potential,
            psid=inner_potential,
        )

    return interp


def solve_poisson_python(
    model: GalaxyModel,
    work_dir: Path,
    *,
    npsi: int = 1000,
    nint: int = 20,
    max_iter: int = 100,
    n_workers: int | None = None,
) -> HarmonicPotential:
    """
    Self-consistent multipole solve for halo with disk and/or bulge.

    Writes ``dbh.dat``, ``h.dat``, ``mr.dat``, ``rtidal.dat``, and DF tables
    for each enabled stellar component.

    Parameters
    ----------
    model : GalaxyModel
        Galaxy with enabled halo and optional disk / bulge.
    work_dir : Path
        Output directory.
    npsi, nint : int, optional
        DF table resolution passed to
        :func:`~galacticsics.potential.poisson.df_tables.write_df_artifacts`.
    max_iter : int, optional
        Upper bound on Poisson iteration count (auto-scaled for coarse grids).
    n_workers : int, optional
        Thread pool size for radial shell integration (``0`` = serial).  Falls
        back to ``GALACTICSICS_SOLVE_WORKERS`` when omitted.

    Returns
    -------
    HarmonicPotential
        Converged combined potential written to ``dbh.dat``.

    Notes
    -----
    **Performance.** Set ``n_workers`` to the number of physical cores (often
    4–8) for a noticeable speedup on disk+halo models.  Coarse grids
    automatically reduce polar node counts, DF table sizes, and iteration caps.

    **Stability.** Accurate monopole seeding uses ``diskdensestimate`` (not
    ``appdiskdens``); see :mod:`galacticsics.potential.poisson.appdisk`.
    """
    _assert_python_supported(model)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    has_disk = model.disk is not None and model.disk.enabled
    has_bulge = model.bulge is not None and model.bulge.enabled
    eff_npsi, eff_nint = _scaled_df_resolution(model, npsi, nint)
    workers = _resolve_solve_workers(n_workers)
    if not has_disk and not has_bulge and model.grid.lmax == 0:
        return _solve_halo_monopole_python(
            model, work_dir, npsi=eff_npsi, nint=eff_nint
        )
    if not has_disk and has_bulge:
        return _solve_spherical_monopole_python(
            model,
            work_dir,
            npsi=eff_npsi,
            nint=eff_nint,
            has_bulge=True,
        )

    radial_step = model.grid.dr
    n_radial_shells = model.grid.nr
    max_harmonic_degree = model.grid.lmax
    n_harmonic_rows = max_harmonic_degree // 2 + 1
    halo = model.halo
    assert halo is not None
    if has_disk:
        assert model.disk is not None
    max_iter = _scaled_max_iter(n_radial_shells, max_harmonic_degree, max_iter)

    (
        reference_potential,
        cutoff_potential,
        inner_potential,
        df_energies,
        dens_psi_halo_tab,
        dens_psi_bulge_tab,
    ) = write_df_artifacts(work_dir, model, npsi=eff_npsi, nint=eff_nint)

    dens_fn_halo = (
        _dens_psi_halo_factory(df_energies, dens_psi_halo_tab, reference_potential, inner_potential)
        if dens_psi_halo_tab is not None
        else None
    )
    dens_fn_bulge = (
        _dens_psi_bulge_factory(
            df_energies,
            dens_psi_bulge_tab,
            reference_potential=reference_potential,
            cutoff_potential=cutoff_potential,
            inner_potential=inner_potential,
        )
        if dens_psi_bulge_tab is not None
        else None
    )
    halo_psi_tables = (
        (df_energies, dens_psi_halo_tab, reference_potential)
        if dens_psi_halo_tab is not None
        else None
    )
    bulge_psi_tables = (
        (df_energies, dens_psi_bulge_tab, reference_potential, inner_potential)
        if dens_psi_bulge_tab is not None
        else None
    )

    halo_normalization = _halo_density_normalization(halo)
    bulge_normalization = model.bulge.bulge_const if has_bulge and model.bulge else 0.0

    grid_radii = np.arange(n_radial_shells + 1, dtype=float) * radial_step
    halo_density = halo_density_spherical_array(grid_radii, halo)
    halo_monopole_potential, _ = monopole_estimate_from_spherical_density(
        halo_density, dr=radial_step, nr=n_radial_shells
    )
    if has_disk:
        disk_monopole_density = disk_monopole_density_grid(model, ntheta=100)
        disk_monopole_potential, _ = monopole_estimate_from_spherical_density(
            disk_monopole_density, dr=radial_step, nr=n_radial_shells
        )
    else:
        disk_monopole_potential = np.zeros(n_radial_shells + 1, dtype=float)
    if has_bulge and model.bulge is not None:
        bulge_density = bulge_density_on_grid(grid_radii, model.bulge)
        bulge_monopole_potential, _ = monopole_estimate_from_spherical_density(
            bulge_density, dr=radial_step, nr=n_radial_shells
        )
    else:
        bulge_monopole_potential = np.zeros(n_radial_shells + 1, dtype=float)

    flags = ComponentFlags(disk=has_disk, bulge=has_bulge, halo=True)
    potential_harmonics = _initial_monopole(
        model, halo_monopole_potential, disk_monopole_potential, bulge_monopole_potential
    )
    force_harmonics = np.zeros((n_harmonic_rows, n_radial_shells + 1), dtype=float)
    force_second_harmonics = np.zeros((n_harmonic_rows, n_radial_shells + 1), dtype=float)
    density_harmonics = np.zeros((n_harmonic_rows, n_radial_shells + 1), dtype=float)

    arrays = PoissonArrays(
        apot=potential_harmonics,
        fr=force_harmonics,
        fr2=force_second_harmonics,
        adens=density_harmonics,
        lmax=max_harmonic_degree,
        lmax_active=0,
        flags=flags,
    )

    active_lmax = 0
    harmonic_ramp_step = 2
    previous_active_lmax = -2
    previous_tidal_radius = 1e10
    tidal_radius_change = 2 * radial_step
    potential_relaxation_factor = 0.75
    tidal_radius = halo.r_outer

    for iteration in range(max_iter):
        if active_lmax == 0 or active_lmax == max_harmonic_degree:
            if tidal_radius_change < radial_step and iteration > 10:
                active_lmax = min(active_lmax + harmonic_ramp_step, max_harmonic_degree)
        else:
            active_lmax = min(active_lmax + harmonic_ramp_step, max_harmonic_degree)
        if active_lmax > max_harmonic_degree:
            break
        arrays.lmax_active = active_lmax
        n_polar_nodes = _scaled_polar_node_count(active_lmax, n_radial_shells)

        density_harmonics.fill(0.0)

        _fill_density_harmonics_at_shells(
            density_harmonics,
            arrays=arrays,
            model=model,
            radial_step=radial_step,
            n_radial_shells=n_radial_shells,
            active_lmax=active_lmax,
            n_polar_nodes=n_polar_nodes,
            dens_fn_halo=dens_fn_halo,
            dens_fn_bulge=dens_fn_bulge,
            cutoff_potential=cutoff_potential,
            n_workers=workers,
            halo_psi_tables=halo_psi_tables,
            bulge_psi_tables=bulge_psi_tables,
        )

        potential_new, force_new, force_second_new = poisson_harmonics_from_density(
            density_harmonics,
            dr=radial_step,
            nr=n_radial_shells,
            lmax=active_lmax,
            apot_in=arrays.apot,
            frac=potential_relaxation_factor,
            lmax_old=previous_active_lmax,
        )
        for harmonic_row in range(active_lmax // 2 + 1):
            arrays.apot[harmonic_row, :] = potential_new[harmonic_row, :]
            arrays.fr[harmonic_row, :] = force_new[harmonic_row, :]
            arrays.fr2[harmonic_row, :] = force_second_new[harmonic_row, :]

        monopole_at_origin = arrays.apot[0, 0]
        arrays.apot[0, :] = (
            arrays.apot[0, :] + reference_potential * math.sqrt(4 * math.pi) - monopole_at_origin
        )

        potential_at_shell = reference_potential
        tidal_radius = halo.r_outer
        for shell_index in range(1, n_radial_shells + 1):
            potential_at_inner_shell = potential_at_shell
            potential_at_shell = potential_at(arrays, model, shell_index * radial_step, 0.0)
            if (
                (potential_at_inner_shell - cutoff_potential)
                * (potential_at_shell - cutoff_potential)
                <= 0
                and potential_at_shell != potential_at_inner_shell
            ):
                tidal_radius = (
                    shell_index - 1
                    - (potential_at_inner_shell - cutoff_potential)
                    / (potential_at_shell - potential_at_inner_shell)
                ) * radial_step
                tidal_radius_change = abs(tidal_radius - previous_tidal_radius)
                previous_tidal_radius = tidal_radius
                break
        else:
            tidal_radius_change = 2 * radial_step

        previous_active_lmax = active_lmax
        if active_lmax >= max_harmonic_degree and tidal_radius_change < radial_step:
            break

    arrays.lmax_active = max_harmonic_degree
    total_mass = arrays.fr[0, n_radial_shells] / math.sqrt(4 * math.pi) * (radial_step * n_radial_shells) ** 2
    halo_mass = float(4 * math.pi * np.sum(halo_density[1:] * grid_radii[1:] ** 2) * radial_step)
    halo_outer_radius = halo.r_outer
    if has_disk and model.disk is not None:
        disk_mass = max(total_mass - halo_mass, model.disk.mass * 0.5)
        disk_outer_radius = model.disk.outer_radius + 2 * model.disk.trunc_width
    else:
        disk_mass = 0.0
        disk_outer_radius = 0.0
    if has_bulge and model.bulge is not None:
        bulge_mass = bulge_mass_spherical(grid_radii, model.bulge, radial_step)
        bulge_outer_radius = 3.0 * model.bulge.a
    else:
        bulge_mass = 0.0
        bulge_outer_radius = 0.0

    (work_dir / "rtidal.dat").write_text(f"{tidal_radius}\n")
    (work_dir / "mr.dat").write_text(
        f"{disk_mass} {disk_outer_radius}\n{bulge_mass} {bulge_outer_radius}\n{halo_mass} {halo_outer_radius}\n"
    )

    pot = HarmonicPotential(
        model=model,
        psi0=reference_potential,
        haloconst=halo_normalization,
        bulgeconst=bulge_normalization,
        psic=cutoff_potential,
        psid=inner_potential,
        flags=flags,
        radii=np.linspace(0, n_radial_shells * radial_step, n_radial_shells + 1),
        adens=arrays.adens.copy(),
        apot=arrays.apot.copy(),
        fr=arrays.fr.copy(),
        fr2=arrays.fr2.copy(),
    )
    write_harmonic_potential(pot, work_dir / "dbh.dat")
    _write_halo_only_harmonics(
        pot,
        model=model,
        work_dir=work_dir,
        max_harmonic_degree=max_harmonic_degree,
        radial_step=radial_step,
        n_radial_shells=n_radial_shells,
    )
    return pot
