"""Tests for accelerated Poisson polar shell integration."""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from galacticsics.models import GalaxyModel, PotentialGrid
from galacticsics.potential.poisson.fast import (
    extension_available,
    fill_density_harmonics,
    openmp_enabled,
)
from galacticsics.potential.poisson.integrals import integrate_polar_harmonics_at_shell
from galacticsics.potential.poisson.potential import PoissonArrays
from galacticsics.potential.harmonics import ComponentFlags
from galacticsics.potential.poisson.solve import solve_poisson_python
from galacticsics.potential.poisson.df_tables import write_df_artifacts
from galacticsics.potential.poisson.solve import (
    _dens_psi_bulge_factory,
    _dens_psi_halo_factory,
)


def _small_model() -> GalaxyModel:
    return replace(GalaxyModel.reference_disk_halo(), grid=PotentialGrid(dr=0.15, nr=60, lmax=2))


def _iteration_context(model: GalaxyModel, work_dir: Path):
    (
        reference_potential,
        cutoff_potential,
        inner_potential,
        df_energies,
        dens_psi_halo_tab,
        dens_psi_bulge_tab,
    ) = write_df_artifacts(work_dir, model, npsi=128, nint=8)
    dens_fn_halo = _dens_psi_halo_factory(
        df_energies, dens_psi_halo_tab, reference_potential, inner_potential
    )
    dens_fn_bulge = None
    halo_psi_tables = (df_energies, dens_psi_halo_tab, reference_potential)
    bulge_psi_tables = None
    n_harmonic_rows = model.grid.lmax // 2 + 1
    n_radial_shells = model.grid.nr
    arrays = PoissonArrays(
        apot=np.ones((n_harmonic_rows, n_radial_shells + 1), dtype=float),
        fr=np.zeros((n_harmonic_rows, n_radial_shells + 1), dtype=float),
        fr2=np.zeros((n_harmonic_rows, n_radial_shells + 1), dtype=float),
        adens=np.zeros((n_harmonic_rows, n_radial_shells + 1), dtype=float),
        lmax=model.grid.lmax,
        lmax_active=model.grid.lmax,
        flags=ComponentFlags(disk=True, halo=True),
    )
    return (
        arrays,
        cutoff_potential,
        dens_fn_halo,
        dens_fn_bulge,
        halo_psi_tables,
        bulge_psi_tables,
    )


@pytest.mark.physics_python
def test_polar_harmonics_match_single_ell_calls() -> None:
    model = _small_model()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        arrays, psic, dens_halo, dens_bulge, halo_tables, bulge_tables = _iteration_context(
            model, work
        )
        active_ells = [0, 2]
        shell_radius = 3.0 * model.grid.dr
        batch = integrate_polar_harmonics_at_shell(
            arrays,
            model,
            shell_radius,
            active_ells,
            ntheta=15,
            dens_psi_halo=dens_halo,
            dens_psi_bulge=dens_bulge,
            psic=psic,
            halo_psi_tables=halo_tables,
            bulge_psi_tables=bulge_tables,
        )
        for ell in active_ells:
            single = integrate_polar_harmonics_at_shell(
                arrays,
                model,
                shell_radius,
                [ell],
                ntheta=15,
                dens_psi_halo=dens_halo,
                dens_psi_bulge=dens_bulge,
                psic=psic,
                halo_psi_tables=halo_tables,
                bulge_psi_tables=bulge_tables,
            )[ell]
            assert batch[ell] == pytest.approx(single, rel=1e-10, abs=1e-12)


@pytest.mark.physics_python
def test_fast_python_fill_matches_direct_integration() -> None:
    model = _small_model()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        arrays, psic, dens_halo, dens_bulge, halo_tables, bulge_tables = _iteration_context(
            model, work
        )
        adens_direct = np.zeros_like(arrays.adens)
        active_lmax = model.grid.lmax
        active_ells = list(range(0, active_lmax + 1, 2))
        for shell_index in range(1, model.grid.nr + 1):
            moments = integrate_polar_harmonics_at_shell(
                arrays,
                model,
                shell_index * model.grid.dr,
                active_ells,
                ntheta=15,
                dens_psi_halo=dens_halo,
                dens_psi_bulge=dens_bulge,
                psic=psic,
                halo_psi_tables=halo_tables,
                bulge_psi_tables=bulge_tables,
            )
            for ell in active_ells:
                adens_direct[ell // 2, shell_index] = moments[ell]

        adens_fast = np.zeros_like(arrays.adens)
        fill_density_harmonics(
            adens_fast,
            arrays=arrays,
            model=model,
            radial_step=model.grid.dr,
            n_radial_shells=model.grid.nr,
            active_lmax=active_lmax,
            n_polar_nodes=15,
            dens_fn_halo=dens_halo,
            dens_fn_bulge=dens_bulge,
            cutoff_potential=psic,
            n_workers=0,
            halo_psi_tables=halo_tables,
            bulge_psi_tables=bulge_tables,
            n_threads=0,
        )
        assert adens_fast == pytest.approx(adens_direct, rel=1e-9, abs=1e-11)


@pytest.mark.physics_python
@pytest.mark.skipif(not extension_available(), reason="Poisson C extension not built")
def test_openmp_fill_matches_python() -> None:
    model = _small_model()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        arrays, psic, dens_halo, dens_bulge, halo_tables, bulge_tables = _iteration_context(
            model, work
        )
        adens_py = np.zeros_like(arrays.adens)
        adens_c = np.zeros_like(arrays.adens)
        kwargs = dict(
            arrays=arrays,
            model=model,
            radial_step=model.grid.dr,
            n_radial_shells=model.grid.nr,
            active_lmax=model.grid.lmax,
            n_polar_nodes=15,
            dens_fn_halo=dens_halo,
            dens_fn_bulge=dens_bulge,
            cutoff_potential=psic,
            n_workers=0,
            halo_psi_tables=halo_tables,
            bulge_psi_tables=bulge_tables,
        )
        fill_density_harmonics(adens_py, n_threads=0, **kwargs)
        fill_density_harmonics(adens_c, n_threads=4, **kwargs)
        assert openmp_enabled()
        assert adens_c == pytest.approx(adens_py, rel=2e-4, abs=1e-9)


@pytest.mark.physics_python
def test_solve_poisson_python_small_grid() -> None:
    model = _small_model()
    with tempfile.TemporaryDirectory() as tmp:
        pot = solve_poisson_python(model, Path(tmp), npsi=128, nint=8, max_iter=12, n_workers=0)
        assert pot.psi0 != 0.0
        assert np.all(np.isfinite(pot.adens))
