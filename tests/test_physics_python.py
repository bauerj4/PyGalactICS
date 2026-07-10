"""Tests for Python physics backend IC chain."""

from __future__ import annotations

import math
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
import numpy as np

from galacticsics.distribution.frequencies import FrequencyTable
from galacticsics.io import read_disk_correction, read_frequency_table, read_harmonic_potential
from galacticsics.io.formats import cordbh_is_valid
from galacticsics.models import GalaxyModel, PotentialGrid
from galacticsics.physics.backend import PhysicsBackendKind
from galacticsics.potential.evaluate import evaluate_potential
from galacticsics.potential.frequencies_tabulate import tabulate_frequencies
from galacticsics.potential.solver import solve_potential
from galacticsics.sampling.sampler import SampleConfig, sample_galaxy


def _small_model() -> GalaxyModel:
    model = GalaxyModel.reference_disk_halo()
    return replace(model, grid=PotentialGrid(dr=0.15, nr=60, lmax=2))


@pytest.fixture(scope="module")
def python_work_dir() -> Path:
    import shutil

    from galacticsics.io.legacy_inputs import write_gendenspsi_input
    from tests.fixtures.milkyway import reference_artifacts_dir

    model = _small_model()
    tmp = tempfile.TemporaryDirectory(prefix="galacticsics_py_physics_")
    work = Path(tmp.name)
    solve_potential(model, work_dir=work, cleanup=False, backend=PhysicsBackendKind.PYTHON)
    ref = reference_artifacts_dir()
    for name in ("cordbh.dat",):
        src = ref / name
        if src.is_file():
            shutil.copy2(src, work / name)
    if not (work / "cordbh.dat").is_file():
        from galacticsics.distribution.diskdf_solve import solve_diskdf_python

        solve_diskdf_python(model, work, n_iterations=1, n_radial_steps=12)
    write_gendenspsi_input(work / "in.gendenspsi")
    yield work
    tmp.cleanup()


@pytest.mark.physics_python
def test_python_solve_writes_dbh(python_work_dir: Path) -> None:
    assert (python_work_dir / "dbh.dat").is_file()
    assert (python_work_dir / "h.dat").is_file()
    assert (python_work_dir / "freqdbh.dat").is_file()
    pot = read_harmonic_potential(python_work_dir / "dbh.dat")
    assert pot.psi0 != 0.0
    psi_cen = evaluate_potential(pot, 0.0, 0.0)
    assert math.isfinite(psi_cen)


@pytest.mark.physics_python
def test_python_getfreqs_engineering_parity(reference_artifacts_dir: Path, tmp_path: Path) -> None:
    """Python ``getfreqs`` on legacy ``dbh.dat`` should match legacy ``freqdbh.dat``."""
    import shutil

    work = tmp_path / "freq"
    work.mkdir()
    for name in ("dbh.dat", "h.dat"):
        shutil.copy2(reference_artifacts_dir / name, work / name)
    tabulate_frequencies(work)
    legacy = read_frequency_table(reference_artifacts_dir / "freqdbh.dat")
    python = read_frequency_table(work / "freqdbh.dat")
    assert isinstance(python, FrequencyTable)
    for r_test in (1.0, 2.0, 5.0, 8.0):
        assert python.omega(r_test) == pytest.approx(legacy.omega(r_test), rel=0.05)
        idx = int(round(r_test / legacy.radius[1]))
        assert python.v_circ_total[idx] == pytest.approx(legacy.v_circ_total[idx], rel=0.05)
    assert python.kappa(2.0) > 0.0


@pytest.mark.physics_python
@pytest.mark.slow
def test_python_diskdf_valid_cordbh(tmp_path: Path) -> None:
    """Python solve + diskdf on the MW coarse campaign grid."""
    from galacticsics.campaign.spec import preview_dbh_model
    from galacticsics.physics.python_backend import python_ensure_disk_df

    model = preview_dbh_model(base="milky_way_disk_halo", coarse=True)
    work = tmp_path / "mw_coarse"
    solve_potential(model, work_dir=work, cleanup=False, backend=PhysicsBackendKind.PYTHON)
    python_ensure_disk_df(model, work, diskdf_backend="python")
    cordbh = work / "cordbh.dat"
    assert cordbh_is_valid(cordbh)
    corr = read_disk_correction(cordbh)
    assert corr.f_d.min() > 0
    assert corr.f_sz.min() > 0


@pytest.mark.physics_python
def test_python_sample_moments(python_work_dir: Path, reference_artifacts_dir: Path) -> None:
    import shutil

    model = _small_model()
    if not cordbh_is_valid(python_work_dir / "cordbh.dat"):
        shutil.copy2(reference_artifacts_dir / "cordbh.dat", python_work_dir / "cordbh.dat")
    result = sample_galaxy(
        model,
        SampleConfig(n_disk=32, n_halo=32, run_diskdf=False),
        work_dir=python_work_dir,
        cleanup=False,
        backend=PhysicsBackendKind.PYTHON,
    )
    disk = result.particles["disk"]
    halo = result.particles["halo"]
    assert len(disk) == 32
    assert len(halo) == 32
    assert abs(disk.center_of_mass).max() < 0.5
    assert abs(halo.center_of_mass).max() < 2.0
    sig = disk.velocity_dispersion()
    assert sig[0] == pytest.approx(model.disk_kinematics.sigma_r0, rel=0.8)


@pytest.mark.physics_python
def test_python_bulge_disk_halo_solve(tmp_path: Path) -> None:
    """Disk + bulge + halo solve writes bulge DF tables and finite potential."""
    from galacticsics.models import NFWHalo, SersicBulge

    model = GalaxyModel(
        halo=NFWHalo(r_outer=40.0, v0=2.0, a=6.0, dr_trunc=6.0, enabled=True),
        disk=GalaxyModel.reference_disk_halo().disk,
        bulge=SersicBulge(n_sersic=4.0, ppp=0.5, v0=1.5, a=0.4, enabled=True),
        grid=PotentialGrid(dr=0.2, nr=40, lmax=2),
    )
    work = tmp_path / "bulge_solve"
    solve_potential(model, work_dir=work, cleanup=False, backend=PhysicsBackendKind.PYTHON)
    assert (work / "dfsersic.dat").is_file()
    assert (work / "denspsibulge.dat").is_file()
    pot = read_harmonic_potential(work / "dbh.dat")
    assert pot.flags.bulge
    assert math.isfinite(evaluate_potential(pot, 0.5, 0.0))


@pytest.mark.physics_python
def test_vectorized_halo_density_matches_scalar() -> None:
    """Array halo density evaluation matches the scalar API."""
    from galacticsics.potential.poisson.densities import halo_density_spherical, halo_density_spherical_array

    model = GalaxyModel.reference_disk_halo()
    assert model.halo is not None
    radii = np.array([0.0, 0.5, 2.0, 8.0, 40.0])
    rho_arr = halo_density_spherical_array(radii, model.halo)
    rho_scalar = np.array([halo_density_spherical(float(r), model.halo) for r in radii])
    np.testing.assert_allclose(rho_arr, rho_scalar, rtol=0.0, atol=0.0)


@pytest.mark.physics_python
def test_disk_vertical_sech2_large_height_no_overflow() -> None:
    """sech² disk factor stays finite at heights far above the scale length."""
    from galacticsics.potential.poisson.densities import disk_density_estimate, disk_vertical_sech2

    zd = 0.25
    for z in (100.0, 400.0, 2000.0):
        g2 = disk_vertical_sech2(z, zd)
        assert math.isfinite(g2)
        assert g2 == 0.0
    model = GalaxyModel.reference_disk_halo()
    rho = disk_density_estimate(50.0, 400.0, model)
    assert math.isfinite(rho)
    assert rho >= 0.0


@pytest.mark.physics_python
def test_python_sample_halo_large_count(tmp_path: Path) -> None:
    """Halo velocity draws use vmax=sqrt(vmax2), matching legacy genhalo."""
    from galacticsics.sampling.python.samplers import sample_halo_python

    model = GalaxyModel.reference_disk_halo()
    work = tmp_path / "halo_sample"
    solve_potential(model, work_dir=work, cleanup=False, backend=PhysicsBackendKind.PYTHON)
    ps = sample_halo_python(work, n_particles=500, seed=-42, center=True)
    assert len(ps) == 500
    assert ps.total_mass > 0.0


@pytest.mark.physics_python
def test_python_sample_bulge(tmp_path: Path) -> None:
    """Bulge rejection sampling produces the requested particle count."""
    from galacticsics.models import NFWHalo, SersicBulge

    model = GalaxyModel(
        halo=NFWHalo(r_outer=40.0, v0=2.0, a=6.0, dr_trunc=6.0, enabled=True),
        bulge=SersicBulge(n_sersic=4.0, ppp=0.5, v0=1.5, a=0.4, enabled=True),
        grid=PotentialGrid(dr=0.2, nr=40, lmax=0),
    )
    work = tmp_path / "bulge_sample"
    solve_potential(model, work_dir=work, cleanup=False, backend=PhysicsBackendKind.PYTHON)
    result = sample_galaxy(
        model,
        SampleConfig(n_disk=0, n_halo=0, n_bulge=24, run_diskdf=False),
        work_dir=work,
        cleanup=False,
        backend=PhysicsBackendKind.PYTHON,
    )
    bulge = result.particles["bulge"]
    assert len(bulge) == 24
    assert abs(bulge.center_of_mass).max() < 1.0
