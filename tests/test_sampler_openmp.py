"""Tests for OpenMP sampler extension."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pytest

from galacticsics.models import GalaxyModel, PotentialGrid
from galacticsics.potential.solver import solve_potential
from galacticsics.sampling.openmp import extension_available, sample_disk_openmp, sample_halo_openmp
from galacticsics.sampling.python.samplers import sample_disk_python, sample_halo_python
from galacticsics.sampling.sampler import SampleConfig


pytestmark = pytest.mark.skipif(not extension_available(), reason="OpenMP sampler not built")


def _small_work_dir(tmp_path: Path) -> Path:
    from galacticsics.distribution.diskdf_solve import solve_diskdf_python

    model = replace(GalaxyModel.reference_disk_halo(), grid=PotentialGrid(dr=0.15, nr=60, lmax=2))
    work = tmp_path / "omp_sampler"
    work.mkdir()
    solve_potential(model, work_dir=work, cleanup=False)
    solve_diskdf_python(model, work, n_iterations=1, n_radial_steps=12)
    return work


def test_openmp_halo_matches_python(tmp_path: Path) -> None:
    work = _small_work_dir(tmp_path)
    cfg = SampleConfig(n_halo=200, use_openmp=False)
    py = sample_halo_python(work, n_particles=200, seed=-7, center=True, config=cfg)
    omp = sample_halo_openmp(work, n_particles=200, seed=-7, center=True)
    assert len(py) == len(omp) == 200
    assert py.total_mass == pytest.approx(omp.total_mass)


def test_openmp_disk_produces_particles(tmp_path: Path) -> None:
    work = _small_work_dir(tmp_path)
    ps = sample_disk_openmp(work, n_particles=100, seed=-3, center=True)
    assert len(ps) == 100
    assert ps.total_mass > 0


@pytest.mark.slow
def test_openmp_disk_faster_than_python(tmp_path: Path) -> None:
    work = _small_work_dir(tmp_path)
    n = 500
    cfg = SampleConfig(use_openmp=False)
    t0 = time.perf_counter()
    sample_disk_python(work, n_particles=n, seed=-1, config=cfg)
    py_dt = time.perf_counter() - t0
    t0 = time.perf_counter()
    sample_disk_openmp(work, n_particles=n, seed=-1)
    omp_dt = time.perf_counter() - t0
    assert omp_dt < py_dt


@pytest.mark.physics_python
@pytest.mark.slow
def test_openmp_disk_velocity_stats_mw_coarse(tmp_path: Path) -> None:
    """OpenMP gendisk must match Python on MW coarse walkthrough model."""
    import numpy as np

    from galacticsics.campaign.benchmarks import ic_looks_stable
    from galacticsics.campaign.spec import _apply_patch, preview_dbh_model
    from galacticsics.physics.python_backend import python_ensure_disk_df
    from galacticsics.potential.solver import solve_potential
    from ntropy.integrations.galacticsics import merge_galacticsics_components

    model = preview_dbh_model(base="milky_way_disk_halo", coarse=True)
    model = _apply_patch(model, {"disk_kinematics.toomre_q_target": 1.5})
    work = tmp_path / "mw_omp"
    solve_potential(model, work_dir=work, cleanup=False, backend="python")
    python_ensure_disk_df(model, work)

    n = 2000
    seed = 42
    cfg = SampleConfig(use_openmp=False)
    py = sample_disk_python(work, n_particles=n, seed=seed, config=cfg)
    omp = sample_disk_openmp(work, n_particles=n, seed=seed, n_threads=1)
    py_speeds = np.sqrt(py.data["vx"] ** 2 + py.data["vy"] ** 2 + py.data["vz"] ** 2)
    omp_speeds = np.sqrt(omp.data["vx"] ** 2 + omp.data["vy"] ** 2 + omp.data["vz"] ** 2)
    assert omp_speeds.max() < 6.0
    assert omp_speeds.max() / np.median(omp_speeds) < 5.0
    assert omp_speeds.max() == pytest.approx(py_speeds.max(), rel=0.35)
    state = merge_galacticsics_components({"disk": omp})
    assert ic_looks_stable(state, disk_v_max=6.0)


@pytest.mark.physics_python
def test_openmp_disk_velocity_stats_match_python(tmp_path: Path) -> None:
    """OpenMP gendisk should not hot-tail after rcirc(am) parity fix."""
    import numpy as np

    work = _small_work_dir(tmp_path)
    n = 400
    seed = -42
    py = sample_disk_python(work, n_particles=n, seed=seed, config=SampleConfig(use_openmp=False))
    omp = sample_disk_openmp(work, n_particles=n, seed=seed)
    py_speeds = np.sqrt(py.data["vx"] ** 2 + py.data["vy"] ** 2 + py.data["vz"] ** 2)
    omp_speeds = np.sqrt(omp.data["vx"] ** 2 + omp.data["vy"] ** 2 + omp.data["vz"] ** 2)
    assert omp_speeds.max() < 6.5
    assert omp_speeds.max() / np.median(omp_speeds) < 5.5
    assert omp_speeds.max() == pytest.approx(py_speeds.max(), rel=0.35)
