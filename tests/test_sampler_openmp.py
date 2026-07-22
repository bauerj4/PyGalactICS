"""Tests for OpenMP sampler extension."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pytest

from galacticsics.models import GalaxyModel, PotentialGrid
from galacticsics.potential.solver import solve_potential
from galacticsics.sampling.openmp import (
    extension_available,
    sample_bulge_openmp,
    sample_disk_openmp,
    sample_halo_openmp,
)
from galacticsics.sampling.python.samplers import (
    sample_bulge_python,
    sample_disk_python,
    sample_halo_python,
)
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


def _bulge_halo_work_dir(tmp_path: Path) -> Path:
    from galacticsics.models import NFWHalo, SersicBulge

    model = GalaxyModel(
        halo=NFWHalo(r_outer=40.0, v0=2.0, a=6.0, dr_trunc=6.0, enabled=True),
        bulge=SersicBulge(n_sersic=4.0, ppp=0.5, v0=1.5, a=0.4, enabled=True),
        grid=PotentialGrid(dr=0.1, nr=80, lmax=0),
    )
    work = tmp_path / "omp_bulge"
    work.mkdir()
    solve_potential(model, work_dir=work, cleanup=False)
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
    assert _mean_v_phi(omp.data) > 0.8
    assert _mean_v_phi(py.data) > 0.8
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


def _mean_v_phi(data) -> float:
    import numpy as np

    x, y = data["x"], data["y"]
    vx, vy = data["vx"], data["vy"]
    r = np.hypot(x, y)
    ok = r > 1e-8
    return float(np.mean((-y[ok] * vx[ok] + x[ok] * vy[ok]) / r[ok]))


def test_cylindrical_to_cartesian_velocity_roundtrip() -> None:
    """Unit check for the (vR, vφ) → (vx, vy) convention used by gendisk."""
    import math

    import numpy as np

    rng = np.random.default_rng(0)
    for _ in range(50):
        phi = float(rng.uniform(0, 2 * math.pi))
        r = float(rng.uniform(0.1, 10.0))
        v_r = float(rng.normal())
        v_phi = float(rng.uniform(0.5, 2.0))
        x = r * math.cos(phi)
        y = r * math.sin(phi)
        cph, sph = x / r, y / r
        vx = v_r * cph - v_phi * sph
        vy = v_r * sph + v_phi * cph
        assert (-y * vx + x * vy) / r == pytest.approx(v_phi, rel=1e-12, abs=1e-12)
        assert (x * vx + y * vy) / r == pytest.approx(v_r, rel=1e-12, abs=1e-12)


@pytest.mark.physics_python
def test_disk_sampler_writes_cartesian_rotation(tmp_path: Path) -> None:
    """Regression: gendisk must convert (vR, vφ) → (vx, vy), not store cylindrical as Cartesian."""
    import numpy as np

    work = _small_work_dir(tmp_path)
    n = 800
    seed = 7
    py = sample_disk_python(work, n_particles=n, seed=seed, config=SampleConfig(use_openmp=False))
    py_vphi = _mean_v_phi(py.data)
    # Pre-fix bug left mean v_φ ≈ 0; coarse toy models still have mild asymmetric drift.
    assert py_vphi > 0.15, f"python mean v_phi={py_vphi}"
    x, y, vx, vy = py.data["x"], py.data["y"], py.data["vx"], py.data["vy"]
    assert ((-y * vx + x * vy) > 0).mean() > 0.55
    if extension_available():
        omp = sample_disk_openmp(work, n_particles=n, seed=seed, n_threads=1)
        omp_vphi = _mean_v_phi(omp.data)
        assert omp_vphi > 0.15, f"openmp mean v_phi={omp_vphi}"


@pytest.mark.physics_python
def test_disk_sampler_radial_jacobian_invu(tmp_path: Path) -> None:
    """Regression: R proposal must use legacy invu (P∝R e^{-R/rd}), not -rd ln u."""
    import numpy as np

    from galacticsics.sampling.python.samplers import _invu

    # u→-1 ⇒ x→0; mid-CDF of x e^{-x} is near x≈1.68
    assert _invu(-1.0) == pytest.approx(0.0, abs=1e-5)
    xs = np.array([_invu(-u) for u in np.linspace(1e-6, 1 - 1e-6, 2000)])
    assert 1.4 < float(np.median(xs)) < 2.0

    work = _small_work_dir(tmp_path)
    py = sample_disk_python(work, n_particles=2000, seed=3, config=SampleConfig(use_openmp=False))
    r = np.hypot(py.data["x"], py.data["y"])
    # Wrong -rd*ln(u) proposal piles up at R_med ≲ 2 for Rd~3; invu gives ~4+.
    assert float(np.median(r)) > 2.5
    if extension_available():
        omp = sample_disk_openmp(work, n_particles=2000, seed=3, n_threads=1)
        r_omp = np.hypot(omp.data["x"], omp.data["y"])
        assert float(np.median(r_omp)) > 2.5
        assert _mean_v_phi(omp.data) > 0.5
        # Toy coarse-grid models are hotter than MW; production MW hits ≳0.95.
        assert ((-omp.data["y"] * omp.data["vx"] + omp.data["x"] * omp.data["vy"]) > 0).mean() > 0.85


@pytest.mark.physics_python
def test_openmp_bulge_matches_python_mass_and_shape(tmp_path: Path) -> None:
    import numpy as np

    work = _bulge_halo_work_dir(tmp_path)
    n = 1500
    seed = -11
    py = sample_bulge_python(
        work, n_particles=n, seed=seed, center=True, config=SampleConfig(use_openmp=False)
    )
    omp = sample_bulge_openmp(work, n_particles=n, seed=seed, center=True)
    assert len(py) == len(omp) == n
    assert py.total_mass == pytest.approx(omp.total_mass, rel=1e-9)

    def half_mass(data):
        r = np.sqrt(data["x"] ** 2 + data["y"] ** 2 + data["z"] ** 2)
        order = np.argsort(r)
        c = np.cumsum(data["mass"][order])
        return float(r[order[np.searchsorted(c, 0.5 * c[-1])]])

    assert half_mass(omp.data) == pytest.approx(half_mass(py.data), rel=0.25)
    R = np.hypot(omp.data["x"], omp.data["y"])
    zrms = float(np.sqrt(np.average(omp.data["z"] ** 2, weights=omp.data["mass"])))
    Rrms = float(np.sqrt(np.average(R * R, weights=omp.data["mass"])))
    assert 0.55 < zrms / Rrms < 0.90


@pytest.mark.physics_python
@pytest.mark.slow
def test_openmp_bulge_faster_than_python(tmp_path: Path) -> None:
    work = _bulge_halo_work_dir(tmp_path)
    n = 2000
    cfg = SampleConfig(use_openmp=False)
    t0 = time.perf_counter()
    sample_bulge_python(work, n_particles=n, seed=-1, config=cfg)
    py_dt = time.perf_counter() - t0
    t0 = time.perf_counter()
    sample_bulge_openmp(work, n_particles=n, seed=-1)
    omp_dt = time.perf_counter() - t0
    assert omp_dt < py_dt
