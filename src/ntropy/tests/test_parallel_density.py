"""Per-component density evolution: serial vs parallel force paths."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ntropy.analysis.density import DensityProfile, bin_spherical_density, compare_density_profiles
from ntropy.analysis.disk_density import SurfaceDensityProfile, bin_midplane_surface_density
from ntropy.config import ForceConfig, IntegratorConfig, ParallelConfig, RunConfig
from ntropy.ics.composite import CompositeICSpec, sample_composite
from ntropy.ics.disk import ExponentialDiskParams
from ntropy.ics.nfw import NFWParams
from ntropy.ics.sersic import SersicParams
from ntropy.parallel.mpi import mpi_available
from ntropy.particles import ParticleState
from ntropy.simulation import Simulation

PARALLEL_SPEC = CompositeICSpec(
    halo=NFWParams(n_particles=96, mass=60.0, a=8.0, r_trunc=50.0, eps=0.05),
    bulge=SersicParams(n_particles=48, mass=6.0, r_e=0.4, eps=0.02),
    disk=ExponentialDiskParams(
        n_particles=256,
        mass=12.0,
        scale_length=2.5,
        outer_radius=20.0,
        scale_height=0.25,
        trunc_width=1.5,
        eps=0.02,
    ),
)

ALL_COMPONENTS = frozenset({"halo", "bulge", "disk"})
COMPONENT_IDS = ("halo", "bulge", "disk")

_MPI_WORKER = Path(__file__).with_name("mpi_density_worker.py")


def _run_cfg(*, parallel: bool) -> RunConfig:
    cfg = RunConfig()
    cfg.integrator = IntegratorConfig(dt=0.002, n_steps=14)
    cfg.force = ForceConfig(method="brute")
    cfg.parallel = ParallelConfig(enabled=parallel, n_workers=2)
    cfg.output.write_final = False
    cfg.output.every = 0
    return cfg


def _run(state: ParticleState, *, parallel: bool) -> Simulation:
    return Simulation(_run_cfg(parallel=parallel), state=state.copy()).run()


def _sigma_to_density_profile(profile: SurfaceDensityProfile) -> DensityProfile:
    return DensityProfile(r_mid=profile.r_mid, rho=profile.sigma, counts=profile.counts)


def _spherical_metrics(
    initial: ParticleState,
    final: ParticleState,
    *,
    n_bins: int = 12,
) -> tuple[float, DensityProfile]:
    r_max = float(np.sqrt(np.sum(initial.pos**2, axis=1)).max()) * 1.2
    init_prof = bin_spherical_density(initial.pos, initial.mass, n_bins=n_bins, r_max=r_max)
    final_prof = bin_spherical_density(final.pos, final.mass, n_bins=n_bins, r_max=r_max)
    drift = compare_density_profiles(init_prof, final_prof, min_count=2)
    return drift, final_prof


def _disk_sigma_metrics(
    initial: ParticleState,
    final: ParticleState,
    *,
    r_max: float,
    n_bins: int = 10,
) -> tuple[float, SurfaceDensityProfile]:
    init_sigma = bin_midplane_surface_density(
        initial.pos, initial.mass, n_bins=n_bins, r_max=r_max
    )
    final_sigma = bin_midplane_surface_density(
        final.pos, final.mass, n_bins=n_bins, r_max=r_max
    )
    drift = compare_density_profiles(
        _sigma_to_density_profile(init_sigma),
        _sigma_to_density_profile(final_sigma),
        min_count=3,
    )
    return drift, final_sigma


def _mpirun_available() -> bool:
    return mpi_available() and shutil.which("mpirun") is not None


@pytest.mark.parametrize("component", COMPONENT_IDS)
def test_component_density_evolution_serial_vs_parallel(component: str):
    """Each tagged component evolves identically under serial and MPI dispatch."""
    state = sample_composite(PARALLEL_SPEC, seed=7, components=ALL_COMPONENTS)
    serial = _run(state, parallel=False)
    parallel = _run(state, parallel=True)

    init_sub = state.mask(component)
    ser_sub = serial.final_state.mask(component)
    par_sub = parallel.final_state.mask(component)

    np.testing.assert_allclose(ser_sub.pos, par_sub.pos, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ser_sub.vel, par_sub.vel, rtol=0, atol=1e-12)

    if component == "disk":
        disk_params = PARALLEL_SPEC.disk
        assert disk_params is not None
        r_max = disk_params.outer_radius * 0.85
        drift_serial, prof_serial = _disk_sigma_metrics(init_sub, ser_sub, r_max=r_max)
        drift_parallel, prof_parallel = _disk_sigma_metrics(init_sub, par_sub, r_max=r_max)
        np.testing.assert_allclose(prof_serial.sigma, prof_parallel.sigma, rtol=0, atol=1e-12)
    else:
        drift_serial, prof_serial = _spherical_metrics(init_sub, ser_sub)
        drift_parallel, prof_parallel = _spherical_metrics(init_sub, par_sub)
        np.testing.assert_allclose(prof_serial.rho, prof_parallel.rho, rtol=0, atol=1e-12)

    assert drift_serial == pytest.approx(drift_parallel, rel=0, abs=1e-12)


@pytest.mark.skipif(not _mpirun_available(), reason="mpirun and mpi4py required")
def test_mpi_multirank_component_density_matches_serial(tmp_path):
    """Multi-rank MPI runs preserve per-component density evolution vs serial."""
    state = sample_composite(PARALLEL_SPEC, seed=11, components=ALL_COMPONENTS)
    serial = _run(state, parallel=False)

    state_path = tmp_path / "initial_state.npz"
    out_path = tmp_path / "mpi_final_state.npz"
    np.savez(
        state_path,
        pos=state.pos,
        vel=state.vel,
        mass=state.mass,
        eps=state.eps,
        tags=state.tags,
    )

    cmd = [
        "mpirun",
        "-n",
        "2",
        sys.executable,
        str(_MPI_WORKER),
        str(state_path),
        str(out_path),
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[3])

    with np.load(out_path, allow_pickle=True) as data:
        mpi_final = ParticleState.from_arrays(
            data["pos"],
            data["vel"],
            data["mass"],
            data["eps"],
        )
        mpi_final.tags = data["tags"]

    disk_params = PARALLEL_SPEC.disk
    assert disk_params is not None
    disk_r_max = disk_params.outer_radius * 0.85

    for component in COMPONENT_IDS:
        init_sub = state.mask(component)
        ser_sub = serial.final_state.mask(component)
        mpi_sub = mpi_final.mask(component)

        np.testing.assert_allclose(ser_sub.pos, mpi_sub.pos, rtol=0, atol=1e-10)
        np.testing.assert_allclose(ser_sub.vel, mpi_sub.vel, rtol=0, atol=1e-10)

        if component == "disk":
            drift_serial, prof_serial = _disk_sigma_metrics(init_sub, ser_sub, r_max=disk_r_max)
            drift_mpi, prof_mpi = _disk_sigma_metrics(init_sub, mpi_sub, r_max=disk_r_max)
            np.testing.assert_allclose(prof_serial.sigma, prof_mpi.sigma, rtol=0, atol=1e-10)
        else:
            drift_serial, prof_serial = _spherical_metrics(init_sub, ser_sub)
            drift_mpi, prof_mpi = _spherical_metrics(init_sub, mpi_sub)
            np.testing.assert_allclose(prof_serial.rho, prof_mpi.rho, rtol=0, atol=1e-10)

        assert drift_serial == pytest.approx(drift_mpi, rel=0, abs=1e-10)
