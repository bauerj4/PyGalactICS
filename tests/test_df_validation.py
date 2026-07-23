"""Tests for IC distribution-function validation diagnostics."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from galacticsics.campaign.analysis import summarize_evolution_health
from galacticsics.diagnostics.df_validation import (
    validate_bulge_df,
    validate_disk_df,
    validate_halo_df,
    validate_ic_distribution_functions,
)
from galacticsics.models import GalaxyModel, NFWHalo, PotentialGrid, SersicBulge
from galacticsics.physics.backend import PhysicsBackendKind
from galacticsics.potential.solver import solve_potential
from galacticsics.sampling.sampler import SampleConfig, sample_galaxy
from ntropy.integrations.galacticsics import merge_galacticsics_components
from ntropy.particle_types import TypeRegistry
from ntropy.particles import ParticleState
from tests.fixtures.milkyway import reference_artifacts_dir


def _small_model() -> GalaxyModel:
    model = GalaxyModel.reference_disk_halo()
    return replace(model, grid=PotentialGrid(dr=0.15, nr=60, lmax=2))


@pytest.fixture(scope="module")
def sampled_work_dir(tmp_path_factory) -> Path:
    work = tmp_path_factory.mktemp("df_validation_sample")
    model = _small_model()
    solve_potential(model, work_dir=work, cleanup=False, backend=PhysicsBackendKind.PYTHON)
    ref = reference_artifacts_dir()
    if (ref / "cordbh.dat").is_file():
        shutil.copy2(ref / "cordbh.dat", work / "cordbh.dat")
    else:
        from galacticsics.distribution.diskdf_solve import solve_diskdf_python

        solve_diskdf_python(model, work, n_iterations=1, n_radial_steps=12)
    sample_galaxy(
        model,
        SampleConfig(n_disk=96, n_halo=96, run_diskdf=False),
        work_dir=work,
        cleanup=False,
        backend=PhysicsBackendKind.PYTHON,
    )
    return work


@pytest.fixture(scope="module")
def sampled_state(sampled_work_dir: Path) -> ParticleState:
    from galacticsics.campaign.runner import _load_ic_particles

    registry = TypeRegistry.default_galaxy()
    particles = _load_ic_particles(sampled_work_dir)
    return merge_galacticsics_components(particles, type_registry=registry)


@pytest.fixture(scope="module")
def sampled_model(sampled_work_dir: Path) -> GalaxyModel:
    from galacticsics.io import read_harmonic_potential

    return read_harmonic_potential(sampled_work_dir / "dbh.dat").model


@pytest.mark.physics_python
def test_validate_halo_df_passes(sampled_state: ParticleState, sampled_model: GalaxyModel, sampled_work_dir: Path):
    report = validate_halo_df(sampled_state, sampled_model, sampled_work_dir)
    assert report["n"] == 96
    assert report["pass"]
    assert report["speed_in_bounds_frac"] >= 0.99
    assert report["energy_df_max_log_rel"] < 0.6


@pytest.mark.physics_python
def test_validate_disk_df_passes(sampled_state: ParticleState, sampled_model: GalaxyModel, sampled_work_dir: Path):
    report = validate_disk_df(sampled_state, sampled_model, sampled_work_dir)
    assert report["n"] == 96
    assert report["pass"]
    assert report["cordbh_valid"]
    assert report["df_positive_frac"] >= 0.99
    assert report["surface_density_max_rel"] <= report["thresholds"]["surface_density_max_rel"]


@pytest.mark.physics_python
def test_validate_ic_distribution_functions(sampled_state, sampled_model, sampled_work_dir):
    report = validate_ic_distribution_functions(sampled_state, sampled_model, sampled_work_dir)
    assert report["overall_pass"]
    assert report["halo"]["pass"]
    assert report["disk"]["pass"]
    assert "virial" not in report

    with_virial = validate_ic_distribution_functions(
        sampled_state, sampled_model, sampled_work_dir, include_virial=True
    )
    assert "virial" in with_virial
    assert "virial_pass" in with_virial


@pytest.mark.physics_python
def test_summarize_evolution_health_includes_df_and_virial(sampled_state, sampled_model, sampled_work_dir):
    health = summarize_evolution_health(
        sampled_state,
        sampled_state,
        model=sampled_model,
        work_dir=sampled_work_dir,
    )
    assert "virial_ic_virial_ratio" in health
    assert "df_validation_pass" in health
    assert "df_halo_pass" in health
    assert "df_disk_pass" in health


@pytest.mark.physics_python
def test_validate_bulge_df_when_present(tmp_path: Path):
    model = GalaxyModel(
        halo=NFWHalo(r_outer=40.0, v0=2.0, a=6.0, dr_trunc=6.0, enabled=True),
        bulge=SersicBulge(n_sersic=4.0, ppp=0.5, v0=1.5, a=0.4, enabled=True),
        grid=PotentialGrid(dr=0.2, nr=40, lmax=0),
    )
    work = tmp_path / "bulge_df"
    solve_potential(model, work_dir=work, cleanup=False, backend=PhysicsBackendKind.PYTHON)
    result = sample_galaxy(
        model,
        SampleConfig(n_disk=0, n_halo=0, n_bulge=48, run_diskdf=False),
        work_dir=work,
        cleanup=False,
        backend=PhysicsBackendKind.PYTHON,
    )
    registry = TypeRegistry.default_galaxy()
    state = merge_galacticsics_components(result.particles, type_registry=registry)
    report = validate_bulge_df(state, model, work)
    assert report["n"] == 48
    assert report["pass"]
