"""End-to-end galacticsics IC generation → ntropy evolution tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ntropy.analysis.density import bin_spherical_density, compare_density_profiles
from ntropy.config import ForceConfig, IntegratorConfig, ParallelConfig, RunConfig
from ntropy.integrations.galacticsics import (
    galacticsics_available,
    nfw_halo_model_fast,
    sample_galacticsics_galaxy,
    sample_galacticsics_halo,
)
from ntropy.simulation import Simulation

pytestmark = pytest.mark.skipif(
    not galacticsics_available(),
    reason="galacticsics and legacy dbh/genhalo binaries required (make install-dev)",
)


@pytest.fixture(scope="session")
def reference_artifacts_dir() -> Path:
    """Reuse generated reference artifacts from the galacticsics test suite."""
    from galacticsics.artifacts.paths import default_artifact_dir
    from galacticsics.artifacts.generate import generate_reference_artifacts
    from galacticsics.legacy.paths import require_binary

    require_binary("gendisk")
    out = default_artifact_dir()
    if not (out / "manifest.json").is_file():
        generate_reference_artifacts(out, verify=True)
    return out


def test_particle_state_from_galacticsics_halo(tmp_path):
    """dbh + genhalo produces particles that ntropy can evolve."""
    result = sample_galacticsics_halo(
        nfw_halo_model_fast(),
        n_particles=64,
        seed=-42,
        work_dir=tmp_path / "halo_ic",
        eps=0.04,
    )
    assert result.state.n == 64
    assert result.components["halo"] == 64
    assert result.state.tags is not None
    assert np.all(result.state.tags == "halo")
    assert result.state.mass.sum() > 0


def test_galacticsics_halo_density_stable_under_ntropy(tmp_path):
    """GalactICS NFW halo IC retains spherical density over a short ntropy run."""
    result = sample_galacticsics_halo(
        n_particles=96,
        seed=-42,
        work_dir=tmp_path / "halo_run",
        eps=0.04,
    )
    state = result.state
    r_max = float(np.sqrt(np.sum(state.pos**2, axis=1)).max()) * 1.2
    init_prof = bin_spherical_density(state.pos, state.mass, n_bins=12, r_max=r_max)

    cfg = RunConfig()
    cfg.integrator = IntegratorConfig(dt=0.002, n_steps=20)
    cfg.force = ForceConfig(method="brute")
    cfg.parallel = ParallelConfig(enabled=False)
    cfg.output.write_final = False
    cfg.output.every = 0

    sim_result = Simulation(cfg, state=state.copy()).run()
    final_prof = bin_spherical_density(
        sim_result.final_state.pos,
        sim_result.final_state.mass,
        n_bins=12,
        r_max=r_max,
    )
    max_drift = compare_density_profiles(init_prof, final_prof, min_count=2)
    assert max_drift < 0.35


def test_galacticsics_reference_disk_halo_through_ntropy(reference_artifacts_dir, tmp_path):
    """Sample disk+halo from reference artifacts and evolve in ntropy."""
    from galacticsics.artifacts.paths import reference_model
    from galacticsics.sampling.sampler import SampleConfig

    config = SampleConfig(
        n_disk=80,
        n_halo=80,
        n_bulge=0,
        seed_disk=-11,
        seed_halo=-22,
        run_diskdf=False,
    )
    result = sample_galacticsics_galaxy(
        reference_model(),
        config,
        work_dir=tmp_path / "ref_sample",
        artifact_dir=reference_artifacts_dir,
        eps_by_component={"disk": 0.02, "halo": 0.05},
        solve=False,
    )
    assert result.components == {"disk": 80, "halo": 80}
    assert result.state.tags is not None
    assert set(np.unique(result.state.tags)) == {"disk", "halo"}

    cfg = RunConfig()
    cfg.integrator = IntegratorConfig(dt=0.001, n_steps=15)
    cfg.force = ForceConfig(method="brute")
    cfg.parallel = ParallelConfig(enabled=False)
    cfg.output.write_final = False

    sim_result = Simulation(cfg, state=result.state.copy()).run()
    assert len(sim_result.energies) == 16
