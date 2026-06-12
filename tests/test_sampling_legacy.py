"""Integration tests calling legacy/bin samplers."""

from __future__ import annotations

import pytest

from galacticsics.legacy.paths import require_binary
from galacticsics.sampling import SampleConfig, sample_galaxy
from galacticsics.models import GalaxyModel


@pytest.fixture(scope="session")
def legacy_samplers_available():
    try:
        require_binary("gendisk")
        require_binary("genhalo")
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.mark.legacy_binary
def test_sample_reference_small(reference_artifacts_dir, reference_model, legacy_samplers_available, tmp_path):
    """Sample a small disk+halo subset from generated reference artifacts."""
    config = SampleConfig(
        n_disk=200,
        n_halo=200,
        n_bulge=0,
        seed_disk=-42,
        seed_halo=-42,
        run_diskdf=False,
    )
    result = sample_galaxy(
        reference_model,
        config,
        work_dir=tmp_path / "sample",
        artifact_dir=reference_artifacts_dir,
        cleanup=False,
    )
    assert "disk" in result.particles
    assert "halo" in result.particles
    assert len(result.particles["disk"]) == 200
    assert len(result.particles["halo"]) == 200
    assert result.particles["disk"].total_mass > 0
