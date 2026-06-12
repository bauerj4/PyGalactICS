"""GalaxyBuilder tests with generated reference artifacts."""

from __future__ import annotations

from galacticsics.builder import GalaxyBuilder
from galacticsics.models import GalaxyModel
from galacticsics.sampling.particles import ParticleSet


def test_load_reference_artifacts(reference_artifacts_dir, reference_model):
    builder = GalaxyBuilder(
        model=reference_model,
        model_dir=str(reference_artifacts_dir),
    ).load_artifacts()
    assert builder.potential is not None
    assert builder.disk_correction is not None
    assert builder.frequencies is not None
    assert builder.halo_potential is not None


def test_load_reference_particles(reference_artifacts_dir):
    disk = ParticleSet.from_ascii(
        reference_artifacts_dir / "disk", component="disk", max_particles=100
    )
    assert len(disk) == 100
    assert disk.total_mass > 0
