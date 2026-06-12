"""Sampling statistics tests for generated reference artifacts."""

from __future__ import annotations

import pytest

from galacticsics.sampling.particles import ParticleSet
from galacticsics.units import velocity_to_kms
from tests.constants import RTOL_SAMPLING


@pytest.mark.legacy_parity
def test_disk_particle_mass(reference_artifacts_dir, reference_model):
    disk = ParticleSet.from_ascii(
        reference_artifacts_dir / "disk", component="disk", max_particles=10000
    )
    header = (reference_artifacts_dir / "disk").read_text().splitlines()[0].split("#")[0].split()
    nobj = int(float(header[0]))
    expected = disk.data["mass"][0] * nobj
    # Monte Carlo sampling + disk truncation can bias total mass by a few percent.
    assert expected == pytest.approx(reference_model.disk.mass, rel=0.05)


@pytest.mark.legacy_parity
def test_halo_particle_com_near_origin(reference_artifacts_dir):
    halo = ParticleSet.from_ascii(
        reference_artifacts_dir / "halo", component="halo", max_particles=50000
    )
    com = halo.center_of_mass
    assert abs(com).max() < 1.0


@pytest.mark.legacy_parity
def test_disk_velocity_dispersion_order_of_magnitude(reference_artifacts_dir, reference_model):
    disk = ParticleSet.from_ascii(
        reference_artifacts_dir / "disk", component="disk", max_particles=20000
    )
    sig = disk.velocity_dispersion()
    expected = reference_model.disk_kinematics.sigma_r0
    assert sig[0] == pytest.approx(expected, rel=0.6)
    assert velocity_to_kms(sig[0]) > 30.0
