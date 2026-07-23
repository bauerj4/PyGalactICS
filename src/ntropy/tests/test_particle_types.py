"""Tests for particle type registry and typed ParticleState."""

from __future__ import annotations

import numpy as np
import pytest

from ntropy.ics.composite import CompositeICSpec, sample_composite
from ntropy.ics.nfw import NFWParams
from ntropy.ics.disk import ExponentialDiskParams
from ntropy.particle_types import ParticleTypeSpec, TypeRegistry
from ntropy.particles import ParticleState


def test_type_registry_default_galaxy():
    reg = TypeRegistry.default_galaxy()
    assert reg.id_for("halo") == 1
    assert reg.id_for("disk") == 3
    halo = reg.types["halo"]
    assert halo.max_timestep_bin == 5
    assert reg.types["disk"].max_timestep_bin == 3


def test_type_registry_duplicate_id_raises():
    reg = TypeRegistry()
    reg.register(ParticleTypeSpec(id=1, label="a"))
    with pytest.raises(ValueError, match="duplicate type id"):
        reg.register(ParticleTypeSpec(id=1, label="b"))


def test_particle_state_mask_type():
    reg = TypeRegistry.default_galaxy()
    state = ParticleState.from_arrays(
        pos=np.zeros((4, 3)),
        vel=np.zeros((4, 3)),
        mass=np.ones(4),
        eps=np.full(4, 0.05),
        type_id=np.array([1, 1, 3, 3], dtype=np.int32),
    )
    disk = state.mask_type(reg.id_for("disk"))
    assert disk.n == 2


def test_sample_composite_sets_type_ids():
    spec = CompositeICSpec(
        halo=NFWParams(n_particles=16, mass=50.0, eps=0.05),
        disk=ExponentialDiskParams(n_particles=16, mass=10.0, eps=0.02),
    )
    state = sample_composite(spec, seed=42, components=frozenset({"halo", "disk"}))
    assert state.type_id is not None
    reg = TypeRegistry.default_galaxy()
    halo_n = state.mask_type(reg.id_for("halo")).n
    disk_n = state.mask_type(reg.id_for("disk")).n
    assert halo_n == 16
    assert disk_n == 16


def test_particle_io_type_column_roundtrip(tmp_path):
    state = ParticleState.from_arrays(
        pos=np.random.randn(5, 3),
        vel=np.random.randn(5, 3),
        mass=np.ones(5),
        eps=np.full(5, 0.01),
        type_id=np.array([1, 1, 2, 3, 3], dtype=np.int32),
    )
    path = tmp_path / "p.dat"
    state.write_ascii(path)
    from ntropy.io.particles import read_particles_ascii

    data = read_particles_ascii(path)
    assert "type_id" in data.dtype.names
    assert list(data["type_id"]) == [1, 1, 2, 3, 3]
