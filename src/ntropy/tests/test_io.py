"""Tests for particle I/O."""

from __future__ import annotations

import numpy as np

from ntropy.io.particles import PARTICLE_DTYPE, read_particles_ascii, write_particles_ascii
from ntropy.particles import ParticleState


def test_roundtrip_ascii(tmp_path):
    data = np.zeros(3, dtype=PARTICLE_DTYPE)
    data["mass"] = [1.0, 1.0, 1.0]
    data["x"] = [0.0, 1.0, 0.0]
    data["y"] = [0.0, 0.0, 1.0]
    data["z"] = [0.0, 0.0, 0.0]
    path = tmp_path / "parts.dat"
    write_particles_ascii(path, data)
    loaded = read_particles_ascii(path)
    np.testing.assert_allclose(loaded["mass"], data["mass"])
    np.testing.assert_allclose(loaded["x"], data["x"])


def test_skip_header_line(tmp_path):
    path = tmp_path / "parts.dat"
    path.write_text("3 0\n1.0 0 0 0 0 0 0\n1.0 1 0 0 0 0 0\n")
    data = read_particles_ascii(path)
    assert len(data) == 2


def test_particle_state_write(tmp_path):
    state = ParticleState.from_arrays(
        pos=np.array([[0.0, 0.0, 0.0]]),
        vel=np.array([[0.0, 0.0, 0.0]]),
        mass=np.array([1.0]),
        eps=0.01,
    )
    path = tmp_path / "out.dat"
    state.write_ascii(path)
    loaded = read_particles_ascii(path)
    assert len(loaded) == 1
