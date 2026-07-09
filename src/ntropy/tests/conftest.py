"""Shared pytest fixtures for ntropy."""

from __future__ import annotations

import numpy as np
import pytest

from ntropy.particles import ParticleState


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_plummer_state(rng):
    """Small random Plummer-like particle set for force tests."""
    n = 32
    pos = rng.normal(size=(n, 3))
    vel = rng.normal(scale=0.1, size=(n, 3))
    mass = np.ones(n)
    eps = np.full(n, 0.05)
    return ParticleState.from_arrays(pos, vel, mass, eps)
