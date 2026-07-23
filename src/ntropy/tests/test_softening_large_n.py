"""Tests for large-N energy fast path."""

from __future__ import annotations

import numpy as np
import pytest

from ntropy.softening import LARGE_N_ENERGY_THRESHOLD, kinetic_energy, total_energy

pytestmark = pytest.mark.essential


def test_total_energy_large_n_uses_kinetic_only():
    n = LARGE_N_ENERGY_THRESHOLD + 100
    pos = np.random.randn(n, 3)
    vel = np.random.randn(n, 3) * 0.1
    mass = np.ones(n)
    eps = np.full(n, 0.01)
    ke = kinetic_energy(vel, mass)
    e = total_energy(pos, vel, mass, eps)
    assert e == ke


def test_total_energy_small_n_includes_potential():
    n = 32
    pos = np.random.randn(n, 3)
    vel = np.random.randn(n, 3) * 0.1
    mass = np.ones(n)
    eps = np.full(n, 0.01)
    e = total_energy(pos, vel, mass, eps)
    ke = kinetic_energy(vel, mass)
    assert e != ke
    assert e < ke
