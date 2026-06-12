"""Tests for brute-force and Barnes-Hut force agreement."""

from __future__ import annotations

import numpy as np

from ntropy.forces.brute import compute_forces_brute
from ntropy.forces.bhtree import compute_forces_bh


def test_brute_bh_agreement(small_plummer_state):
    state = small_plummer_state
    acc_brute = compute_forces_brute(state.pos, state.mass, state.eps)
    acc_bh = compute_forces_bh(
        state.pos, state.mass, state.eps, theta=0.3
    )
    np.testing.assert_allclose(acc_brute, acc_bh, rtol=0.15, atol=1e-3)


def test_variable_softening_symmetry():
    pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    mass = np.array([1.0, 1.0])
    eps = np.array([0.1, 0.3])
    acc = compute_forces_brute(pos, mass, eps)
    np.testing.assert_allclose(acc[0, 0], -acc[1, 0], rtol=1e-10)
