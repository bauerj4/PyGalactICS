"""Tests for Plummer softening kernel."""

from __future__ import annotations

import numpy as np

from ntropy.softening import pairwise_softening, softened_acceleration_vectorized


def test_pairwise_symmetry():
    eps = np.array([0.1, 0.2, 0.3])
    h = pairwise_softening(eps, eps)
    np.testing.assert_allclose(h[0, 1], h[1, 0])
    np.testing.assert_allclose(h[0, 1], 0.15)


def test_acceleration_at_large_distance():
    pos = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    mass = np.array([1.0, 1.0])
    eps = np.array([0.01, 0.01])
    acc = softened_acceleration_vectorized(pos, mass, eps)
    r2 = 100.0
    h2 = 0.01**2
    separation = 10.0
    expected = separation / (r2 + h2) ** 1.5
    np.testing.assert_allclose(acc[0, 0], expected, rtol=1e-3)
