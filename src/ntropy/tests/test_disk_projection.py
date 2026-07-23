"""Tests for 2D disk projection binning."""

from __future__ import annotations

import numpy as np

from ntropy.analysis.disk_density import bin_plane_density


def test_bin_plane_density_face_on():
    n = 200
    rng = np.random.default_rng(0)
    pos = np.column_stack([rng.normal(0, 2, n), rng.normal(0, 2, n), rng.normal(0, 0.1, n)])
    mass = np.ones(n)
    m = bin_plane_density(pos, mass, axes=(0, 1), n_bins=16, half_extent=10.0)
    assert m.density.shape == (16, 16)
    assert m.counts.sum() == n


def test_bin_plane_density_z_filter():
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    mass = np.ones(2)
    z_filter = np.abs(pos[:, 2]) < 1.0
    m = bin_plane_density(pos, mass, axes=(0, 1), n_bins=4, half_extent=2.0, z_filter=z_filter)
    assert m.counts.sum() == 1
