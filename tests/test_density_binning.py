"""Tests for log-spherical density binning."""

from __future__ import annotations

import numpy as np

from ntropy.analysis.density import bin_spherical_density


def test_bin_spherical_density_log_bins_monotonic_edges():
    pos = np.array([[1.0, 0.0, 0.0], [10.0, 0.0, 0.0], [30.0, 0.0, 0.0]])
    mass = np.ones(3)
    prof = bin_spherical_density(pos, mass, n_bins=8, r_max=40.0, log_bins=True)
    assert np.all(np.diff(prof.r_mid) > 0)
    assert prof.counts.sum() == 3


def test_bin_spherical_density_linear_bins():
    pos = np.array([[1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    mass = np.ones(2)
    prof = bin_spherical_density(pos, mass, n_bins=4, r_max=10.0, log_bins=False)
    assert len(prof.rho) == 4
