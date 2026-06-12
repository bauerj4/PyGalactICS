"""Tests for exponential disk target density (legacy prescription)."""

from __future__ import annotations

import numpy as np

from ntropy.analysis.disk_density import (
    bin_midplane_surface_density,
    compare_surface_density,
    target_surface_density,
)
from ntropy.ics.disk import ExponentialDiskParams, exponential_disk_density, sample_exponential_disk


def test_exponential_disk_volume_density_matches_legacy():
    """ρ(R, z) uses Σ(R) × sech²(z/z_d) / (2 z_d)."""
    params = ExponentialDiskParams(mass=10.0, scale_length=2.0, scale_height=0.2)
    r = np.array([1.0, 5.0, 10.0])
    z = np.array([0.0, 0.1, -0.1])
    rho = exponential_disk_density(r, z, params)
    sigma = target_surface_density(r, params)
    vertical = 0.5 / params.scale_height / np.cosh(z / params.scale_height) ** 2
    np.testing.assert_allclose(rho, sigma * vertical, rtol=1e-10)


def test_sampled_disk_matches_target_sigma():
    """Sampled midplane particles reproduce legacy Σ(R) within tolerance."""
    params = ExponentialDiskParams(n_particles=800, mass=20.0)
    state = sample_exponential_disk(params, seed=42)
    measured = bin_midplane_surface_density(
        state.pos, state.mass, n_bins=12, r_max=params.outer_radius * 0.8
    )
    max_rel = compare_surface_density(measured, params, min_count=8)
    assert max_rel < 0.35
