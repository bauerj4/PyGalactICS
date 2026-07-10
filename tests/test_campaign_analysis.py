"""Tests for campaign density-map and halo-profile analysis helpers."""

from __future__ import annotations

import numpy as np
import pytest

from galacticsics.campaign.analysis import (
    HALO_PROFILE_MIN_COUNT,
    add_density_colorbar,
    density_map_color_limits,
    density_map_log10,
    halo_profile_bin_edges,
    halo_projections,
    halo_spherical_profile,
    plot_density_map,
    plot_halo_spherical_profile,
    profile_plot_series,
)
from ntropy.analysis.disk_density import bin_plane_density
from ntropy.particles import ParticleState


def _state_with_tags(*, n: int = 200, component: str = "halo") -> ParticleState:
    pos = np.column_stack(
        [
            np.linspace(2.0, 15.0, n),
            np.zeros(n),
            np.zeros(n),
        ]
    )
    mass = np.full(n, 1.0 / n)
    eps = np.full(n, 0.1)
    tags = np.array([component] * n)
    return ParticleState.from_arrays(pos, np.zeros((n, 3)), mass, eps, tags=tags)


def test_halo_profile_bin_edges_hybrid_monotonic():
    edges = halo_profile_bin_edges(n_bins=24, r_min=1.0, r_max=40.0, r_log_max=10.0)
    assert len(edges) == 25
    assert edges[0] == pytest.approx(1.0)
    assert edges[-1] == pytest.approx(40.0)
    assert np.all(np.diff(edges) > 0)


def test_halo_spherical_profile_shared_edges_match():
    state_a = _state_with_tags()
    state_b = _state_with_tags(n=180)
    edges = halo_profile_bin_edges()
    prof_a = halo_spherical_profile(state_a, bin_edges=edges)
    prof_b = halo_spherical_profile(state_b, bin_edges=edges)
    np.testing.assert_allclose(prof_a.r_mid, prof_b.r_mid)
    assert len(prof_a.r_mid) == len(edges) - 1


def test_halo_spherical_profile_respects_r_min():
    state = _state_with_tags()
    prof = halo_spherical_profile(state, r_max=20.0, r_min=2.0, n_bins=8, log_bins=False)
    assert prof.r_mid[0] >= 1.0


def test_profile_plot_series_masks_sparse_bins():
    state = _state_with_tags(n=30)
    edges = halo_profile_bin_edges(n_bins=12)
    prof = halo_spherical_profile(state, bin_edges=edges)
    r_mid, rho, valid = profile_plot_series(prof, min_count=HALO_PROFILE_MIN_COUNT)
    assert np.all(np.isnan(r_mid[~valid]))
    assert np.all(np.isnan(rho[~valid]))
    assert np.isfinite(rho[valid]).all()


def test_halo_projections_halo_component_only():
    n = 64
    pos = np.random.default_rng(0).normal(size=(n, 3))
    mass = np.full(n, 1.0 / n)
    eps = np.full(n, 0.1)
    tags = np.array(["halo"] * (n // 2) + ["disk"] * (n // 2))
    state = ParticleState.from_arrays(pos, np.zeros((n, 3)), mass, eps, tags=tags)
    face, edge = halo_projections(state, half_extent=5.0, n_bins=8)
    assert face.density.shape == (8, 8)
    assert edge.density.shape == (8, 8)


def test_plot_halo_spherical_profile_runs():
    import matplotlib.pyplot as plt

    state_i = _state_with_tags(n=400)
    state_f = _state_with_tags(n=360)
    edges = halo_profile_bin_edges()
    prof_i = halo_spherical_profile(state_i, bin_edges=edges)
    prof_f = halo_spherical_profile(state_f, bin_edges=edges)
    fig, ax = plt.subplots()
    plot_halo_spherical_profile(ax, prof_i, prof_f, min_count=5, show_counts=True)
    plt.close(fig)


def test_density_map_color_limits_ignores_empty_bins():
    map_a = bin_plane_density(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([1.0, 1.0]),
        n_bins=4,
        half_extent=2.0,
    )
    data = density_map_log10(map_a)
    assert np.isfinite(data).any()
    vmin, vmax = density_map_color_limits(map_a)
    assert vmax > vmin
    assert vmin > -20.0


def test_density_map_shared_limits_across_maps():
    pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    mass = np.array([1.0, 1.0, 2.0])
    map_lo = bin_plane_density(pos, mass, n_bins=8, half_extent=5.0)
    map_hi = bin_plane_density(pos, mass * 3.0, n_bins=8, half_extent=5.0)
    vmin, vmax = density_map_color_limits(map_lo, map_hi)
    assert vmax > vmin

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
    im0 = plot_density_map(axes[0], map_lo, title="lo", vmin=vmin, vmax=vmax)
    im1 = plot_density_map(axes[1], map_hi, title="hi", vmin=vmin, vmax=vmax)
    add_density_colorbar(fig, im1, axes, label="log10")
    plt.close(fig)
