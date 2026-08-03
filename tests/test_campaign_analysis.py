"""Tests for campaign density-map and halo-profile analysis helpers."""

from __future__ import annotations

import numpy as np
import pytest

from galacticsics.campaign.analysis import (
    HALO_PROFILE_MIN_COUNT,
    HALO_PROFILE_PLOT_MIN_COUNT,
    add_density_colorbar,
    component_radial_stats,
    conserved_quantity_drift,
    density_map_color_limits,
    density_map_log10,
    disk_axisymmetry_diagnostic,
    evolution_checkpoint_times,
    halo_profile_bin_edges,
    halo_profile_min_count,
    halo_projections,
    halo_spherical_profile,
    load_evolution_checkpoints,
    plot_density_map,
    plot_halo_spherical_profile,
    profile_plot_series,
    projection_colorbar_label,
    state_virial_summary,
    summarize_evolution_health,
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


def test_halo_profile_min_count_plot_vs_drift():
    assert halo_profile_min_count(50_000, purpose="drift") == HALO_PROFILE_MIN_COUNT
    plot_mc = halo_profile_min_count(50_000, purpose="plot")
    assert HALO_PROFILE_PLOT_MIN_COUNT <= plot_mc <= HALO_PROFILE_MIN_COUNT


def test_component_radial_stats_and_health_summary():
    state_i = _state_with_tags(n=200)
    state_f = _state_with_tags(n=200)
    stats = component_radial_stats(state_i, "halo")
    assert stats["n"] == 200.0
    assert stats["r_max"] > stats["r_min"]
    drift = conserved_quantity_drift(state_i, state_f)
    assert "dE_over_E0" in drift
    virial = state_virial_summary(state_i)
    assert "virial_ratio" in virial
    health = summarize_evolution_health(state_i, state_f)
    assert health["halo_rho_drift"] >= 0.0
    assert "virial_ic_virial_ratio" in health
    assert "disk_m2_a0_median_ic" in health


def test_disk_axisymmetry_on_uniform_ring():
    n = 400
    phi = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pos = np.column_stack([8.0 * np.cos(phi), 8.0 * np.sin(phi), np.zeros(n)])
    mass = np.full(n, 1.0 / n)
    eps = np.full(n, 0.1)
    tags = np.array(["disk"] * n)
    state = ParticleState.from_arrays(pos, np.zeros((n, 3)), mass, eps, tags=tags)
    ax = disk_axisymmetry_diagnostic(state, min_count=30)
    assert ax["n_disk"] == n
    assert ax["a_m_over_a0_median"] < 0.1


def test_projection_colorbar_label_is_surface_density():
    assert projection_colorbar_label("halo") == "log₁₀ Σ"
    assert projection_colorbar_label("disk") == "log₁₀ Σ"


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


def test_evolution_checkpoint_times_includes_end():
    times = evolution_checkpoint_times([0.0, 0.05, 0.1], end_time_gyr=0.5)
    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(0.5)
    assert 0.05 in times and 0.1 in times


def test_load_evolution_checkpoints_ic_and_final(tmp_path):
    state_i = _state_with_tags(n=80, component="disk")
    state_f = _state_with_tags(n=80, component="disk")
    state_f.pos[:, 0] += 0.5
    evo = tmp_path / "evolution"
    evo.mkdir(parents=True)
    np.savez(
        tmp_path / "ic_state.npz",
        pos=state_i.pos,
        vel=state_i.vel,
        mass=state_i.mass,
        eps=state_i.eps,
        tags=state_i.tags,
    )
    state_f.write_ascii(evo / "final.dat")
    checkpoints = load_evolution_checkpoints(
        tmp_path, [0.0, 0.5], template=state_i, end_time_gyr=0.5
    )
    assert len(checkpoints) == 2
    assert checkpoints[0][0] == pytest.approx(0.0)
    assert checkpoints[1][0] == pytest.approx(0.5)
    assert checkpoints[1][1].pos[0, 0] == pytest.approx(state_f.pos[0, 0])


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


def test_write_component_projection_pngs(tmp_path):
    from galacticsics.campaign.analysis import (
        present_components,
        write_campaign_projection_pngs,
        write_component_projection_pngs,
    )

    rng = np.random.default_rng(1)
    n_disk, n_bulge, n_halo = 40, 20, 30
    pos = rng.normal(size=(n_disk + n_bulge + n_halo, 3))
    mass = np.full(pos.shape[0], 0.01)
    eps = np.full(pos.shape[0], 0.1)
    tags = np.array(["disk"] * n_disk + ["bulge"] * n_bulge + ["halo"] * n_halo)
    state = ParticleState.from_arrays(pos, np.zeros_like(pos), mass, eps, tags=tags)
    assert present_components(state) == ["disk", "bulge", "halo"]

    out = tmp_path / "proj"
    paths = write_component_projection_pngs(state, out, dpi=72)
    names = sorted(p.name for p in paths)
    assert names == [
        "bulge_face_on.png",
        "bulge_side_on.png",
        "disk_face_on.png",
        "disk_side_on.png",
        "halo_face_on.png",
        "halo_side_on.png",
    ]
    assert all(p.is_file() and p.stat().st_size > 0 for p in paths)

    model = tmp_path / "model_a"
    model.mkdir()
    np.savez_compressed(
        model / "ic_state.npz",
        pos=state.pos,
        vel=state.vel,
        mass=state.mass,
        eps=state.eps,
        tags=state.tags,
    )
    campaign_paths = write_campaign_projection_pngs(
        tmp_path, snapshots=["ic"], components=["disk"]
    )
    assert len(campaign_paths) == 2
    assert (model / "projections" / "ic" / "disk_face_on.png").is_file()

    particles = model / "evolution" / "particles"
    particles.mkdir(parents=True)
    # Minimal dump (halo-only tags already on state)
    np.savez_compressed(
        particles / "step_000100.npz",
        pos=state.pos.astype(np.float32),
        vel=state.vel.astype(np.float32),
        mass=state.mass.astype(np.float32),
        eps=state.eps.astype(np.float32),
        timestep_bin=np.zeros(state.n, dtype=np.int32),
        type_id=np.zeros(state.n, dtype=np.int32),
    )
    step_paths = write_campaign_projection_pngs(
        tmp_path, snapshots=["steps"], components=["halo"], model_dirs=[model]
    )
    assert len(step_paths) == 2
    assert (model / "projections" / "step_000100" / "halo_side_on.png").is_file()


def test_halo_side_on_has_no_midplane_cut_by_default():
    """Halo side-on must not apply a thin |z| slice (that looks disk-like)."""
    from galacticsics.campaign.analysis import component_projections

    rng = np.random.default_rng(2)
    n = 200
    pos = rng.normal(scale=3.0, size=(n, 3))
    pos[:40, 2] = rng.uniform(5.0, 8.0, size=40)  # high-|z| particles
    mass = np.full(n, 0.01)
    tags = np.array(["halo"] * n)
    state = ParticleState.from_arrays(pos, np.zeros_like(pos), mass, np.full(n, 0.1), tags=tags)
    _, side_default = component_projections(state, "halo", half_extent=10.0, n_bins=32)
    _, side_sliced = component_projections(
        state, "halo", half_extent=10.0, n_bins=32, z_slice=0.3
    )
    assert side_default.density.sum() > side_sliced.density.sum()
