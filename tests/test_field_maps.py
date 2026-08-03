"""Tests for vertical-slice / voxel field maps and particle resample."""

from __future__ import annotations

import numpy as np
import pytest

from galacticsics.ml.fields.binning import (
    MOMENT_KEYS,
    MultiScaleSliceConfig,
    MultiScaleVoxelConfig,
    SliceMapConfig,
    VoxelMapConfig,
    bin_multiscale_slice_stacks,
    bin_multiscale_voxel_stacks,
    bin_vertical_slice_stack,
    bin_voxel_stack,
    dens_channel_indices,
    dens_channel_indices_component,
    moment_channel_indices_component,
    resolve_moment_keys,
)
from galacticsics.ml.fields.normalize import (
    denormalize_stack,
    estimate_norm_stats,
    estimate_norm_stats_component,
    normalize_stack,
)
from galacticsics.ml.fields.resample import (
    resample_particles_from_multiscale,
    resample_particles_from_slice_stack,
)


def _toy_galaxy(n_disk=800, n_halo=400, n_bulge=200, seed=0):
    rng = np.random.default_rng(seed)
    R = rng.exponential(3.0, n_disk).clip(0.2, 12.0)
    phi = rng.uniform(0, 2 * np.pi, n_disk)
    w = 1.0 + 0.5 * np.cos(2 * phi)
    w = w / w.sum()
    idx = rng.choice(n_disk, size=n_disk, replace=True, p=w)
    R, phi = R[idx], phi[idx]
    pos_d = np.stack(
        [R * np.cos(phi), R * np.sin(phi), rng.normal(0, 0.2, n_disk)], axis=1
    )
    vel_d = np.stack(
        [-0.8 * np.sin(phi), 0.8 * np.cos(phi), rng.normal(0, 0.05, n_disk)], axis=1
    )
    pos_h = rng.normal(0, 8.0, size=(n_halo, 3))
    vel_h = rng.normal(0, 0.3, size=(n_halo, 3))
    pos_b = rng.normal(0, 0.8, size=(n_bulge, 3))
    vel_b = rng.normal(0, 0.5, size=(n_bulge, 3))
    pos = np.concatenate([pos_d, pos_h, pos_b], axis=0)
    vel = np.concatenate([vel_d, vel_h, vel_b], axis=0)
    mass = np.full(pos.shape[0], 1.0 / pos.shape[0])
    cid = np.concatenate(
        [
            np.zeros(n_disk, dtype=np.int64),
            np.ones(n_halo, dtype=np.int64),
            np.full(n_bulge, 2, dtype=np.int64),
        ]
    )
    return pos, vel, mass, cid


@pytest.mark.essential
def test_bin_vertical_slice_stack_shapes():
    pos, vel, mass, cid = _toy_galaxy()
    cfg = SliceMapConfig(n_pix=16, n_z=4, r_max=12.0, z_max=1.5, moment_set="base")
    stack, meta = bin_vertical_slice_stack(pos, vel, mass, cid, cfg=cfg)
    assert stack.shape == (3 * 4 * 4, 16, 16)
    assert len(meta["channel_names"]) == stack.shape[0]
    dens_idx = dens_channel_indices(cfg)
    assert float(stack[dens_idx].sum()) > 0


@pytest.mark.essential
def test_dispersion_moments_positive():
    pos, vel, mass, cid = _toy_galaxy()
    cfg = MultiScaleSliceConfig.smoke_defaults(
        include_potential=False, moment_set="disp"
    )
    maps = bin_multiscale_slice_stacks(pos, vel, mass, cid, cfg=cfg)
    disk, meta = maps["disk"]
    g = cfg.grid_for("disk")
    assert disk.shape == (g.n_z * 7, g.n_pix, g.n_pix)
    assert list(meta["moment_keys"]) == list(MOMENT_KEYS)
    sx_idx = moment_channel_indices_component(g, "sx")
    assert float(disk[sx_idx].sum()) > 0


@pytest.mark.essential
def test_anisotropy_moment_channel():
    pos, vel, mass, cid = _toy_galaxy(n_disk=400, n_halo=100, n_bulge=80)
    cfg = MultiScaleSliceConfig.smoke_defaults(
        include_potential=False, moment_set="full"
    )
    maps = bin_multiscale_slice_stacks(pos, vel, mass, cid, cfg=cfg)
    g = cfg.grid_for("disk")
    assert g.n_mom == 8
    beta_idx = moment_channel_indices_component(g, "beta")
    beta = maps["disk"][0][beta_idx]
    assert np.isfinite(beta).all()
    assert np.abs(beta).max() <= 2.0 + 1e-5


@pytest.mark.essential
def test_multiscale_slice_native_fovs():
    pos, vel, mass, cid = _toy_galaxy()
    cfg = MultiScaleSliceConfig.smoke_defaults(include_potential=False)
    maps = bin_multiscale_slice_stacks(pos, vel, mass, cid, cfg=cfg)
    assert set(maps) == {"disk", "halo", "bulge"}
    disk_stack, disk_meta = maps["disk"]
    bulge_stack, bulge_meta = maps["bulge"]
    assert disk_stack.shape[1] == cfg.grid_for("disk").n_pix
    assert cfg.grid_for("disk").n_pix >= 64
    assert bulge_meta["r_max"] < disk_meta["r_max"]
    dens_idx = dens_channel_indices_component(cfg.grid_for("disk"))
    assert float(disk_stack[dens_idx].sum()) > 0


@pytest.mark.essential
def test_baseline32_preset():
    cfg = MultiScaleSliceConfig.baseline_32_defaults()
    assert cfg.grid_for("disk").n_pix == 32
    assert cfg.grid_for("disk").moment_set == "base"


@pytest.mark.essential
def test_multiscale_voxels_anisotropic_disk():
    pos, vel, mass, cid = _toy_galaxy(n_disk=300, n_halo=100, n_bulge=80)
    cfg = MultiScaleVoxelConfig.smoke_defaults()
    maps = bin_multiscale_voxel_stacks(pos, vel, mass, cid, cfg=cfg)
    disk = maps["disk"][0]
    g = cfg.grid_for("disk")
    assert disk.shape == (g.n_mom, g.n_z, g.n_xy, g.n_xy)
    assert g.r_z < g.r_xy
    assert g.n_mom == 7


@pytest.mark.essential
def test_normalize_roundtrip_dens():
    pos, vel, mass, cid = _toy_galaxy()
    cfg = SliceMapConfig(n_pix=16, n_z=2, r_max=12.0, z_max=1.0, moment_set="disp")
    stack, _ = bin_vertical_slice_stack(pos, vel, mass, cid, cfg=cfg)
    stats = estimate_norm_stats([stack], cfg=cfg, n_phi=0)
    back = denormalize_stack(normalize_stack(stack, stats), stats)
    dens_idx = dens_channel_indices(cfg)
    assert np.allclose(back[dens_idx], stack[dens_idx], rtol=0.05, atol=1e-5)


@pytest.mark.essential
def test_normalize_component_disp_scale():
    pos, vel, mass, cid = _toy_galaxy()
    cfg = MultiScaleSliceConfig.smoke_defaults(include_potential=False)
    maps = bin_multiscale_slice_stacks(pos, vel, mass, cid, cfg=cfg)
    g = cfg.grid_for("disk")
    stats = estimate_norm_stats_component([maps["disk"][0]], g)
    assert stats.disp_scale > 0
    assert stats.n_mom == 7


@pytest.mark.essential
def test_resample_multiscale_disk():
    pos, vel, mass, cid = _toy_galaxy()
    cfg = MultiScaleSliceConfig.smoke_defaults(include_potential=False)
    maps = bin_multiscale_slice_stacks(pos, vel, mass, cid, cfg=cfg)
    stacks = {k: v[0] for k, v in maps.items()}
    disk_only = MultiScaleSliceConfig(grids=(cfg.grid_for("disk"),))
    out = resample_particles_from_multiscale(
        {"disk": stacks["disk"]},
        cfg=disk_only,
        n_particles=2000,
        rng=np.random.default_rng(0),
    )
    assert out["pos"].shape == (2000, 3)
    assert np.all(out["component_id"] == 0)
    assert float(np.std(out["pos"][:, 2])) < 1.2


@pytest.mark.essential
def test_resample_conserves_fov_slab_mass():
    """Slice dens is Σ=mass/area; bin→resample must recover FOV mass (no Δz)."""
    pos, vel, mass, cid = _toy_galaxy()
    cfg = MultiScaleSliceConfig.smoke_defaults(include_potential=False)
    maps = bin_multiscale_slice_stacks(pos, vel, mass, cid, cfg=cfg)
    stacks = {k: v[0] for k, v in maps.items()}
    for g in cfg.grids:
        grid = g
        cid_i = {"disk": 0, "halo": 1, "bulge": 2}[g.name]
        p = pos[cid == cid_i]
        m = mass[cid == cid_i]
        if p.size == 0:
            continue
        from galacticsics.ml.fields.binning import z_edges_for_grid

        ze = z_edges_for_grid(grid)
        in_fov = (
            (np.abs(p[:, 0]) <= grid.r_max)
            & (np.abs(p[:, 1]) <= grid.r_max)
            & (p[:, 2] >= ze[0])
            & (p[:, 2] <= ze[-1])
        )
        m_fov = float(m[in_fov].sum())
        out = resample_particles_from_multiscale(
            {g.name: stacks[g.name]},
            cfg=MultiScaleSliceConfig(grids=(grid,)),
            n_particles=1500,
            rng=np.random.default_rng(1),
        )
        m_res = float(out["mass"].sum())
        if m_fov > 0:
            assert abs(m_res - m_fov) / m_fov < 0.05, (
                f"{g.name}: resampled mass {m_res:.4g} vs FOV {m_fov:.4g}"
            )


@pytest.mark.essential
def test_resample_shared_stack():
    pos, vel, mass, cid = _toy_galaxy()
    cfg = SliceMapConfig(
        n_pix=24, n_z=4, r_max=12.0, z_max=1.5, moment_set="base"
    )
    stack, _ = bin_vertical_slice_stack(pos, vel, mass, cid, cfg=cfg)
    out = resample_particles_from_slice_stack(
        stack, cfg=cfg, n_particles=1500, components=("disk",), rng=np.random.default_rng(1)
    )
    assert out["pos"].shape[0] == 1500


@pytest.mark.essential
def test_resolve_moment_keys():
    assert resolve_moment_keys("base") == ("dens", "vx", "vy", "vz")
    assert "sx" in resolve_moment_keys("disp")
    assert "beta" in resolve_moment_keys("full")


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="PyTorch not installed",
)
@pytest.mark.essential
def test_multitower_unet_forward():
    import torch

    from galacticsics.ml.fields.autoencoder import MultiTowerSliceAE

    cfg = MultiScaleSliceConfig.smoke_defaults(include_potential=False)
    model = MultiTowerSliceAE(
        cfg,
        include_potential=False,
        base_channels=8,
        latent_channels=16,
        arch="unet",
        cross_tower_attention=True,
    )
    batch = {
        g.name: torch.randn(1, g.n_moment_channels, g.n_pix, g.n_pix) for g in cfg.grids
    }
    out = model(batch)
    assert set(out) == set(batch)
    for k in batch:
        assert out[k].shape == batch[k].shape
        assert torch.isfinite(out[k]).all()


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="PyTorch not installed",
)
@pytest.mark.essential
def test_soft_a2_and_loss():
    import torch

    from galacticsics.ml.fields.autoencoder import (
        dens_map_azimuthal_fourier_numpy,
        reconstruction_loss,
        soft_a2_from_dens_maps,
        soft_am_from_dens_maps,
        soft_am_radial_from_dens_maps,
        soft_fourier_match_loss,
    )

    dens = torch.zeros(2, 32, 32)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), indexing="ij")
    r = torch.sqrt(xx * xx + yy * yy + 1e-16)
    phi = torch.atan2(yy, xx)
    # Radially varying m=2: strong near R~0.4, weak outside (not a single scalar).
    dens[0] = 1.0 + 0.8 * torch.exp(-((r - 0.4) / 0.15) ** 2) * torch.cos(2 * phi)
    dens[1] = 1.0
    a2 = soft_a2_from_dens_maps(dens)
    assert a2[0] > a2[1]
    a1 = soft_am_from_dens_maps(dens, m=1)
    a3 = soft_am_from_dens_maps(dens, m=3)
    assert torch.isfinite(a1).all() and torch.isfinite(a3).all()

    rad = soft_am_radial_from_dens_maps(dens, m=2, n_bins=10)
    assert rad["amp"].shape == (2, 10)
    assert rad["amp"][0].max() > rad["amp"][1].max()
    # Peak of A₂(R) should sit near the injected ring, not be flat.
    peak_bin = int(rad["amp"][0].argmax())
    assert 1 <= peak_bin <= 8
    assert torch.isfinite(rad["cos"]).all() and torch.isfinite(rad["sin"]).all()

    # Radial match loss: identical maps → ~0; axisym vs barred → larger.
    zero = soft_fourier_match_loss(dens[:1], dens[:1], modes=(1, 2, 3))
    big = soft_fourier_match_loss(dens[:1], dens[1:2], modes=(1, 2, 3))
    assert float(zero) < 1e-6
    assert float(big) > float(zero)
    # Quiet gate: same mismatch is down-weighted when floor is high (treat as quiet).
    rawish = soft_fourier_match_loss(
        dens[:1], dens[1:2], modes=(2,), quiet_gate_floor=-1.0, quiet_gate_temp=0.01
    )
    gated = soft_fourier_match_loss(
        dens[:1], dens[1:2], modes=(2,), quiet_gate_floor=0.25, quiet_gate_temp=0.03
    )
    assert float(gated) < float(rawish)

    from galacticsics.ml.fields.autoencoder import (
        axisym_residual_dens_loss,
        spatial_fft_morphology_loss,
    )

    r0 = axisym_residual_dens_loss(dens[:1], dens[:1])
    r1 = axisym_residual_dens_loss(dens[:1], dens[1:2])
    assert float(r0) < 1e-6
    assert float(r1) > float(r0)

    # Spatial FFT morphology: identical → ~0; barred vs quiet → larger; quiet gate.
    fft0 = spatial_fft_morphology_loss(dens[:1], dens[:1])
    fft1 = spatial_fft_morphology_loss(
        dens[1:2], dens[:1], quiet_gate_floor=-1.0
    )  # pred quiet, tgt barred
    assert float(fft0) < 1e-4
    assert float(fft1) > float(fft0) * 10
    # Quiet gate: inventing a bar on a quiet *target* is down-weighted.
    fft_raw = spatial_fft_morphology_loss(
        dens[:1], dens[1:2], quiet_gate_floor=-1.0, quiet_gate_temp=0.01
    )
    fft_gated = spatial_fft_morphology_loss(
        dens[:1], dens[1:2], quiet_gate_floor=0.05, quiet_gate_temp=0.02
    )
    assert float(fft_gated) < float(fft_raw)

    # R_d ring focus: mismatch near focus weighs more than mismatch far out.
    from galacticsics.ml.fields.autoencoder import (
        soft_a2_at_r_match_loss,
        soft_interp_amp_at_r,
    )

    dens_focus = dens[:1].clone()
    dens_far = dens[:1].clone()
    # Perturb near R~0.4 (injected bar) vs outer.
    dens_focus = dens_focus + 0.3 * torch.exp(-((r - 0.4) / 0.1) ** 2) * dens_focus
    dens_far = dens_far + 0.3 * torch.exp(-((r - 0.85) / 0.1) ** 2) * dens_far
    loss_near = soft_fourier_match_loss(
        dens_focus,
        dens[:1],
        modes=(2,),
        quiet_gate_floor=-1.0,
        r_focus=0.4,
        r_focus_peak=6.0,
        r_focus_floor=0.1,
    )
    loss_far = soft_fourier_match_loss(
        dens_far,
        dens[:1],
        modes=(2,),
        quiet_gate_floor=-1.0,
        r_focus=0.4,
        r_focus_peak=6.0,
        r_focus_floor=0.1,
    )
    assert float(loss_near) > float(loss_far)

    a2rd0 = soft_a2_at_r_match_loss(
        dens[:1], dens[:1], r_focus=0.4, quiet_gate_floor=-1.0
    )
    a2rd1 = soft_a2_at_r_match_loss(
        dens[:1], dens[1:2], r_focus=0.4, quiet_gate_floor=-1.0
    )
    assert float(a2rd0) < 1e-5
    assert float(a2rd1) > float(a2rd0)
    rad0 = soft_am_radial_from_dens_maps(dens[:1], m=2, n_bins=10)
    a_at = soft_interp_amp_at_r(rad0, 0.4)
    assert a_at.shape == (1,) and torch.isfinite(a_at).all()

    # Finite grads through FFT morphology + reconstruction_loss.
    pred_g = dens[:1].detach().clone().requires_grad_(True)
    fft_l = spatial_fft_morphology_loss(pred_g, dens[:1] + 0.1 * dens[:1])
    fft_l.backward()
    assert pred_g.grad is not None and torch.isfinite(pred_g.grad).all()

    np_out = dens_map_azimuthal_fourier_numpy(dens[0].numpy(), m=2, n_bins=10, r_max=1.0)
    assert np_out["a_m_over_a0"].shape == (10,)
    assert np.isfinite(np_out["a_m_over_a0"]).any()

    pred = torch.randn(1, 8, 16, 16).abs().requires_grad_(True)
    tgt = pred.detach() + 0.1
    m = reconstruction_loss(
        pred,
        tgt,
        dens_channel_indices=[0, 4],
        dens_weight=5.0,
        moment_weight=5.0,
        dens_resid_weight=1.0,
        fourier_weight=1.0,
        fourier_modes=(1, 2, 3),
        fourier_n_bins=8,
        fft_weight=0.5,
        dens_scale=1.0,
        r_focus=0.35,
        a2_rd_weight=1.0,
    )
    assert "a2_mse" in m and "fourier_mse" in m and "dens_resid_mse" in m
    assert "fft_mse" in m and "a2_rd_mse" in m
    assert "mse_mom" in m
    assert torch.isfinite(m["loss"])
    m["loss"].backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()

    # Moments ≥ dens: moment channels should dominate when dens_weight is tiny.
    m_mom = reconstruction_loss(
        pred.detach(),
        tgt.detach(),
        dens_channel_indices=[0, 4],
        dens_weight=0.1,
        moment_weight=8.0,
        fourier_weight=0.0,
        fft_weight=0.0,
    )
    assert float(m_mom["mse_mom"]) >= 0.0
    assert "fourier_mse" not in m_mom
    assert "fft_mse" not in m_mom


@pytest.mark.essential
def test_shared_centering_preserves_component_offsets():
    """Global COM centering must keep relative disk/bulge offsets."""
    from galacticsics.ml.fields.frame import component_com, prepare_shared_frame

    rng = np.random.default_rng(1)
    n_d, n_b = 500, 200
    # Disk offset +2 kpc in x relative to bulge at origin.
    pos_d = rng.normal(0, 2.0, size=(n_d, 3))
    pos_d[:, 0] += 2.0
    pos_b = rng.normal(0, 0.5, size=(n_b, 3))
    pos = np.concatenate([pos_d, pos_b], axis=0)
    vel = rng.normal(0, 0.1, size=pos.shape)
    mass = np.ones(pos.shape[0])
    cid = np.concatenate(
        [np.zeros(n_d, dtype=np.int64), np.full(n_b, 2, dtype=np.int64)]
    )
    offset_before = component_com(pos, mass, cid, "disk") - component_com(
        pos, mass, cid, "bulge"
    )
    pos_c, vel_c, meta = prepare_shared_frame(pos, vel, mass, center=True, rotate=False)
    offset_after = component_com(pos_c, mass, cid, "disk") - component_com(
        pos_c, mass, cid, "bulge"
    )
    assert np.allclose(offset_before, offset_after, atol=1e-10)
    assert np.linalg.norm(meta["com"]) > 0.5  # was not already at origin
    # Wrong approach (per-component recenter) would zero both COMs:
    assert np.linalg.norm(component_com(pos_c, mass, cid, "disk")) > 0.5


@pytest.mark.essential
def test_conditioning_theta_keys_stable():
    from galacticsics.ml.conditioning import DEFAULT_THETA_KEYS, theta_from_record, theta_vector

    assert len(DEFAULT_THETA_KEYS) == 10
    assert DEFAULT_THETA_KEYS[-1] == "t_gyr"
    v = theta_vector({"disk.mass": 1.0}, keys=DEFAULT_THETA_KEYS)
    assert v.shape == (10,)
    assert v[0] == 1.0
    v2 = theta_from_record({}, t_gyr=1.5)
    assert v2[-1] == 1.5
    v3 = theta_from_record({"t_gyr": float("nan")}, t_gyr=0.83)
    assert abs(v3[-1] - 0.83) < 1e-6


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="PyTorch not installed",
)
@pytest.mark.essential
def test_field_vae_encode_sample_shapes():
    import torch

    from galacticsics.ml.fields.binning import (
        MultiScaleSliceConfig,
        dens_channel_indices_component,
    )
    from galacticsics.ml.fields.vae import FieldVAEConfig, MultiTowerSliceVAE, field_vae_loss

    cfg = MultiScaleSliceConfig.baseline_32_defaults(
        include_potential=False, moment_set="disp"
    )
    model = MultiTowerSliceVAE(
        cfg,
        vae_cfg=FieldVAEConfig(
            latent_dim=8,
            base_channels=16,
            bottleneck_channels=32,
            beta=1e-3,
            prior_decode_weight=0.2,
            skip_dropout=0.0,
        ),
    )
    batch = {
        g.name: torch.randn(2, g.n_moment_channels, g.n_pix, g.n_pix) for g in cfg.grids
    }
    theta = torch.randn(2, 10)
    out = model(batch, theta)
    assert out["z"].shape == (2, 8)
    for g in cfg.grids:
        assert out["recon"][g.name].shape == batch[g.name].shape
    prior = model.sample(theta)
    for g in cfg.grids:
        assert prior[g.name].shape == batch[g.name].shape
    dens_idx = {g.name: dens_channel_indices_component(g) for g in cfg.grids}
    metrics = field_vae_loss(model, batch, theta, dens_indices=dens_idx)
    assert torch.isfinite(metrics["loss"])
    metrics["loss"].backward()


@pytest.mark.essential
def test_progressive_nz_scales_with_disk_pix():
    c128 = MultiScaleSliceConfig.progressive_defaults(disk_n_pix=128)
    c160 = MultiScaleSliceConfig.progressive_defaults(disk_n_pix=160)
    d128 = c128.grid_for("disk")
    d160 = c160.grid_for("disk")
    assert d128.n_pix == 128 and d128.n_z == 14
    assert d160.n_pix == 160 and d160.n_z == 18
    assert c160.grid_for("bulge").n_z == 14
    assert c160.grid_for("halo").n_z == 10
    over = MultiScaleSliceConfig.progressive_defaults(disk_n_pix=160, disk_n_z=20)
    assert over.grid_for("disk").n_z == 20


@pytest.mark.essential
def test_cusp_bulge_defaults_finer_than_progressive():
    prog = MultiScaleSliceConfig.progressive_defaults(disk_n_pix=128)
    cusp = MultiScaleSliceConfig.cusp_bulge_defaults(disk_n_pix=128)
    pb, cb = prog.grid_for("bulge"), cusp.grid_for("bulge")
    assert cb.n_pix >= 128 and cb.n_z >= 32
    assert cb.n_pix > pb.n_pix or cb.n_z > pb.n_z
    assert cusp.grid_for("disk").n_pix == prog.grid_for("disk").n_pix


@pytest.mark.essential
def test_copy_dens_and_rescale_mass():
    from galacticsics.ml.fields.resample import (
        copy_dens_channels,
        integrated_slab_mass,
        rescale_dens_channels_to_mass,
    )

    cfg = MultiScaleSliceConfig.smoke_defaults(moment_set="disp")
    maps: dict[str, np.ndarray] = {}
    for g in cfg.grids:
        stack = np.zeros((g.n_z * g.n_mom, g.n_pix, g.n_pix), dtype=np.float64)
        dens_i = g.moment_keys.index("dens")
        for iz in range(g.n_z):
            stack[iz * g.n_mom + dens_i] = 0.1
            if "vx" in g.moment_keys:
                stack[iz * g.n_mom + g.moment_keys.index("vx")] = 0.5
        maps[g.name] = stack
    recon = {k: v * 0.5 for k, v in maps.items()}
    out = copy_dens_channels(maps, recon, cfg=cfg, components=("disk",))
    g = cfg.grid_for("disk")
    assert np.allclose(out["disk"][0], maps["disk"][0])
    assert np.allclose(
        out["disk"][g.moment_keys.index("vx")],
        recon["disk"][g.moment_keys.index("vx")],
    )
    m0 = integrated_slab_mass(recon["disk"], g)
    scaled = rescale_dens_channels_to_mass(
        recon,
        cfg=cfg,
        mass_total_per_component={"disk": 2 * m0, "halo": 1.0, "bulge": 1.0},
    )
    assert abs(integrated_slab_mass(scaled["disk"], g) - 2 * m0) / m0 < 1e-6

    from galacticsics.ml.fields.resample import (
        amplify_axisym_residual_dens,
        alpha_match_residual_power,
        dens_axisym_residual_rms,
    )

    # Synthetic bar: dens = axisym + m=2 residual; α=2 should double residual.
    g = cfg.grid_for("disk")
    n = g.n_pix
    yy, xx = np.mgrid[0:n, 0:n]
    cx = 0.5 * (n - 1)
    phi = np.arctan2(yy - cx, xx - cx)
    dens0 = 1.0 + 0.2 * np.cos(2 * phi)
    dens_ref = 1.0 + 0.4 * np.cos(2 * phi)
    stack = np.zeros((g.n_z * g.n_mom, n, n), dtype=np.float64)
    dens_i = g.moment_keys.index("dens")
    for iz in range(g.n_z):
        stack[iz * g.n_mom + dens_i] = dens0
    amp = amplify_axisym_residual_dens(
        {g.name: stack}, cfg=cfg, components=(g.name,), alpha=2.0
    )[g.name][dens_i]
    # Residual ≈ 0.2 cos(2φ); after α=2 → ≈ 0.4 cos(2φ) about axisym≈1.
    assert float(amp.max() - amp.min()) > float(dens0.max() - dens0.min()) * 1.5
    # m2-only amplify should also strengthen a pure m=2 residual.
    from galacticsics.ml.fields.resample import m2_residual_field

    amp_m2 = amplify_axisym_residual_dens(
        {g.name: stack},
        cfg=cfg,
        components=(g.name,),
        alpha=2.0,
        mode="m2",
        other_alpha=1.0,
    )[g.name][dens_i]
    assert float(amp_m2.max() - amp_m2.min()) > float(dens0.max() - dens0.min()) * 1.4
    m2 = m2_residual_field(dens0 - 1.0, r_max=float(g.r_max))
    assert float(np.corrcoef(m2.ravel(), (dens0 - 1.0).ravel())[0, 1]) > 0.9
    a_match = alpha_match_residual_power(dens0, dens_ref)
    assert 1.8 < a_match < 2.2
    assert dens_axisym_residual_rms(dens_ref) > dens_axisym_residual_rms(dens0)


@pytest.mark.essential
def test_inject_predicted_contrast_preserves_axisym():
    from galacticsics.ml.fields.resample import (
        inject_predicted_contrast_on_f0_dens,
        _axisym_and_rbin,
    )

    cfg = MultiScaleSliceConfig.smoke_defaults(moment_set="base")
    g = cfg.grid_for("disk")
    n_mom = g.n_mom
    dens_i = g.moment_keys.index("dens")
    n = g.n_pix
    yy, xx = np.mgrid[0:n, 0:n]
    cx = 0.5 * (n - 1)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cx) ** 2)
    phi = np.arctan2(yy - cx, xx - cx)
    axisym = np.exp(-rr / max(n / 4.0, 1.0))
    stack = np.zeros((g.n_z * n_mom, n, n), dtype=np.float64)
    for iz in range(g.n_z):
        stack[iz * n_mom + dens_i] = axisym
    contrast = 0.4 * np.cos(2 * phi)
    out = inject_predicted_contrast_on_f0_dens(
        {"disk": stack},
        contrast,
        cfg=cfg,
        components=("disk",),
        alpha=1.0,
        preserve_axisym=True,
    )
    mid = out["disk"][(g.n_z // 2) * n_mom + dens_i]
    ax_out, _ = _axisym_and_rbin(mid)
    ax_in, _ = _axisym_and_rbin(axisym)
    rel = float(np.max(np.abs(ax_out - ax_in) / np.maximum(ax_in, 1e-12)))
    assert rel < 1e-5
    assert float(mid.std()) > float(axisym.std())


@pytest.mark.essential
def test_inject_morph_residual_on_f0_and_transplant():
    from galacticsics.ml.fields.resample import (
        inject_morph_residual_on_f0_dens,
        transplant_velocities_knn,
        _axisym_and_rbin,
    )

    pos, vel, mass, cid = _toy_galaxy(n_disk=600, n_halo=200, n_bulge=100, seed=3)
    cfg = MultiScaleSliceConfig.smoke_defaults(moment_set="base")
    binned = bin_multiscale_slice_stacks(pos, vel, mass, cid, cfg=cfg)
    maps_f0 = {k: v[0].copy() for k, v in binned.items()}
    maps_morph = {k: v.copy() for k, v in maps_f0.items()}
    g = cfg.grid_for("disk")
    dens_i = g.moment_keys.index("dens")
    iz0 = g.n_z // 2
    n = g.n_pix
    yy, xx = np.mgrid[0:n, 0:n]
    cx = 0.5 * (n - 1)
    phi = np.arctan2(yy - cx, xx - cx)
    sl = iz0 * g.n_mom + dens_i
    base = np.maximum(maps_morph["disk"][sl], 0.0)
    maps_morph["disk"][sl] = base * (1.0 + 0.5 * np.cos(2 * phi))

    inj = inject_morph_residual_on_f0_dens(
        maps_f0,
        maps_morph,
        cfg=cfg,
        components=("disk",),
        alpha=1.0,
        mode="m2",
        other_morph_alpha=0.0,
        preserve_axisym=True,
    )
    d0 = maps_f0["disk"][sl]
    d1 = inj["disk"][sl]
    assert float(d1.max() - d1.min()) > float(d0.max() - d0.min()) + 1e-6
    ax0, _ = _axisym_and_rbin(np.maximum(d0, 0.0))
    ax1, _ = _axisym_and_rbin(np.maximum(d1, 0.0))
    mask = ax0 > 0.05 * float(ax0.max())
    # Axisym floor preserved (residual is pure m=2 contrast).
    assert float(np.max(np.abs(ax1[mask] - ax0[mask]) / np.maximum(ax0[mask], 1e-30))) < 0.05

    # Strong α must not collapse radial mass via clipping.
    inj25 = inject_morph_residual_on_f0_dens(
        maps_f0,
        maps_morph,
        cfg=cfg,
        components=("disk",),
        alpha=2.5,
        mode="m2",
        preserve_axisym=True,
        factor_floor=0.05,
    )
    d25 = inj25["disk"][sl]
    ax25, _ = _axisym_and_rbin(np.maximum(d25, 0.0))
    assert float(np.max(np.abs(ax25[mask] - ax0[mask]) / np.maximum(ax0[mask], 1e-30))) < 0.05
    assert float(np.mean(d25 <= 0.0)) < float(np.mean(d0 <= 0.0)) + 0.05

    mom_i = g.moment_keys.index("vx") if "vx" in g.moment_keys else None
    if mom_i is not None:
        assert np.allclose(
            inj["disk"][iz0 * g.n_mom + mom_i],
            maps_f0["disk"][iz0 * g.n_mom + mom_i],
        )

    pos2 = pos.copy()
    pos2[cid == 0, 0] += 0.05
    vel2, meta = transplant_velocities_knn(
        pos2,
        cid,
        pos,
        vel,
        cid,
        components=("disk",),
        n_ref=400,
        rng=np.random.default_rng(0),
    )
    disk = cid == 0
    assert "disk" in meta["components"]
    assert float(np.median(np.linalg.norm(vel2[disk], axis=1))) > 0.1


@pytest.mark.essential
def test_spherical_shell_bulge_cusp_and_stitch():
    from galacticsics.ml.fields.resample import (
        bin_spherical_shell_moments,
        fuse_shell_bulge_with_multiscale,
        stitch_retained_components,
    )

    rng = np.random.default_rng(7)
    # Hernquist-like cusp: dens ∝ 1/(r (r+a)^3)
    a = 0.4
    u = rng.random(8000)
    r = a * np.sqrt(u) / (1.0 - np.sqrt(u) + 1e-6)
    r = np.clip(r, 0.02, 4.0)
    mu = rng.uniform(-1, 1, r.size)
    phi = rng.uniform(0, 2 * np.pi, r.size)
    s = np.sqrt(1 - mu * mu)
    pos_b = np.stack([r * s * np.cos(phi), r * s * np.sin(phi), r * mu], axis=1)
    vel_b = rng.normal(0, 0.4, size=pos_b.shape)
    mass_b = np.full(pos_b.shape[0], 1.0 / pos_b.shape[0])
    shells = bin_spherical_shell_moments(
        pos_b, vel_b, mass_b, n_shells=32, r_min=0.05, r_max=4.0
    )
    assert float(shells["dens"][0]) > float(shells["dens"][-1])

    # Toy disk/halo maps via multiscale binning of a small system.
    pos, vel, mass, cid = _toy_galaxy(n_disk=600, n_halo=300, n_bulge=200, seed=3)
    cfg = MultiScaleSliceConfig.smoke_defaults(moment_set="base")
    stacks = bin_multiscale_slice_stacks(pos, vel, mass, cid, cfg=cfg)
    maps = {k: v[0] for k, v in stacks.items()}
    fused = fuse_shell_bulge_with_multiscale(
        maps,
        cfg=cfg,
        bulge_shells=shells,
        n_particles=2000,
        count_fractions={"disk": 4 / 7, "halo": 2 / 7, "bulge": 1 / 7},
        rng=rng,
    )
    assert fused["bulge_method"] == "spherical_shells"
    assert (fused["component_id"] == 2).sum() > 0

    kept = stitch_retained_components(
        fused,
        source_pos=pos,
        source_vel=vel,
        source_mass=mass,
        source_cid=cid,
        retain=("bulge",),
        n_retain={"bulge": 150},
        rng=rng,
    )
    assert "bulge" in kept["retained_components"]
    assert (kept["component_id"] == 2).sum() == 150


@pytest.mark.essential
def test_fit_index_basis_pca_lda_shapes():
    from galacticsics.ml.fields.feature_library import fit_index_basis

    rng = np.random.default_rng(0)
    Z = rng.normal(size=(60, 24))
    a2 = np.concatenate(
        [
            rng.uniform(0.25, 0.55, 22),
            rng.uniform(0.0, 0.04, 18),
            rng.uniform(0.06, 0.18, 20),
        ]
    )
    for method in ("pca", "lda_concat", "whiten_pca", "pls_a2", "multiclass_lda"):
        mean, W, codes, meta = fit_index_basis(
            Z, a2, method=method, n_comp=8, bar_floor=0.22, quiet_ceil=0.05
        )
        assert mean.shape == (24,)
        assert W.shape == (24, 8)
        assert codes.shape == (60, 8)
        assert meta["method"] == method
        assert np.isfinite(codes).all()


@pytest.mark.essential
def test_resample_match_cell_moments_redeposit():
    """B2: cell affine match → occupied-cell sample ⟨v⟩ hits deposit targets."""
    from galacticsics.ml.fields.binning import z_edges_for_grid

    pos, vel, mass, cid = _toy_galaxy(n_disk=3000, n_halo=200, n_bulge=100, seed=3)
    cfg = MultiScaleSliceConfig.smoke_defaults(include_potential=False)
    maps = bin_multiscale_slice_stacks(pos, vel, mass, cid, cfg=cfg)
    stacks = {k: v[0] for k, v in maps.items()}
    disk_only = MultiScaleSliceConfig(grids=(cfg.grid_for("disk"),))
    g = cfg.grid_for("disk")
    keys = g.moment_keys
    n_mom = len(keys)
    n_pix = int(g.n_pix)
    xy = np.linspace(-float(g.r_max), float(g.r_max), n_pix + 1)
    ze = z_edges_for_grid(g)

    def _occupied_cell_mean_mse(out: dict) -> float:
        x, y, z = out["pos"].T
        ix = np.clip(np.searchsorted(xy, x, side="right") - 1, 0, n_pix - 1)
        iy = np.clip(np.searchsorted(xy, y, side="right") - 1, 0, n_pix - 1)
        iz = np.clip(np.searchsorted(ze, z, side="right") - 1, 0, int(g.n_z) - 1)
        flat = iz * (n_pix * n_pix) + ix * n_pix + iy
        tgt = np.zeros((out["pos"].shape[0], 3), dtype=np.float64)
        for k, name in enumerate(("vx", "vy", "vz")):
            ch = np.stack(
                [stacks["disk"][iz_ * n_mom + keys.index(name)] for iz_ in range(g.n_z)]
            )
            tgt[:, k] = ch[iz, ix, iy]
        order = np.argsort(flat)
        sf = flat[order]
        sv = out["vel"][order]
        st = tgt[order]
        breaks = np.flatnonzero(np.diff(sf)) + 1
        starts = np.concatenate([[0], breaks])
        ends = np.concatenate([breaks, [len(sf)]])
        errs = []
        for a, b in zip(starts, ends):
            errs.append(float(np.mean((sv[a:b].mean(axis=0) - st[a]) ** 2)))
        return float(np.mean(errs)) if errs else 0.0

    base = resample_particles_from_multiscale(
        {"disk": stacks["disk"]},
        cfg=disk_only,
        n_particles=12_000,
        rng=np.random.default_rng(0),
        velocity_frame="cartesian",
        match_cell_moments=False,
    )
    matched = resample_particles_from_multiscale(
        {"disk": stacks["disk"]},
        cfg=disk_only,
        n_particles=12_000,
        rng=np.random.default_rng(0),
        velocity_frame="cylindrical",
        match_cell_moments=True,
    )
    mse_base = _occupied_cell_mean_mse(base)
    mse_matched = _occupied_cell_mean_mse(matched)
    assert mse_matched < 1e-20, f"matched cell mean MSE {mse_matched:.3e}"
    assert mse_matched < mse_base * 1e-6, (
        f"match_cell_moments should crush cell-mean error: "
        f"{mse_matched:.4e} vs baseline {mse_base:.4e}"
    )


@pytest.mark.essential
def test_resample_cylindrical_frame_runs():
    pos, vel, mass, cid = _toy_galaxy()
    cfg = MultiScaleSliceConfig.smoke_defaults(include_potential=False)
    maps = bin_multiscale_slice_stacks(pos, vel, mass, cid, cfg=cfg)
    stacks = {k: v[0] for k, v in maps.items()}
    disk_only = MultiScaleSliceConfig(grids=(cfg.grid_for("disk"),))
    out = resample_particles_from_multiscale(
        {"disk": stacks["disk"]},
        cfg=disk_only,
        n_particles=1500,
        rng=np.random.default_rng(2),
        velocity_frame="cylindrical",
        match_cell_moments=True,
    )
    assert out["pos"].shape == (1500, 3)
    assert np.isfinite(out["vel"]).all()


@pytest.mark.essential
def test_transport_ot_lite_matches_radial_cdf_and_vphi():
    """F: radial CDF remap + ⟨v_φ⟩ profile → closer to target than raw source."""
    from galacticsics.ml.fields.resample import transport_ot_lite
    from galacticsics.ml.profiles import cylindrical_radius, v_phi_cylindrical

    rng = np.random.default_rng(11)
    n = 4000
    # Compact cold source disk vs extended hotter target (mass-model offset).
    R_s = rng.exponential(2.0, size=n)
    phi_s = rng.uniform(0, 2 * np.pi, size=n)
    pos_s = np.column_stack(
        [R_s * np.cos(phi_s), R_s * np.sin(phi_s), rng.normal(0, 0.2, n)]
    )
    vphi_s = 180.0 * np.ones(n)  # too slow / wrong
    vel_s = np.column_stack(
        [-vphi_s * np.sin(phi_s), vphi_s * np.cos(phi_s), rng.normal(0, 10, n)]
    )
    R_t = rng.exponential(4.0, size=n)
    phi_t = rng.uniform(0, 2 * np.pi, size=n)
    pos_t = np.column_stack(
        [R_t * np.cos(phi_t), R_t * np.sin(phi_t), rng.normal(0, 0.3, n)]
    )
    vphi_t = 220.0 * np.ones(n)
    vel_t = np.column_stack(
        [-vphi_t * np.sin(phi_t), vphi_t * np.cos(phi_t), rng.normal(0, 15, n)]
    )
    mass = np.ones(n)
    cid = np.zeros(n, dtype=np.int32)
    src = {"pos": pos_s, "vel": vel_s, "mass": mass, "component_id": cid}
    tgt = {"pos": pos_t, "vel": vel_t, "mass": mass.copy(), "component_id": cid.copy()}
    out, meta = transport_ot_lite(src, tgt, match_dispersion=False)
    assert meta["method"] == "ot_lite_radial_vphi"
    R_out = cylindrical_radius(out["pos"])
    # Median R should move toward target.
    assert abs(np.median(R_out) - np.median(R_t)) < abs(np.median(R_s) - np.median(R_t))
    vphi_out = v_phi_cylindrical(out["pos"], out["vel"])
    assert abs(float(np.mean(vphi_out)) - 220.0) < abs(float(np.mean(vphi_s)) - 220.0)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch required",
)
@pytest.mark.essential
def test_dens_weighted_moment_and_vphi_loss():
    """A: √Σ moment / vφ phys losses are finite and backprop-ready."""
    import torch
    from galacticsics.ml.fields.autoencoder import (
        dens_weighted_moment_loss,
        dens_weighted_vphi_field_loss,
        reconstruction_loss,
    )

    b, n_z, n_mom, h, w = 2, 3, 7, 16, 16
    dens_idx = [i * n_mom for i in range(n_z)]
    target = torch.randn(b, n_z * n_mom, h, w)
    # dens channels ≥ 0 in log1p space
    for di in dens_idx:
        target[:, di].clamp_(min=0.0)
    pred = target + 0.1 * torch.randn_like(target)
    dens_scale = 1.0
    mp = dens_weighted_moment_loss(
        pred, target, dens_channel_indices=dens_idx, dens_scale=dens_scale, n_mom=n_mom
    )
    vp = dens_weighted_vphi_field_loss(
        pred, target, dens_channel_indices=dens_idx, dens_scale=dens_scale, n_mom=n_mom
    )
    assert torch.isfinite(mp) and float(mp) > 0
    assert torch.isfinite(vp) and float(vp) >= 0
    pred2 = pred.detach().requires_grad_(True)
    m = reconstruction_loss(
        pred2,
        target,
        dens_channel_indices=dens_idx,
        dens_weight=4.0,
        moment_weight=6.0,
        dens_scale=dens_scale,
        moment_phys_weight=2.0,
        vphi_phys_weight=2.0,
        n_mom=n_mom,
    )
    m["loss"].backward()
    assert pred2.grad is not None
    assert "moment_phys_mse" in m and "vphi_phys_mse" in m
