"""Unit tests for frozen-AE + skip-distill latent code path."""

from __future__ import annotations

import torch

from galacticsics.ml.fields.autoencoder import MultiTowerSliceAE, SliceAutoencoderConfig, SliceUNet
from galacticsics.ml.fields.binning import MultiScaleSliceConfig
from galacticsics.ml.fields.latent_code import (
    FrozenAECodeVAE,
    LatentCodeConfig,
    latent_code_loss,
    morph_a2_summary,
)


def test_slice_unet_encode_decode_features_roundtrip():
    cfg = SliceAutoencoderConfig(
        in_channels=14, base_channels=8, latent_channels=16, n_pix=16, n_z=2, n_mom=7
    )
    net = SliceUNet(cfg)
    x = torch.randn(1, 14, 16, 16)
    b, skips = net.encode_with_skips(x)
    y = net.decode_from_features(b, skips, target_hw=(16, 16), n_out_channels=14)
    assert y.shape == x.shape
    # Same path as forward should be close (heads identical).
    y2 = net(x)
    assert torch.allclose(y, y2, atol=1e-5)


def test_frozen_ae_code_vae_shapes_and_grad():
    from galacticsics.ml.fields.binning import ComponentSliceGrid

    slice_cfg = MultiScaleSliceConfig(
        grids=(
            ComponentSliceGrid("bulge", n_pix=16, n_z=2, r_max=4.0, z_max=4.0, moment_set="disp"),
            ComponentSliceGrid("disk", n_pix=16, n_z=2, r_max=12.0, z_max=1.5, moment_set="disp"),
            ComponentSliceGrid("halo", n_pix=8, n_z=2, r_max=40.0, z_max=40.0, moment_set="disp"),
        ),
        include_potential=False,
    )
    teacher = MultiTowerSliceAE(
        slice_cfg,
        base_channels=8,
        latent_channels=16,
        arch="unet",
        cross_tower_attention=False,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    model = FrozenAECodeVAE(
        teacher,
        cfg=LatentCodeConfig(
            latent_dim=32,
            theta_dim=10,
            morph_dim=0,
            enc_grid=2,
            synth_grid=4,
            beta=1e-3,
        ),
        use_flow_prior=False,
    )
    batch = {
        g.name: torch.randn(1, g.n_moment_channels, g.n_pix, g.n_pix)
        for g in slice_cfg.grids
    }
    theta = torch.randn(1, 10)
    dens_idx = {
        g.name: list(range(0, g.n_moment_channels, g.n_mom)) for g in slice_cfg.grids
    }
    metrics = latent_code_loss(
        model, batch, theta, dens_indices=dens_idx, dens_weight=4.0, moment_weight=6.0
    )
    assert torch.isfinite(metrics["loss"])
    metrics["loss"].backward()
    # Teacher must stay frozen (no grads).
    for p in teacher.parameters():
        assert p.grad is None
    # Student synth has grads.
    assert any(p.grad is not None for p in model.synth.parameters())

    with torch.no_grad():
        samp = model.sample(theta)
    assert set(samp.keys()) == set(batch.keys())
    for k, v in samp.items():
        assert v.shape == batch[k].shape


def test_morph_a2_summary_shape():
    dens = torch.rand(2, 32, 32)
    m = morph_a2_summary(dens)
    assert m.shape == (2, 4)
    assert torch.isfinite(m).all()
