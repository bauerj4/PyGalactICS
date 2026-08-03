"""Tests for the conditional set VAE and soft profile auxiliaries."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch  # noqa: F401

    _TORCH = True
except ImportError:  # pragma: no cover
    _TORCH = False


@pytest.mark.essential
def test_numpy_profile_helpers():
    from galacticsics.ml.profiles import cylindrical_radius, spherical_radius, v_phi_cylindrical

    pos = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
    vel = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    assert np.allclose(cylindrical_radius(pos), [5.0, 0.0])
    assert np.allclose(spherical_radius(pos), [5.0, 5.0])
    assert np.allclose(v_phi_cylindrical(pos, vel)[0], (-4.0 * 0.0 + 3.0 * 1.0) / 5.0)


@pytest.mark.skipif(not _TORCH, reason="PyTorch not installed")
@pytest.mark.essential
def test_soft_azimuthal_fourier_axisym_vs_bar():
    import torch

    from galacticsics.ml.profiles import soft_azimuthal_fourier

    rng = np.random.default_rng(0)
    n = 2000
    R = rng.exponential(scale=3.0, size=n).clip(0.2, 12.0)
    phi = rng.uniform(0, 2 * np.pi, size=n)
    pos_ax = np.stack([R * np.cos(phi), R * np.sin(phi), rng.normal(0, 0.05, n)], axis=1)
    mass = torch.full((1, n), 1.0 / n)
    ax = soft_azimuthal_fourier(torch.as_tensor(pos_ax[None], dtype=torch.float32), mass, m=2)
    amp = 0.4
    w = 1.0 + amp * np.cos(2 * phi)
    w = w / w.sum()
    idx = rng.choice(n, size=n, replace=True, p=w)
    pos_bar = pos_ax[idx]
    bar = soft_azimuthal_fourier(torch.as_tensor(pos_bar[None], dtype=torch.float32), mass, m=2)
    assert float(ax["amp"].mean()) < float(bar["amp"].mean())


@pytest.mark.skipif(not _TORCH, reason="PyTorch not installed")
@pytest.mark.essential
def test_soft_plane_density_map_shape_and_mass():
    import torch

    from galacticsics.ml.profiles import soft_plane_density_map

    b, n, n_pix = 2, 50, 16
    pos = torch.randn(b, n, 3) * 3.0
    mass = torch.full((b, n), 1.0 / n)
    dens = soft_plane_density_map(pos, mass, n_pix=n_pix, r_max=10.0)
    assert dens.shape == (b, n_pix, n_pix)
    # Softmax splat conserves mass for particles inside the soft mask
    assert torch.allclose(dens.sum(dim=(1, 2)), mass.sum(dim=1), rtol=0.15, atol=0.05)


@pytest.mark.skipif(not _TORCH, reason="PyTorch not installed")
@pytest.mark.essential
def test_plummer_virial_stats_bound_cloud():
    import torch

    from galacticsics.ml.profiles import plummer_virial_stats, virial_consistency_loss

    rng = np.random.default_rng(0)
    n = 128
    pos = torch.as_tensor(rng.normal(0, 2.0, size=(1, n, 3)), dtype=torch.float32)
    # Near-circular support: v ~ sqrt(GM/r) scale → ratio O(1)
    vel = torch.as_tensor(rng.normal(0, 0.4, size=(1, n, 3)), dtype=torch.float32)
    st = plummer_virial_stats(pos, vel, eps=0.15)
    assert st["ke"].shape == (1,)
    assert torch.isfinite(st["virial_ratio"]).all()
    assert float(st["pe"]) < 0.0

    c = torch.zeros(1, n, dtype=torch.long)
    c[:, n // 3 : 2 * n // 3] = 1
    c[:, 2 * n // 3 :] = 2
    pos2 = (pos + 0.05 * torch.randn_like(pos)).requires_grad_(True)
    vel2 = (vel + 0.05 * torch.randn_like(vel)).requires_grad_(True)
    loss = virial_consistency_loss(pos, vel, c, pos2, vel2, n_sub=64, eps=0.15)
    assert torch.isfinite(loss["virial"])
    loss["virial"].backward()
    assert pos2.grad is not None and torch.isfinite(pos2.grad).all()


@pytest.mark.skipif(not _TORCH, reason="PyTorch not installed")
@pytest.mark.essential
def test_sequence_vae_forward_loss_and_diverse_generate():
    import torch

    from galacticsics.ml.models.sequence_vae import SequenceVAE, SequenceVAEConfig
    from galacticsics.ml.profiles import profile_reconstruction_loss, soft_surface_density

    cfg = SequenceVAEConfig(
        n_particles=32,
        theta_dim=4,
        d_model=32,
        latent_dim=8,
        n_layers=1,
        n_heads=2,
        n_decode_layers=1,
        beta=0.01,
        map_n_pix=16,
        lambda_virial=1.0,
        virial_n_sub=24,
    )
    model = SequenceVAE(cfg)
    b, n = 2, 32
    batch = {
        "c": torch.randint(0, 3, (b, n)),
        "dm": torch.randn(b, n),
        "dx": torch.randn(b, n, 3),
        "v": torch.randn(b, n, 3) * 0.1,
        "theta": torch.randn(b, cfg.theta_dim),
    }
    out = model(batch["c"], batch["dm"], batch["dx"], batch["v"], batch["theta"])
    assert out["logits_c"].shape == (b, n, 3)
    assert out["logits_c_data"].shape == (b, n, 3)
    assert out["mix_logits"].shape == (b, 3)
    assert out["dx"].shape == (b, n, 3)
    metrics = model.loss(batch, out)
    assert torch.isfinite(metrics["loss"])
    assert "ce" in metrics and torch.isfinite(metrics["ce"])
    assert "ce_acc" in metrics and torch.isfinite(metrics["ce_acc"])
    assert "mix_mse" in metrics and torch.isfinite(metrics["mix_mse"])
    assert "virial" in metrics and torch.isfinite(metrics["virial"])
    assert "profile" in metrics and torch.isfinite(metrics["profile"])
    assert "nonaxisym" in metrics and torch.isfinite(metrics["nonaxisym"])
    assert "maps" in metrics and torch.isfinite(metrics["maps"])
    assert "am2" in metrics and torch.isfinite(metrics["am2"])
    assert "virial" in metrics and torch.isfinite(metrics["virial"])

    sigma = soft_surface_density(
        batch["dx"], torch.full((b, n), 1.0 / n), n_bins=cfg.profile_n_bins
    )
    assert sigma.shape == (b, cfg.profile_n_bins)
    prof = profile_reconstruction_loss(
        batch["dx"],
        batch["v"],
        batch["c"],
        out["dx"],
        out["v"],
        out["logits_c"],
        n_bins=cfg.profile_n_bins,
        map_n_pix=16,
    )
    assert torch.isfinite(prof["profile"])
    assert torch.isfinite(prof["nonaxisym"])
    assert torch.isfinite(prof["maps"])

    gen = model.generate(batch["theta"][:1], n=48, chunk_size=16)
    assert gen["dx"].shape == (1, 48, 3)
    assert float(np.std(gen["dx"][0])) > 1e-4
    # Stratified prior init ≈ 4:2:1 — not near-uniform
    counts = np.bincount(gen["c"][0].astype(int), minlength=3).astype(float)
    assert counts[0] > counts[2]
    z1 = torch.randn(1, cfg.latent_dim)
    z2 = torch.randn(1, cfg.latent_dim)
    g1 = model.generate(batch["theta"][:1], n=32, z=z1)
    g2 = model.generate(batch["theta"][:1], n=32, z=z2)
    assert not np.allclose(g1["dx"], g2["dx"])


@pytest.mark.skipif(not _TORCH, reason="PyTorch not installed")
@pytest.mark.essential
def test_phase_space_ce_is_easy_on_separated_components():
    """Disk / halo / bulge are dynamically distinct — CE should drop fast."""
    import torch

    from galacticsics.ml.models.sequence_vae import SequenceVAE, SequenceVAEConfig

    cfg = SequenceVAEConfig(
        n_particles=96,
        theta_dim=4,
        d_model=32,
        latent_dim=8,
        n_layers=0,
        n_heads=2,
        enc_attn_n=0,
        lambda_virial=0.0,
        lambda_nonaxisym=0.0,
        lambda_maps=0.0,
        map_n_pix=8,
    )
    model = SequenceVAE(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    b, n = 4, 96
    # Synthetic, well-separated components
    c = torch.cat(
        [
            torch.zeros(b, n // 2, dtype=torch.long),
            torch.ones(b, n // 4, dtype=torch.long),
            torch.full((b, n - n // 2 - n // 4), 2, dtype=torch.long),
        ],
        dim=1,
    )
    dx = torch.randn(b, n, 3)
    # disk: thin midplane; halo: large sphere; bulge: compact core
    dx[:, : n // 2, 2] *= 0.05
    dx[:, : n // 2, :2] *= 3.0
    dx[:, n // 2 : n // 2 + n // 4] *= 25.0
    dx[:, n // 2 + n // 4 :] *= 1.5
    v = torch.randn(b, n, 3) * 0.2
    v[:, : n // 2, 0] = -dx[:, : n // 2, 1] * 0.3
    v[:, : n // 2, 1] = dx[:, : n // 2, 0] * 0.3
    batch = {
        "c": c,
        "dm": torch.zeros(b, n),
        "dx": dx,
        "v": v,
        "theta": torch.randn(b, cfg.theta_dim),
    }
    model.train()
    last_ce = None
    last_acc = None
    for _ in range(60):
        out = model(batch["c"], batch["dm"], batch["dx"], batch["v"], batch["theta"])
        m = model.loss(batch, out)
        opt.zero_grad(set_to_none=True)
        m["loss"].backward()
        opt.step()
        last_ce = float(m["ce"].detach())
        last_acc = float(m["ce_acc"].detach())
    assert last_acc is not None and last_acc > 0.85
    assert last_ce is not None and last_ce < 0.7
