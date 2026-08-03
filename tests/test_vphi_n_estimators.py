"""⟨v_φ⟩(R) must not fall with decreasing N when bins are empty.

Empty / under-populated bins previously returned 0 (particle profiles) or were
included as v=0 in equal-pixel map reductions; deposit ``histogram2d`` layout
must be respected when forming v_φ from (vx, vy).
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.essential
def test_numpy_mass_weighted_profile_empty_is_nan_not_zero():
    from galacticsics.ml.profiles import numpy_mass_weighted_profile

    # Particles only in the inner half of [0, 10].
    rng = np.random.default_rng(0)
    r = rng.uniform(0.0, 4.0, size=500)
    v = np.full(500, 2.0)
    m = np.ones(500)
    r_mid, prof = numpy_mass_weighted_profile(r, v, m, n_bins=10, r_max=10.0)
    assert np.all(np.isfinite(prof[r_mid < 4.0]))
    assert np.all(np.isnan(prof[r_mid > 5.0]))
    # Legacy trap: empty→0 would look like a falling rotation curve.
    assert not np.any(prof[r_mid > 5.0] == 0.0)


@pytest.mark.essential
def test_numpy_mass_weighted_profile_min_count():
    from galacticsics.ml.profiles import numpy_mass_weighted_profile

    r = np.array([0.5, 0.6, 3.0])
    v = np.array([2.0, 2.0, 9.0])
    m = np.ones(3)
    _, prof = numpy_mass_weighted_profile(
        r, v, m, n_bins=4, r_max=4.0, min_count=2
    )
    # First bin has 2 particles → finite; lone particle bin → NaN.
    assert np.isfinite(prof[0])
    assert np.isnan(prof[2]) or np.isnan(prof[3]) or np.isnan(prof[1])


@pytest.mark.essential
def test_radial_vphi_from_deposit_excludes_empty_and_matches_orientation():
    from galacticsics.ml.fields.binning import radial_vphi_from_deposit_slab

    n = 32
    r_max = 8.0
    edges = np.linspace(-r_max, r_max, n + 1)
    xc = 0.5 * (edges[:-1] + edges[1:])
    X = xc[:, None]
    Y = xc[None, :]
    R = np.sqrt(X * X + Y * Y)
    omega = 0.25
    # Rigid rotation: vx=-Ωy, vy=Ωx → v_φ=ΩR
    dens = np.exp(-R / 3.0)
    dens[R > 6.0] = 0.0  # empty outer ring
    vx = -omega * Y
    vy = omega * X
    # Zero-fill empties the way deposit does
    vx = np.where(dens > 0, vx, 0.0)
    vy = np.where(dens > 0, vy, 0.0)

    r_mid, mean, mass = radial_vphi_from_deposit_slab(
        dens, vx, vy, r_max=r_max, n_bins=16
    )
    ok = mass > 0
    # Dens-weighted mean should track ΩR, not collapse toward 0.
    expect = omega * r_mid
    assert np.allclose(mean[ok], expect[ok], rtol=0.05, atol=0.05)
    assert np.all(np.isnan(mean[~ok]) | (mass[~ok] == 0))


@pytest.mark.essential
def test_dens_weighted_vphi_loss_sees_rotation():
    """Orientation fix: true rotating disk must yield large pred/target gap."""
    torch = pytest.importorskip("torch")
    from galacticsics.ml.fields.autoencoder import dens_weighted_vphi_field_loss

    b, n_z, n_mom, h, w = 1, 1, 4, 24, 24  # dens,vx,vy,vz
    dens_idx = [0]
    # histogram2d layout: first index = x, second = y
    xx_h, yy_h = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h),
        torch.linspace(-1.0, 1.0, w),
        indexing="ij",
    )
    omega = 2.0
    dens = torch.exp(-torch.sqrt(xx_h**2 + yy_h**2) / 0.5).clamp_min(0.0)
    vx = -omega * yy_h
    vy = omega * xx_h
    target = torch.stack([dens, vx, vy, torch.zeros_like(dens)], dim=0).unsqueeze(0)
    # Pred: no rotation
    pred = torch.stack(
        [dens, torch.zeros_like(dens), torch.zeros_like(dens), torch.zeros_like(dens)],
        dim=0,
    ).unsqueeze(0)
    loss = dens_weighted_vphi_field_loss(
        pred, target, dens_channel_indices=dens_idx, dens_scale=None, n_mom=n_mom
    )
    assert float(loss) > 0.1
