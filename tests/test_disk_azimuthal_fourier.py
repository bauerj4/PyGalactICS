"""Tests for disk azimuthal Fourier axisymmetry diagnostic."""

from __future__ import annotations

import numpy as np

from ntropy.analysis.disk_density import disk_azimuthal_fourier


def _ring_particles(n: int, r: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    pos = np.column_stack([r * np.cos(phi), r * np.sin(phi), rng.normal(scale=0.05, size=n)])
    mass = np.full(n, 1.0 / n)
    return pos, mass


def test_axisymmetric_disk_low_m2_amplitude():
    pos, mass = _ring_particles(2000, r=8.0)
    fourier = disk_azimuthal_fourier(pos, mass, n_bins=5, r_max=12.0, z_max=1.0, min_count=50)
    assert fourier["a_m_over_a0_median"] < 0.08
    assert fourier["recenter"] is True


def test_m2_perturbation_elevated_amplitude():
    pos, mass = _ring_particles(2000, r=8.0)
    r_cyl = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    # m=2 surface-density modulation
    pos[:, 0] = r_cyl * np.cos(phi) * (1.0 + 0.25 * np.cos(2.0 * phi))
    pos[:, 1] = r_cyl * np.sin(phi) * (1.0 + 0.25 * np.cos(2.0 * phi))
    fourier = disk_azimuthal_fourier(pos, mass, n_bins=5, r_max=12.0, z_max=1.0, min_count=50)
    assert fourier["a_m_over_a0_median"] > 0.1


def test_offset_axisym_disk_needs_recenter():
    """A translated quiet disk looks barred about the wrong origin."""
    pos, mass = _ring_particles(4000, r=5.0)
    pos = pos + np.array([1.5, 0.8, 0.0])
    raw = disk_azimuthal_fourier(
        pos, mass, n_bins=8, r_max=12.0, z_max=1.0, min_count=50, recenter=False
    )
    cen = disk_azimuthal_fourier(
        pos, mass, n_bins=8, r_max=12.0, z_max=1.0, min_count=50, recenter=True
    )
    assert raw["a_m_over_a0_median"] > 0.15
    assert cen["a_m_over_a0_median"] < 0.08
    assert np.allclose(cen["com"][:2], [1.5, 0.8], atol=0.05)
