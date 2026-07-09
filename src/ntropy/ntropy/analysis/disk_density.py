"""Cylindrical surface-density analysis for disk components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ntropy.ics.disk import ExponentialDiskParams, disk_surface_density


@dataclass
class SurfaceDensityProfile:
    """
    Binned midplane surface-density profile.

    Attributes
    ----------
    r_mid : ndarray
        Ring centers [kpc].
    sigma : ndarray
        Estimated surface density per ring.
    counts : ndarray
        Particle count per ring.
    """

    r_mid: np.ndarray
    sigma: np.ndarray
    counts: np.ndarray


def bin_midplane_surface_density(
    pos: np.ndarray,
    mass: np.ndarray,
    n_bins: int = 15,
    *,
    r_max: float | None = None,
    z_max: float | None = None,
) -> SurfaceDensityProfile:
    """
    Estimate ``Σ(R)`` by binning particles in cylindrical annuli.

    When ``z_max`` is ``None`` (default), all particles in each annulus are
    used.  Because :math:`\\int \\rho(R,z)\\,dz = \\Sigma(R)` for the legacy
    disk model, the annulus mass divided by area yields the correct surface
    density.  Set ``z_max`` to restrict to a thin midplane slice.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Particle positions [kpc].
    mass : ndarray, shape (N,)
        Particle masses.
    n_bins : int
        Number of radial bins.
    r_max : float or None
        Maximum cylindrical radius.  Auto from data when ``None``.
    z_max : float or None
        If set, include only particles with ``|z| < z_max`` [kpc].

    Returns
    -------
    profile : SurfaceDensityProfile
        Binned surface-density estimate.
    """
    r_cyl = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    if r_max is None:
        r_max = float(r_cyl.max()) if len(r_cyl) else 1.0
    if r_max <= 0:
        r_max = 1.0
    edges = np.linspace(0.0, r_max, n_bins + 1)
    ring_mass = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=int)
    z_filter = np.ones(len(pos), dtype=bool)
    if z_max is not None:
        z_filter = np.abs(pos[:, 2]) < z_max
    for i in range(n_bins):
        mask = z_filter & (r_cyl >= edges[i]) & (r_cyl < edges[i + 1])
        ring_mass[i] = mass[mask].sum()
        counts[i] = int(mask.sum())
    areas = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    areas = np.maximum(areas, 1e-30)
    sigma = ring_mass / areas
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    return SurfaceDensityProfile(r_mid=r_mid, sigma=sigma, counts=counts)


def target_surface_density(
    r: np.ndarray,
    params: ExponentialDiskParams,
) -> np.ndarray:
    """
    Legacy target ``Σ(R)`` for an exponential disk component.

    Parameters
    ----------
    r : ndarray
        Cylindrical radii [kpc].
    params : ExponentialDiskParams
        Disk parameters.

    Returns
    -------
    sigma : ndarray
        Theoretical surface density.
    """
    sigma0 = params.mass / (2.0 * np.pi * params.scale_length**2)
    return disk_surface_density(
        r,
        sigma0=sigma0,
        scale_length=params.scale_length,
        outer_radius=params.outer_radius,
        trunc_width=params.trunc_width,
    )


def compare_surface_density(
    measured: SurfaceDensityProfile,
    params: ExponentialDiskParams,
    *,
    min_count: int = 3,
    skip_edges: int = 1,
) -> float:
    """
    Maximum relative deviation between measured and target ``Σ(R)``.

    Parameters
    ----------
    measured : SurfaceDensityProfile
        Binned midplane estimate from particles.
    params : ExponentialDiskParams
        Disk parameters defining the target profile.
    min_count : int
        Minimum particles per ring for comparison.
    skip_edges : int
        Exclude this many innermost and outermost rings (noisy for small N).

    Returns
    -------
    max_rel : float
        Maximum relative error over valid rings.
    """
    target = target_surface_density(measured.r_mid, params)
    max_rel = 0.0
    n = len(measured.r_mid)
    for i in range(skip_edges, max(n - skip_edges, skip_edges)):
        if measured.counts[i] < min_count:
            continue
        ref = max(target[i], 1e-30)
        rel = abs(measured.sigma[i] - target[i]) / ref
        max_rel = max(max_rel, rel)
    return max_rel
