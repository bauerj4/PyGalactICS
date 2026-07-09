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


def compare_surface_profiles(
    initial: SurfaceDensityProfile,
    final: SurfaceDensityProfile,
    *,
    min_count: int = 1,
) -> float:
    """
    Maximum relative deviation between two midplane surface-density profiles.

    Parameters
    ----------
    initial, final : SurfaceDensityProfile
        Profiles to compare (must share compatible binning).
    min_count : int
        Minimum particle count per ring for inclusion.

    Returns
    -------
    max_rel : float
        Maximum ``|Σ_final - Σ_initial| / Σ_initial`` over valid rings.
    """
    max_rel = 0.0
    for i in range(min(len(initial.sigma), len(final.sigma))):
        if initial.counts[i] < min_count or final.counts[i] < min_count:
            continue
        ref = max(initial.sigma[i], 1e-30)
        rel = abs(final.sigma[i] - initial.sigma[i]) / ref
        max_rel = max(max_rel, rel)
    return max_rel


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


@dataclass
class DensityMap2D:
    """
    Mass surface density on a projected plane.

    Attributes
    ----------
    x_edges, y_edges : ndarray
        Bin edges along the two projected axes [kpc].
    density : ndarray, shape (nx, ny)
        Mass per unit area in each bin.
    counts : ndarray, shape (nx, ny)
        Particle count per bin.
    """

    x_edges: np.ndarray
    y_edges: np.ndarray
    density: np.ndarray
    counts: np.ndarray


def bin_plane_density(
    pos: np.ndarray,
    mass: np.ndarray,
    *,
    axes: tuple[int, int] = (0, 1),
    n_bins: int = 80,
    half_extent: float = 25.0,
    z_filter: np.ndarray | None = None,
) -> DensityMap2D:
    """
    Bin particle mass into a 2D projected surface-density map.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Particle positions [kpc].
    mass : ndarray, shape (N,)
        Particle masses.
    axes : tuple of int
        Position column indices for the horizontal and vertical image axes.
        ``(0, 1)`` is face-on (x–y); ``(0, 2)`` is edge-on (x–z).
    n_bins : int
        Number of bins along each axis.
    half_extent : float
        Half-width of the square field of view [kpc].
    z_filter : ndarray of bool, optional
        When set, include only particles where ``z_filter`` is True.

    Returns
    -------
    DensityMap2D
        Binned surface-density map.
    """
    ax0, ax1 = axes
    mask = np.ones(len(pos), dtype=bool) if z_filter is None else z_filter
    coords = pos[mask]
    weights = mass[mask]
    if coords.size == 0:
        edges = np.linspace(-half_extent, half_extent, n_bins + 1)
        empty = np.zeros((n_bins, n_bins), dtype=float)
        return DensityMap2D(
            x_edges=edges,
            y_edges=edges,
            density=empty,
            counts=empty.astype(int),
        )

    x_edges = np.linspace(-half_extent, half_extent, n_bins + 1)
    y_edges = x_edges
    counts, _, _ = np.histogram2d(
        coords[:, ax0],
        coords[:, ax1],
        bins=[x_edges, y_edges],
    )
    mass_hist, _, _ = np.histogram2d(
        coords[:, ax0],
        coords[:, ax1],
        bins=[x_edges, y_edges],
        weights=weights,
    )
    bin_area = ((x_edges[1] - x_edges[0]) ** 2)
    density = mass_hist / max(bin_area, 1e-30)
    return DensityMap2D(
        x_edges=x_edges,
        y_edges=y_edges,
        density=density,
        counts=counts.astype(int),
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
