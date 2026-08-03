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
        Binned surface-density map.  ``density[i, j]`` is the bin centred at
        ``(x_i, y_j)`` (``np.histogram2d`` layout).  For ``matplotlib.imshow``
        with ``origin='lower'`` and horizontal ``x``, pass ``density.T``.
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


def _interp_am_at_r(
    r_mid: np.ndarray,
    a_m_over_a0: np.ndarray,
    r_eval: float,
) -> tuple[float, float]:
    """
    Evaluate ``A_m(R)`` at ``r_eval`` by linear interpolation on finite rings.

    Returns ``(A_m(r_eval), r_used)``.  Falls back to the nearest finite ring
    centre when fewer than two finite samples exist.
    """
    r = np.asarray(r_mid, dtype=float)
    a = np.asarray(a_m_over_a0, dtype=float)
    ok = np.isfinite(a) & np.isfinite(r)
    if not np.any(ok):
        return float("nan"), float(r_eval)
    rr = r[ok]
    aa = a[ok]
    if rr.size == 1:
        return float(aa[0]), float(rr[0])
    order = np.argsort(rr)
    rr = rr[order]
    aa = aa[order]
    return float(np.interp(float(r_eval), rr, aa)), float(r_eval)


def disk_azimuthal_fourier(
    pos: np.ndarray,
    mass: np.ndarray,
    *,
    m: int = 2,
    n_bins: int = 15,
    r_max: float | None = None,
    z_max: float | None = 0.3,
    min_count: int = 20,
    r_eval: float | None = None,
    recenter: bool = True,
) -> dict[str, np.ndarray | float | int]:
    """
    Azimuthal Fourier amplitude ``|a_m| / a_0`` in cylindrical rings.

    For an axisymmetric disk, ``m=2`` amplitudes are noise-level
    (``∝ 1/√N`` per ring).  Elevated ``A2/A0`` indicates bars, spirals,
    or other non-axisymmetric structure.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Particle positions [kpc].
    mass : ndarray, shape (N,)
        Particle masses.
    m : int
        Azimuthal mode number (default 2).
    n_bins : int
        Number of radial annuli.
    r_max : float or None
        Maximum cylindrical radius; auto from data when ``None``.
    z_max : float or None
        Include only particles with ``|z| < z_max`` [kpc]; ``None`` uses all z.
    min_count : int
        Minimum particles per ring for inclusion in the global median.
    r_eval : float or None
        If set, also return ``a_m_over_a0_at_r`` = ``A_m`` linearly
        interpolated onto this radius (e.g. one disk scale length ``R_d``).
    recenter : bool
        If True (default), subtract the mass-weighted COM of the supplied
        particles before annular Fourier.  Callers typically pass disk-only
        particles, so this is a disk-COM recenter — required so cylindrical
        ``(R, φ)`` are measured about the disk centre rather than a
        halo-dominated / global origin offset.

    Returns
    -------
    dict
        ``r_mid``, ``a_m_over_a0`` (per ring), ``counts``,
        ``a_m_over_a0_median``, ``m``, ``recenter``, ``com``; when ``r_eval``
        is set also ``a_m_over_a0_at_r`` and ``r_eval``.
    """
    pos = np.asarray(pos, dtype=float)
    mass = np.asarray(mass, dtype=float).reshape(-1)
    com = np.zeros(3, dtype=float)
    if recenter and pos.size:
        w = mass / max(float(mass.sum()), 1e-30)
        com = (pos * w[:, None]).sum(axis=0)
        pos = pos - com
    r_cyl = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    if r_max is None:
        r_max = float(r_cyl.max()) if len(r_cyl) else 1.0
    if r_max <= 0:
        r_max = 1.0
    edges = np.linspace(0.0, r_max, n_bins + 1)
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    z_filter = np.ones(len(pos), dtype=bool)
    if z_max is not None:
        z_filter = np.abs(pos[:, 2]) < z_max
    a_m_over_a0 = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        mask = z_filter & (r_cyl >= edges[i]) & (r_cyl < edges[i + 1])
        counts[i] = int(mask.sum())
        if counts[i] == 0:
            continue
        m_ring = mass[mask]
        phi_ring = phi[mask]
        a0 = float(m_ring.sum())
        if a0 <= 0:
            continue
        am = np.abs(np.sum(m_ring * np.exp(1j * m * phi_ring))) / a0
        a_m_over_a0[i] = float(am)
    valid = (counts >= min_count) & np.isfinite(a_m_over_a0)
    median = float(np.median(a_m_over_a0[valid])) if valid.any() else float("nan")
    out: dict[str, np.ndarray | float | int] = {
        "m": m,
        "r_mid": r_mid,
        "a_m_over_a0": a_m_over_a0,
        "counts": counts,
        "a_m_over_a0_median": median,
        "recenter": bool(recenter),
        "com": com,
    }
    if r_eval is not None:
        am_r, r_used = _interp_am_at_r(r_mid, a_m_over_a0, float(r_eval))
        out["r_eval"] = float(r_used)
        out["a_m_over_a0_at_r"] = float(am_r)
    return out


def plane_density_azimuthal_fourier(
    dens: np.ndarray,
    *,
    half_extent: float,
    m: int = 2,
    n_bins: int = 12,
    r_max: float = 12.0,
    r_eval: float | None = None,
) -> dict[str, np.ndarray | float | int]:
    """
    Ring Fourier ``|a_m|/a_0`` from a face-on dens map (``histogram2d`` layout).

    ``dens[i, j]`` is the surface density in bin centred at
    ``(x_i, y_j)`` spanning ``[-half_extent, half_extent]²`` — matching
    :func:`bin_plane_density`.
    """
    dens = np.asarray(dens, dtype=float)
    if dens.ndim != 2 or dens.shape[0] != dens.shape[1]:
        raise ValueError(f"expected square dens map, got shape {dens.shape}")
    n = int(dens.shape[0])
    half = float(half_extent)
    # Centres of histogram2d bins (same edges as bin_plane_density).
    edges_xy = np.linspace(-half, half, n + 1)
    xc = 0.5 * (edges_xy[:-1] + edges_xy[1:])
    yc = xc
    xx, yy = np.meshgrid(xc, yc, indexing="ij")
    r = np.sqrt(xx * xx + yy * yy)
    phi = np.arctan2(yy, xx)
    r_max = float(r_max)
    if r_max <= 0:
        r_max = half
    edges = np.linspace(0.0, r_max, int(n_bins) + 1)
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    a_m_over_a0 = np.full(int(n_bins), np.nan, dtype=float)
    for i in range(int(n_bins)):
        mask = (r >= edges[i]) & (r < edges[i + 1])
        wgt = dens[mask]
        a0 = float(wgt.sum())
        if a0 <= 0:
            continue
        am = np.abs(np.sum(wgt * np.exp(1j * float(m) * phi[mask]))) / a0
        a_m_over_a0[i] = float(am)
    valid = np.isfinite(a_m_over_a0)
    median = float(np.median(a_m_over_a0[valid])) if valid.any() else float("nan")
    out: dict[str, np.ndarray | float | int] = {
        "m": int(m),
        "r_mid": r_mid,
        "a_m_over_a0": a_m_over_a0,
        "a_m_over_a0_median": median,
        "half_extent": half,
        "r_max": r_max,
    }
    if r_eval is not None:
        am_r, r_used = _interp_am_at_r(r_mid, a_m_over_a0, float(r_eval))
        out["r_eval"] = float(r_used)
        out["a_m_over_a0_at_r"] = float(am_r)
    return out
