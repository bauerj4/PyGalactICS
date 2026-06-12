"""
Fit axisymmetric NFW halo parameters from a particle distribution.

Used in the halo-first workflow when the dark matter component is specified
as an N-body set (e.g. ``models/MilkyWay/Xhalo``) rather than analytic
parameters in ``in.dbh``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from galacticsics.models import GalaxyModel, NFWHalo


@dataclass
class NFWFitResult:
    """
    Best-fit NFW parameters from a particle sample.

    Attributes
    ----------
    v0 : float
        Characteristic velocity scale [100 km/s] passed to ``dbh``.
    a : float
        NFW scale radius [kpc].
    r_outer : float
        Outer truncation radius ``chalo`` [kpc] (maximum particle radius).
    m_scale : float
        Mass normalization from the enclosed-mass fit [GalactICS mass units].
    rms_residual : float
        RMS fractional error of the enclosed-mass fit.
    """

    v0: float
    a: float
    r_outer: float
    m_scale: float
    rms_residual: float


def center_particles_on_core(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    mass: np.ndarray,
    *,
    core_radius: float = 10.0,
    max_iter: int = 20,
    tolerance: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Iteratively shift particles to the dense core center.

    Reimplements ``legacy/fortran/centre1.c``: within a sphere of radius
    ``core_radius`` [kpc], compute the center of mass, subtract it from all
    positions, and repeat until the offset is below ``tolerance * core_radius``.

    Parameters
    ----------
    x, y, z : ndarray
        Particle positions [kpc], shape ``(N,)``.
    mass : ndarray
        Particle masses [GalactICS units], shape ``(N,)``.
    core_radius : float, optional
        Centring sphere radius [kpc].  Default 10.
    max_iter : int, optional
        Maximum iterations (``centre1.c`` uses 20).
    tolerance : float, optional
        Convergence criterion as a fraction of ``core_radius``.

    Returns
    -------
    x, y, z : ndarray
        Centered coordinates.
    """
    x = np.asarray(x, dtype=float).copy()
    y = np.asarray(y, dtype=float).copy()
    z = np.asarray(z, dtype=float).copy()
    mass = np.asarray(mass, dtype=float)
    eps = tolerance * core_radius
    r2_limit = core_radius * core_radius

    for _ in range(max_iter):
        r2 = x * x + y * y + z * z
        mask = r2 < r2_limit
        if not np.any(mask):
            break
        mtot = mass[mask].sum()
        if mtot <= 0:
            break
        cmx = (mass[mask] * x[mask]).sum() / mtot
        cmy = (mass[mask] * y[mask]).sum() / mtot
        cmz = (mass[mask] * z[mask]).sum() / mtot
        if abs(cmx) < eps and abs(cmy) < eps and abs(cmz) < eps:
            break
        x -= cmx
        y -= cmy
        z -= cmz
    return x, y, z


def enclosed_mass_profile(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    mass: np.ndarray,
    *,
    n_bins: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Spherically averaged enclosed mass :math:`M(<r)`.

    Parameters
    ----------
    x, y, z, mass : ndarray
        Particle data [kpc, GalactICS mass units].
    n_bins : int, optional
        Number of logarithmic radial bins.

    Returns
    -------
    r : ndarray
        Bin center radii [kpc].
    m_enc : ndarray
        Cumulative mass interior to each bin edge [GalactICS units].
    """
    r = np.sqrt(x * x + y * y + z * z)
    r_max = float(np.max(r))
    if r_max <= 0:
        raise ValueError("particles have zero extent")
    edges = np.logspace(np.log10(max(r[r > 0].min(), 1e-3)), np.log10(r_max), n_bins + 1)
    m_enc = np.zeros(n_bins, dtype=float)
    centers = np.sqrt(edges[:-1] * edges[1:])
    for i in range(n_bins):
        m_enc[i] = mass[r <= edges[i + 1]].sum()
    return centers, m_enc


def _nfw_mass_enclosed(r: np.ndarray, m_scale: float, a: float) -> np.ndarray:
    """NFW enclosed mass with arbitrary normalization ``m_scale``."""
    x = r / a
    return m_scale * (np.log(1.0 + x) - x / (1.0 + x))


def estimate_nfw_from_particles(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    mass: np.ndarray,
    *,
    center: bool = True,
    core_radius: float = 10.0,
    n_bins: int = 40,
) -> NFWFitResult:
    """
    Estimate NFW ``v0`` and ``a`` from a halo particle set.

    After optional centring, bins particles in radius and fits

    .. math::

       M(<r) = M_s \\left[\\ln\\left(1+\\frac{r}{a}\\right)
       - \\frac{r/a}{1+r/a}\\right]

    to the cumulative mass profile.  The scale velocity is obtained from the
    circular velocity at :math:`r = 2a`:

    .. math::

       v_0 \\approx v_{\\mathrm{circ}}(2a)\\,
       \\left/\\sqrt{\\frac{\\ln 3}{2}}\\right.

    Parameters
    ----------
    x, y, z, mass : ndarray
        Particle phase-space positions and masses.
    center : bool, optional
        Apply :func:`center_particles_on_core` first.
    core_radius, n_bins
        Centring and binning controls.

    Returns
    -------
    NFWFitResult
        Fitted parameters for :class:`~galacticsics.models.NFWHalo`.

    See Also
    --------
    galacticsics.potential.halo_first.solve_halo_potential
    """
    if center:
        x, y, z = center_particles_on_core(x, y, z, mass, core_radius=core_radius)

    r, m_enc = enclosed_mass_profile(x, y, z, mass, n_bins=n_bins)
    valid = m_enc > 0
    r = r[valid]
    m_enc = m_enc[valid]
    if len(r) < 3:
        raise ValueError("insufficient radial bins for NFW fit")

    p0 = (float(m_enc[-1]), float(r[len(r) // 2]))
    bounds = ([0.0, r.min() * 0.5], [np.inf, r.max() * 2.0])
    coeff, _ = curve_fit(_nfw_mass_enclosed, r, m_enc, p0=p0, bounds=bounds, maxfev=10_000)
    m_scale, a = float(coeff[0]), float(coeff[1])
    model_m = _nfw_mass_enclosed(r, m_scale, a)
    rms = float(np.sqrt(np.mean(((model_m - m_enc) / m_enc) ** 2)))

    r_outer = float(np.sqrt(x * x + y * y + z * z).max())
    r_test = 2.0 * a
    m_test = float(_nfw_mass_enclosed(np.array([r_test]), m_scale, a)[0])
    v_circ = np.sqrt(m_test / r_test) if r_test > 0 else 0.0
    v0 = v_circ / np.sqrt(np.log(3.0) / 2.0) if v_circ > 0 else 1.0

    return NFWFitResult(
        v0=v0,
        a=a,
        r_outer=r_outer,
        m_scale=m_scale,
        rms_residual=rms,
    )


def apply_nfw_fit(model: GalaxyModel, fit: NFWFitResult) -> GalaxyModel:
    """
    Return a copy of *model* with halo parameters replaced by *fit*.

    Parameters
    ----------
    model : GalaxyModel
        Base model (baryon components preserved).
    fit : NFWFitResult
        Output of :func:`estimate_nfw_from_particles`.

    Returns
    -------
    GalaxyModel
        Updated halo block for :func:`~galacticsics.potential.halo_first.solve_halo_potential`.
    """
    from dataclasses import replace

    halo = model.halo or NFWHalo(r_outer=fit.r_outer, v0=fit.v0, a=fit.a)
    halo = replace(
        halo,
        r_outer=max(fit.r_outer, halo.r_outer),
        v0=fit.v0,
        a=fit.a,
        enabled=True,
    )
    return replace(model, halo=halo)
