"""Potential evaluation during the Python Poisson iteration."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from galacticsics.models import GalaxyModel
from galacticsics.numerics import legendre_even_l
from galacticsics.potential.evaluate import approximate_disk_potential_from_model
from galacticsics.potential.harmonics import ComponentFlags


@dataclass
class PoissonArrays:
    """
    Working harmonic arrays during the ``dbh`` Poisson iteration.

    Attributes
    ----------
    apot : ndarray, shape (n_harm, nr + 1)
        Potential multipole coefficients.
    fr : ndarray, shape (n_harm, nr + 1)
        Radial force harmonics.
    fr2 : ndarray, shape (n_harm, nr + 1)
        Second radial derivatives of the force harmonics.
    adens : ndarray, shape (n_harm, nr + 1)
        Density multipole coefficients accumulated each iteration.
    lmax : int
        Maximum harmonic degree configured on the grid.
    lmax_active : int
        Active degree during the current iteration (ramped upward).
    flags : ComponentFlags
        Enabled baryon / halo component flags.
    """

    apot: np.ndarray
    fr: np.ndarray
    fr2: np.ndarray
    adens: np.ndarray
    lmax: int
    lmax_active: int
    flags: ComponentFlags

    @property
    def n_harm(self) -> int:
        """Number of even-harmonic rows stored (``lmax // 2 + 1``)."""
        return self.lmax // 2 + 1


def _interp_row(coeff: np.ndarray, ir_hi: int, t: float) -> np.ndarray:
    """Linear interpolation of harmonic rows between radial bins."""
    ir_lo = max(ir_hi - 1, 0)
    return coeff[:, ir_hi] * t + coeff[:, ir_lo] * (1.0 - t)


def _harmonic_psi(arrays: PoissonArrays, model: GalaxyModel, s: float, z: float) -> float:
    """
    Multipole potential without the approximate disk correction.

    Parameters
    ----------
    arrays : PoissonArrays
        Current harmonic state.
    model : GalaxyModel
        Radial grid parameters.
    s, z : float
        Cylindrical coordinates [kpc].

    Returns
    -------
    float
        Harmonic potential contribution [100 km/s]\\ :sup:`2`].
    """
    r = math.hypot(s, z)
    if r == 0.0:
        return float(arrays.apot[0, 0] / math.sqrt(4.0 * math.pi))
    dr = model.grid.dr
    nr = model.grid.nr
    lmax = arrays.lmax_active
    ihi = min(max(int(r / dr) + 1, 1), nr)
    r1 = dr * (ihi - 1)
    r2 = dr * ihi
    t = (r - r1) / (r2 - r1) if r2 > r1 else 0.0
    costheta = z / r
    p_norm, _ = legendre_even_l(costheta, lmax)
    apot_i = _interp_row(arrays.apot, ihi, t)
    n_active = lmax // 2 + 1
    plcon = np.sqrt((2 * np.arange(0, lmax + 1, 2) + 1) / (4.0 * math.pi))
    return float(np.dot(p_norm[:n_active], plcon * apot_i[:n_active]))


def potential_at_batch(
    arrays: PoissonArrays,
    model: GalaxyModel,
    s: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Vectorized :func:`potential_at` for arrays of cylindrical coordinates."""
    from scipy.special import eval_legendre

    s = np.asarray(s, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.hypot(s, z)
    dr = model.grid.dr
    nr = model.grid.nr
    lmax = arrays.lmax_active
    n_active = lmax // 2 + 1
    psi = np.zeros_like(r, dtype=float)

    pos = r > 0.0
    if np.any(pos):
        rp = r[pos]
        sp = s[pos]
        zp = z[pos]
        ihi = np.clip(np.floor(rp / dr).astype(int) + 1, 1, nr)
        r1 = dr * (ihi - 1)
        r2 = dr * ihi
        t = np.where(r2 > r1, (rp - r1) / (r2 - r1), 0.0)
        costheta = zp / rp
        harmonic = np.zeros(rp.shape[0], dtype=float)
        for li, ell in enumerate(range(0, lmax + 1, 2)):
            pl = eval_legendre(ell, costheta)
            plcon = math.sqrt((2 * ell + 1) / (4.0 * math.pi))
            apot_hi = arrays.apot[li, ihi]
            apot_lo = arrays.apot[li, np.maximum(ihi - 1, 0)]
            apot_i = apot_hi * t + apot_lo * (1.0 - t)
            harmonic += plcon * pl * apot_i
        psi[pos] = harmonic

    zero = r == 0.0
    if np.any(zero):
        psi[zero] = arrays.apot[0, 0] / math.sqrt(4.0 * math.pi)

    if arrays.flags.disk and model.disk and model.disk.enabled:
        idx = np.where(pos)[0]
        for i in idx:
            psi[i] += approximate_disk_potential_from_model(model, float(s[i]), float(z[i]))
    return psi


def potential_at(
    arrays: PoissonArrays,
    model: GalaxyModel,
    s: float,
    z: float,
) -> float:
    """
    Evaluate ``pot(r, z)`` from the current harmonic state.

    When the disk flag is set, the approximate softened-disk potential from
    :func:`~galacticsics.potential.evaluate.approximate_disk_potential_from_model`
    is added to the multipole sum, matching legacy ``pot.f``.

    Parameters
    ----------
    arrays : PoissonArrays
        Working arrays for the in-progress solve.
    model : GalaxyModel
        Galaxy configuration.
    s, z : float
        Cylindrical coordinates [kpc].

    Returns
    -------
    float
        Total potential [100 km/s]\\ :sup:`2`].
    """
    psi = _harmonic_psi(arrays, model, s, z)
    if arrays.flags.disk and model.disk and model.disk.enabled:
        psi += approximate_disk_potential_from_model(model, s, z)
    return float(psi)
