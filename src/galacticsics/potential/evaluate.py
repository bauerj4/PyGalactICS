"""
Potential and force evaluation using SciPy Legendre functions.

Implements the multipole evaluation logic of ``legacy/fortran/pot.f`` and
``legacy/fortran/force.f`` in pure Python, using
:func:`~scipy.special.eval_legendre` instead of ``plgndr1.f``.

Approximate disk, second-disk, and gas contributions use the same analytic
prescriptions as ``appdiskpot.f`` / ``appdiskpot2.f`` (exponential/sech²).

Notes
-----
Forces from the approximate disk are obtained by numerical differentiation of
the disk potential contribution; harmonic forces use the Binney & Tremaine
multipole formulae identical to ``force.f``.

Examples
--------
>>> from galacticsics.io import read_harmonic_potential
>>> from galacticsics.potential.evaluate import evaluate_potential
>>> pot = read_harmonic_potential("models/MilkyWay/dbh.dat")
>>> evaluate_potential(pot, 8.0, 0.0) < 0
True
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from galacticsics.numerics import legendre_even_l
from galacticsics.potential.harmonics import HarmonicPotential


def _erfc_approx(x: float) -> float:
    """Complementary error function (math.erfc)."""
    return math.erfc(x)


def _disk_surface_density(r: float, disk) -> float:
    """Truncated exponential disk surface density."""
    if disk is None or not disk.enabled or r <= 0:
        return 0.0
    t = (r - disk.outer_radius) / disk.trunc_width
    if t < -4.0:
        trunc = 1.0
    elif t > 4.0:
        return 0.0
    else:
        trunc = 0.5 * _erfc_approx(t)
    return disk.disk_const * math.exp(-r / disk.scale_length) * trunc


def _disk_vertical_factor(z: float, zdisk: float) -> float:
    """Sech^2 vertical factor g(z) used in appdiskpot."""
    zz = abs(z) / zdisk
    return 0.5 / zdisk / math.cosh(zz) ** 2


def approximate_disk_potential(potential: HarmonicPotential, s: float, z: float) -> float:
    """Approximate stellar disk potential (appdiskpot.f)."""
    disk = potential.model.disk
    if disk is None or not disk.enabled:
        return 0.0
    r = math.hypot(s, z)
    f = _disk_surface_density(r, disk)
    if f == 0.0:
        return 0.0
    g = _disk_vertical_factor(z, disk.scale_height)
    return -4.0 * math.pi * f * disk.scale_height * g / 2.0


def approximate_disk2_potential(potential: HarmonicPotential, s: float, z: float) -> float:
    """Second disk component (appdiskpot2.f)."""
    disk = potential.model.disk2
    if disk is None or not disk.enabled:
        return 0.0
    r = math.hypot(s, z)
    t = math.sqrt(0.5) * (r - disk.outer_radius) / disk.trunc_width
    if t < -4.0:
        eerfc = 1.0
    elif t > 4.0:
        return 0.0
    else:
        eerfc = 0.5 * _erfc_approx(t)
    f = disk.disk_const * math.exp(-r / disk.scale_length)
    zz = abs(z / disk.scale_height)
    return -4.0 * math.pi * f * disk.scale_height**2 * (zz + math.log(0.5 * (1.0 + math.exp(-2.0 * zz)))) * eerfc


def _interpolate_harmonic(
    coeff: np.ndarray, ir_hi: int, t: float, n_harm: int
) -> np.ndarray:
    """Linear interpolation in radius for each harmonic coefficient."""
    ir_lo = max(ir_hi - 1, 0)
    return coeff[:n_harm, ir_hi] * t + coeff[:n_harm, ir_lo] * (1.0 - t)


def evaluate_potential_harmonic_only(potential: HarmonicPotential, s: float, z: float) -> float:
    """Multipole harmonic contribution to potential (pot.f, without approximate disk)."""
    r = math.hypot(s, z)
    if r == 0.0:
        return float(potential.apot[0, 0] / math.sqrt(4.0 * math.pi))

    dr = potential.dr
    nr = potential.nr
    lmax = potential.lmax
    ihi = min(max(int(r / dr) + 1, 1), nr)
    r1 = dr * (ihi - 1)
    r2 = dr * ihi
    t = (r - r1) / (r2 - r1) if r2 > r1 else 0.0
    costheta = z / r
    lmaxx = 0 if r == 0.0 else lmax

    p_norm, _ = legendre_even_l(costheta, lmaxx)
    apot_interp = _interpolate_harmonic(potential.apot, ihi, t, potential.n_harmonics)

    psi = 0.0
    for i, ell in enumerate(range(0, lmaxx + 1, 2)):
        psi += p_norm[i] * potential.plcon(ell) * apot_interp[i]
    return float(psi)


def evaluate_potential(potential: HarmonicPotential, s: float, z: float) -> float:
    """Total potential Psi(R,z) including approximate disk components."""
    psi = evaluate_potential_harmonic_only(potential, s, z)
    if potential.flags.disk:
        psi += approximate_disk_potential(potential, s, z)
    if potential.flags.disk2:
        psi += approximate_disk2_potential(potential, s, z)
    return psi


def evaluate_force(potential: HarmonicPotential, s: float, z: float) -> Tuple[float, float, float]:
    """Cylindrical force components (F_R, F_z) and potential.

    Returns (F_R, F_z, Psi) in GalactICS units. Matches force.f structure.
    """
    r = math.hypot(s, z)
    if r == 0.0:
        psi = evaluate_potential(potential, s, z)
        return 0.0, 0.0, psi

    dr = potential.dr
    nr = potential.nr
    lmax = potential.lmax
    redge = potential.r_edge
    ihi = min(max(int(r / dr) + 1, 1), nr)
    r1 = dr * (ihi - 1)
    r2 = dr * ihi
    t = (r - r1) / (r2 - r1) if r2 > r1 else 0.0
    tm1 = 1.0 - t
    costheta = z / r
    sintheta = s / r

    p_vals, dp_dtheta = legendre_even_l(costheta, lmax)
    pc = np.array([potential.plcon(2 * i) for i in range(potential.n_harmonics)])
    p = p_vals * pc
    dp = dp_dtheta * pc

    if r <= redge:
        ihim1 = max(ihi - 1, 0)
        apot_hi = potential.apot[: potential.n_harmonics, ihi]
        apot_lo = potential.apot[: potential.n_harmonics, ihim1]
        fr_hi = potential.fr[: potential.n_harmonics, ihi]
        fr_lo = potential.fr[: potential.n_harmonics, ihim1]
        apot_interp = t * apot_hi + tm1 * apot_lo
        fr_interp = t * fr_hi + tm1 * fr_lo

        frr = float(np.sum(p * fr_interp))
        fth = 0.0
        for i in range(1, potential.n_harmonics):
            fth -= sintheta * dp[i] * apot_interp[i]
        psi = float(np.sum(p * apot_interp))
    else:
        frr = 0.0
        fth = 0.0
        psi = 0.0
        for i in range(potential.n_harmonics):
            ell = 2 * i
            frr += -(ell + 1) * p[i] * potential.apot[i, nr] / redge * (redge / r) ** (ell + 2)
            psi += p[i] * potential.apot[i, nr] * (redge / r) ** (ell + 1)
        for i in range(1, potential.n_harmonics):
            ell = 2 * i
            fth -= sintheta * dp[i] * potential.apot[i, nr] * (redge / r) ** (ell + 1)

    psi_harm = psi
    psi_disk = 0.0
    if potential.flags.disk:
        psi_disk += approximate_disk_potential(potential, s, z)
    if potential.flags.disk2:
        psi_disk += approximate_disk2_potential(potential, s, z)
    psi = psi_harm + psi_disk

    fth = -fth
    fs = -(sintheta * frr + costheta / r * fth)
    fz = -(costheta * frr - sintheta / r * fth)

    if psi_disk != 0.0 or potential.flags.disk or potential.flags.disk2:
        eps = 1e-5
        d_disk = lambda rs, zs: (
            (approximate_disk_potential(potential, rs, zs) if potential.flags.disk else 0.0)
            + (approximate_disk2_potential(potential, rs, zs) if potential.flags.disk2 else 0.0)
        )
        fs -= (d_disk(s + eps, z) - d_disk(s - eps, z)) / (2.0 * eps)
        fz -= (d_disk(s, z + eps) - d_disk(s, z - eps)) / (2.0 * eps)

    return float(fs), float(fz), float(psi)
