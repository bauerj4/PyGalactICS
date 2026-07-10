"""Spherical Sersic bulge profiles (``sersicprofiles.f``)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammainc, gammaincc, gammaln, gamma as gamma_fn

from galacticsics.models import SersicBulge


@dataclass(frozen=True)
class SersicParams:
    """
    Cached Sersic normalization parameters.

    Attributes
    ----------
    n : float
        Sersic index ``nnn``.
    ppp : float
        Inner slope parameter.
    Re : float
        Effective radius ``abulge`` [kpc].
    v0 : float
        Bulge potential scale [100 km/s].
    butt : float
        Dimensionless truncation parameter from ``setsersicparameters``.
    rho0 : float
        Central density normalization ``Rho0`` [mass kpc\\ :sup:`-3`].
    """

    n: float
    ppp: float
    Re: float
    v0: float
    butt: float
    rho0: float


def sersic_params_from_bulge(bulge: SersicBulge) -> SersicParams:
    """
    Compute legacy Sersic constants for a :class:`~galacticsics.models.SersicBulge`.

    Parameters
    ----------
    bulge : SersicBulge
        Enabled bulge component.

    Returns
    -------
    SersicParams
        Normalization used by :func:`sersic_density` and related functions.
    """
    nnn = bulge.n_sersic
    ppp = bulge.ppp
    re = bulge.a
    v0 = bulge.v0
    if nnn > 0.50:
        butt1 = 0.6 * (2.0 * nnn)
    else:
        butt1 = 1.0e-4
    butt2 = 1.20 * (2.0 * nnn)
    butt = float(brentq(lambda b: gammainc(2.0 * nnn, b) - 0.5, butt1, butt2, xtol=1.0e-4))
    rho0 = (v0**2) / (
        4.0
        * math.pi
        * re
        * re
        * nnn
        * (butt ** (nnn * (ppp - 2.0)))
        * math.exp(gammaln(nnn * (2.0 - ppp)))
    )
    return SersicParams(n=nnn, ppp=ppp, Re=re, v0=v0, butt=butt, rho0=rho0)


def sersic_density(r: float, params: SersicParams) -> float:
    """
    Three-dimensional Sersic density at spherical radius ``r``.

    Parameters
    ----------
    r : float
        Spherical radius [kpc].
    params : SersicParams
        Cached normalization from :func:`sersic_params_from_bulge`.

    Returns
    -------
    float
        Mass density [GalactICS units kpc\\ :sup:`-3`].
    """
    if r <= 0.0:
        return 0.0
    u = r / params.Re
    un = u ** (1.0 / params.n)
    return params.rho0 * (u ** (-params.ppp)) * math.exp(-params.butt * un)


def sersic_density_prime(r: float, params: SersicParams) -> float:
    """Radial derivative ``d rho / d r`` of :func:`sersic_density`."""
    if r <= 0.0:
        return 0.0
    u = r / params.Re
    un = u ** (1.0 / params.n)
    rho = sersic_density(r, params)
    return -rho / params.Re * (params.ppp * params.n + params.butt * un) / params.n / u


def sersic_density_2prime(r: float, params: SersicParams) -> float:
    """Second radial derivative ``d\\ :sup:`2` rho / d r\\ :sup:`2` ``."""
    if r <= 0.0:
        return 0.0
    u = r / params.Re
    un = u ** (1.0 / params.n)
    rho = sersic_density(r, params)
    num = (
        (params.ppp * params.n) ** 2
        + params.ppp * params.n * params.n
        + 2.0 * params.ppp * params.butt * params.n * un
        + params.butt * un * (params.n - 1.0)
        + (params.butt * un) ** 2
    )
    return rho / (params.Re**2) * num / ((params.n * u) ** 2)


def sersic_force(r: float, params: SersicParams) -> float:
    """
    Spherical gravitational force magnitude ``|d Psi / d r|`` from the bulge.

    Returns
    -------
    float
        Force contribution [100 km/s]\\ :sup:`2` kpc\\ :sup:`-1`], negative inward.
    """
    if r <= 0.0:
        return 0.0
    u = r / params.Re
    un = u ** (1.0 / params.n)
    aaa = params.n * (3.0 - params.ppp)
    l2 = params.rho0 * (params.Re**3) * params.n * (params.butt ** (-aaa)) * math.exp(gammaln(aaa))
    arg = params.butt * un
    if aaa + 1.0 > arg:
        l2 *= gamma_fn(aaa) * gammainc(aaa, arg)
    else:
        l2 *= gamma_fn(aaa) * gammaincc(aaa, arg)
    return -4.0 * math.pi * l2 / (r * r)


def sersic_d2rho_dpsi2(
    r: float,
    params: SersicParams,
    *,
    total_density: float,
    total_force: float,
) -> float:
    """
    ``d\\ :sup:`2` rho_bulge / d Psi\\ :sup:`2` `` for Eddington DF inversion.

    Parameters
    ----------
    r : float
        Radius where potential equals the tabulated energy.
    params : SersicParams
        Bulge profile constants.
    total_density : float
        Combined spherical density at ``r`` (all enabled components).
    total_force : float
        Combined inward force magnitude at ``r``.

    Returns
    -------
    float
        Second derivative used in :func:`~galacticsics.potential.poisson.df_tables.compute_sersic_df_table`.
    """
    if total_force == 0.0:
        return 0.0
    den = sersic_density(r, params)
    denp = sersic_density_prime(r, params)
    denpp = sersic_density_2prime(r, params)
    bbb = 4.0 * math.pi * total_density * denp / total_force
    ccc = 2.0 * denp / r
    ddd = denpp
    return (bbb + ccc + ddd) / (total_force**2)


def bulge_density_spherical(r: float, bulge: SersicBulge, params: SersicParams | None = None) -> float:
    """Alias for :func:`sersic_density` using a :class:`~galacticsics.models.SersicBulge`."""
    if not bulge.enabled or r <= 0.0:
        return 0.0
    p = params or sersic_params_from_bulge(bulge)
    return sersic_density(r, p)


def bulge_mass_spherical(radii: np.ndarray, bulge: SersicBulge, dr: float) -> float:
    """
    Estimate bulge mass from a spherical density tabulation.

    Parameters
    ----------
    radii : ndarray, shape (nr + 1,)
        Radial grid [kpc].
    bulge : SersicBulge
        Bulge model.
    dr : float
        Radial step [kpc].

    Returns
    -------
    float
        Integrated mass [GalactICS units].
    """
    params = sersic_params_from_bulge(bulge)
    rho = np.array([bulge_density_spherical(max(float(r), 1e-6), bulge, params) for r in radii[1:]])
    r = radii[1:]
    return float(4.0 * math.pi * np.sum(rho * r * r) * dr)
