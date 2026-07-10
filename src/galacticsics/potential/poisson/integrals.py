"""Harmonic synthesis integrals (``dbh.f`` / ``halopotential.f``)."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy import integrate
from scipy.special import eval_legendre

from galacticsics.numerics import quadrature_node_count, simpson_on_grid


def _simpson_quadrature_nodes(ntheta: int) -> tuple[np.ndarray, np.ndarray]:
    """Cos(theta) nodes and Simpson weights on [0, 1] (legacy ``polardens`` layout)."""
    dctheta = 1.0 / ntheta
    ctheta = np.array([1.0, 0.0], dtype=float)
    weights = np.array([1.0, 1.0], dtype=float)
    for is_ in range(1, ntheta - 1, 2):
        ctheta = np.append(ctheta, is_ * dctheta)
        weights = np.append(weights, 4.0)
    for is_ in range(2, ntheta - 2, 2):
        ctheta = np.append(ctheta, is_ * dctheta)
        weights = np.append(weights, 2.0)
    return ctheta, weights


def _polar_cos_nodes(ntheta: int) -> np.ndarray:
    """Uniform cos(theta) nodes on [0, 1] with odd count for SciPy Simpson."""
    return np.linspace(0.0, 1.0, quadrature_node_count(ntheta))


def _integrate_polar_legendre(
    r: float,
    ell: int,
    ctheta: np.ndarray,
    rho: np.ndarray,
) -> float:
    """Quadrature of rho(cos theta) * Y_l(cos theta) over a polar quadrant."""
    pl = eval_legendre(ell, ctheta)
    plcon = math.sqrt((2 * ell + 1) / (4.0 * math.pi))
    return float(plcon * simpson_on_grid(rho * pl, ctheta) * 4.0 * math.pi)


def _pad_to_even_shells(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Pad harmonic rows to an even shell count for legacy radial Simpson."""
    nr = values.shape[1] - 1
    if nr % 2 == 0:
        return values, nr
    padded = np.zeros((values.shape[0], nr + 2), dtype=float)
    padded[:, : nr + 1] = values
    padded[:, nr + 1] = values[:, nr]
    return padded, nr + 1


def integrate_polar_density_at_shell(
    arrays,
    model,
    r: float,
    ell: int,
    ntheta: int,
    *,
    dens_psi_halo,
    dens_psi_bulge=None,
    psic: float,
    halo_psi_tables: tuple[np.ndarray, np.ndarray, float] | None = None,
    bulge_psi_tables: tuple[np.ndarray, np.ndarray, float, float] | None = None,
) -> float:
    """
    Synthesize one density harmonic at cylindrical radius ``r``.

    Integrates ``rho(s, z) * Y_l(cos theta)`` over a polar quadrant using
    Simpson quadrature on ``cos(theta) in [0, 1]``.

    Parameters
    ----------
    arrays : PoissonArrays
        Current harmonic state during the Poisson iteration.
    model : GalaxyModel
        Galaxy with any enabled combination of halo, disk, and bulge.
    r : float
        Cylindrical radius of the integration shell [kpc].
    ell : int
        Even spherical-harmonic degree (``0, 2, 4, ...``).
    ntheta : int
        Number of polar quadrature nodes.
    dens_psi_halo, dens_psi_bulge : callable or None
        Scalar ``rho(psi)`` interpolators from DF tables.
    psic : float
        DF cutoff potential [100 km/s]\\ :sup:`2`].
    halo_psi_tables : tuple of ndarray, optional
        ``(energies, dens_psi, psi0)`` for vectorized halo lookup.
    bulge_psi_tables : tuple, optional
        ``(energies, dens_psi, psi0, psid)`` for vectorized bulge lookup.

    Returns
    -------
    harmonic_moment : float
        Harmonic density moment :math:`\\int \\rho Y_\\ell \\, d\\Omega`.
    """
    from galacticsics.potential.poisson.densities import total_density_harmonic_batch
    from galacticsics.potential.poisson.potential import potential_at_batch

    if r <= 0.0:
        return 0.0
    ctheta = _polar_cos_nodes(ntheta)
    z = r * ctheta
    s = r * np.sqrt(np.maximum(0.0, 1.0 - ctheta * ctheta))
    zd = model.disk.scale_height if model.disk else 1.0
    psi = potential_at_batch(arrays, model, s, z)
    psi_mid = potential_at_batch(arrays, model, s, np.zeros_like(s))
    psi_3zd = potential_at_batch(arrays, model, s, np.full_like(s, 3.0 * zd))
    rho = total_density_harmonic_batch(
        s,
        z,
        psi,
        psi_mid,
        psi_3zd,
        model,
        dens_psi_halo=dens_psi_halo,
        dens_psi_bulge=dens_psi_bulge,
        psic=psic,
        halo_psi_tables=halo_psi_tables,
        bulge_psi_tables=bulge_psi_tables,
    )
    return _integrate_polar_legendre(r, ell, ctheta, rho)


def integrate_polar_density(
    r: float,
    ntheta: int,
    ell: int,
    dens_fn: Callable[[float, float], float],
) -> float:
    """Integrate ``dens(s,z) * Y_l`` over a quadrant for one even harmonic ``ell``."""
    if r <= 0.0:
        return 0.0
    ctheta = _polar_cos_nodes(ntheta)
    z = r * ctheta
    s = r * np.sqrt(np.maximum(0.0, 1.0 - ctheta * ctheta))
    rho = np.fromiter((dens_fn(float(si), float(zi)) for si, zi in zip(s, z)), dtype=float, count=len(s))
    return _integrate_polar_legendre(r, ell, ctheta, rho)


def integrate_polar_density_spherical(
    r: float,
    ntheta: int,
    ell: int,
    rho_fn: Callable[[float], float],
    *,
    rho_array_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> float:
    """
    Polar synthesis when density depends only on spherical radius.

    Parameters
    ----------
    r : float
        Cylindrical shell radius [kpc].
    ntheta : int
        Number of polar quadrature nodes.
    ell : int
        Even spherical-harmonic degree.
    rho_fn : callable
        Scalar density ``rho(r_spherical)``.
    rho_array_fn : callable, optional
        Vectorized density evaluator; used when provided for speed.

    Returns
    -------
    harmonic_moment : float
        Harmonic density moment on the shell.
    """
    if r <= 0.0:
        return 0.0
    ctheta = _polar_cos_nodes(ntheta)
    s = r * np.sqrt(np.maximum(0.0, 1.0 - ctheta * ctheta))
    z = r * ctheta
    rad = np.hypot(s, z)
    if rho_array_fn is not None:
        rho = rho_array_fn(rad)
    else:
        rho = np.fromiter((rho_fn(float(ri)) for ri in rad), dtype=float, count=len(rad))
    return _integrate_polar_legendre(r, ell, ctheta, rho)


def integrated_disk_density_on_cylinder(r: float, model, *, ntheta: int = 32) -> float:
    """
    Azimuthally integrated legacy ``diskdensestimate`` at cylindrical radius ``r``.

    Delegates to :func:`~galacticsics.potential.poisson.appdisk.integrated_disk_densestimate_on_shell`
    (``diskpotentialestimate`` quadrature). This is **not** ``appdiskdens``.

    Parameters
    ----------
    r : float
        Shell radius [kpc].
    model : GalaxyModel
        Galaxy with an enabled disk.
    ntheta : int, optional
        Polar quadrature count.

    Returns
    -------
    float
        Shell-averaged disk density estimate.
    """
    from galacticsics.potential.poisson.appdisk import integrated_disk_densestimate_on_shell

    return integrated_disk_densestimate_on_shell(r, model, ntheta=ntheta)


def poisson_harmonics_from_density(
    adens: np.ndarray,
    *,
    dr: float,
    nr: int,
    lmax: int,
    apot_in: np.ndarray | None = None,
    frac: float = 0.75,
    lmax_old: int = -2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert density harmonics to potential / force harmonics (BT eq. 2-208).

    Parameters
    ----------
    adens : ndarray, shape (n_harmonic_rows, nr + 1)
        Density multipole coefficients on the radial grid.
    dr : float
        Radial grid spacing [kpc].
    nr : int
        Number of radial shells.
    lmax : int
        Maximum even harmonic degree for this synthesis pass.
    apot_in : ndarray, optional
        Previous potential harmonics for under-relaxation when ``lmax`` increases.
    frac : float, optional
        Relaxation factor (default ``0.75``): new = frac * old + (1-frac) * raw.
    lmax_old : int, optional
        Previously active ``lmax``; degrees above this are fully replaced.

    Returns
    -------
    apot : ndarray, shape (n_harmonic_rows, nr + 1)
        Potential multipole coefficients.
    fr : ndarray, shape (n_harmonic_rows, nr + 1)
        Radial force harmonics.
    fr2 : ndarray, shape (n_harmonic_rows, nr + 1)
        Second radial derivatives of the force harmonics.
    """
    adens_work, nr_work = _pad_to_even_shells(adens)
    apot_in_work = None
    if apot_in is not None:
        apot_in_work, _ = _pad_to_even_shells(apot_in)
    apot, fr, fr2 = _poisson_harmonics_even_shells(
        adens_work,
        dr=dr,
        nr=nr_work,
        lmax=lmax,
        apot_in=apot_in_work,
        frac=frac,
        lmax_old=lmax_old,
    )
    if nr_work != nr:
        apot = apot[:, : nr + 1]
        fr = fr[:, : nr + 1]
        fr2 = fr2[:, : nr + 1]
    return apot, fr, fr2


def _poisson_harmonics_even_shells(
    adens: np.ndarray,
    *,
    dr: float,
    nr: int,
    lmax: int,
    apot_in: np.ndarray | None = None,
    frac: float = 0.75,
    lmax_old: int = -2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Radial BT synthesis requiring an even shell count."""
    if nr % 2 != 0:
        raise ValueError("even shell count required for legacy radial Simpson march")
    n_harm = lmax // 2 + 1
    if apot_in is not None:
        apot = apot_in.copy()
        fr = np.zeros((n_harm, nr + 1), dtype=float)
        fr2 = np.zeros((n_harm, nr + 1), dtype=float)
    else:
        apot = np.zeros((n_harm, nr + 1), dtype=float)
        fr = np.zeros((n_harm, nr + 1), dtype=float)
        fr2 = np.zeros((n_harm, nr + 1), dtype=float)

    for li, ell in enumerate(range(0, lmax + 1, 2)):
        s1 = np.zeros(nr + 1, dtype=float)
        r = 2 * dr
        if nr >= 2:
            s1[2] = (r * dr / 3.0) * (
                4 * adens[li, 1] * (1.0 - dr / r) ** (ell + 2) + adens[li, 2]
            )
        rold = r
        for ir in range(4, nr + 1, 2):
            r = ir * dr
            s1a = (r * dr / 3.0) * (
                adens[li, ir - 2] * (1.0 - 2 * dr / r) ** (ell + 2)
                + 4 * adens[li, ir - 1] * (1.0 - dr / r) ** (ell + 2)
                + adens[li, ir]
            )
            s1[ir] = s1a + s1[ir - 2] * (rold / r) ** (ell + 1)
            rold = r

        s2 = np.zeros(nr + 1, dtype=float)
        rold = nr * dr
        for ir in range(nr - 2, 1, -2):
            r = ir * dr
            s2a = (r * dr / 3.0) * (
                adens[li, ir + 2] * (1.0 + 2 * dr / r) ** (1 - ell)
                + 4 * adens[li, ir + 1] * (1.0 + dr / r) ** (1 - ell)
                + adens[li, ir]
            )
            s2[ir] = s2a + s2[ir + 2] * (r / rold) ** ell
            rold = r

        for ir in range(2, nr + 1, 2):
            r = ir * dr
            raw = 4 * math.pi / (2 * ell + 1) * (s1[ir] + s2[ir])
            if ell <= lmax_old:
                apot[li, ir] = frac * apot[li, ir] + (1.0 - frac) * raw
            else:
                apot[li, ir] = raw
            fr[li, ir] = -4 * math.pi / (2 * ell + 1) * (-(ell + 1) * s1[ir] + ell * s2[ir]) / r
            fr2[li, ir] = -4 * math.pi / (2 * ell + 1) * (
                (ell + 1) * (ell + 2) * s1[ir] / r**2
                + ell * (ell - 1) * s2[ir] / r**2
                - (2 * ell + 1) * adens[li, ir]
            )

    apot[0, 0] = 3 * (apot[0, 2] - apot[0, 4]) + apot[0, 6] if nr >= 6 else apot[0, 0]
    fr2[0, 0] = 2 * fr2[0, 2] - fr2[0, 4] if nr >= 4 else 0.0
    for li in range(1, n_harm):
        apot[li, 0] = 0.0
        fr[li, 0] = 0.0
        fr2[li, 0] = 0.0

    odd = np.arange(1, nr, 2)
    if odd.size:
        apot[:, odd] = 0.5 * (apot[:, odd - 1] + apot[:, odd + 1])
        fr[:, odd] = 0.5 * (fr[:, odd - 1] + fr[:, odd + 1])
        fr2[:, odd] = 0.5 * (fr2[:, odd - 1] + fr2[:, odd + 1])

    return apot, fr, fr2


def monopole_estimate_from_spherical_density(
    rho: np.ndarray,
    *,
    dr: float,
    nr: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Halo/disk potential estimate on radial grid (``halopotentialestimate``)."""
    rho_work = rho
    nr_work = nr
    if nr % 2 != 0:
        rho_work = np.empty(nr + 2, dtype=float)
        rho_work[: nr + 1] = rho
        rho_work[nr + 1] = rho[nr]
        nr_work = nr + 1
    hpot, hfr = _monopole_estimate_even_shells(rho_work, dr=dr, nr=nr_work)
    if nr_work != nr:
        hpot = hpot[: nr + 1]
        hfr = hfr[: nr + 1]
    return hpot, hfr


def _monopole_estimate_even_shells(
    rho: np.ndarray,
    *,
    dr: float,
    nr: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Monopole estimate with legacy even-index Simpson (requires even ``nr``)."""
    if nr % 2 != 0:
        raise ValueError("even shell count required for legacy radial Simpson march")
    hpot = np.zeros(nr + 1, dtype=float)
    hfr = np.zeros(nr + 1, dtype=float)
    s1 = np.zeros(nr + 1, dtype=float)
    r = 2 * dr
    if nr >= 2:
        s1[2] = (r * dr / 3.0) * (4 * rho[1] * (1.0 - dr / r) ** 2 + rho[2])
    rold = r
    for ir in range(4, nr + 1, 2):
        r = ir * dr
        s1a = (r * dr / 3.0) * (
            rho[ir - 2] * (1.0 - 2 * dr / r) ** 2
            + 4 * rho[ir - 1] * (1.0 - dr / r) ** 2
            + rho[ir]
        )
        s1[ir] = s1a + s1[ir - 2] * rold / r
        rold = r
    s2 = np.zeros(nr + 1, dtype=float)
    rold = nr * dr
    for ir in range(nr - 2, 1, -2):
        r = ir * dr
        s2a = (r * dr / 3.0) * (
            rho[ir + 2] * (1.0 + 2 * dr / r)
            + 4 * rho[ir + 1] * (1.0 + dr / r)
            + rho[ir]
        )
        s2[ir] = s2a + s2[ir + 2]
        rold = r
    even = np.arange(2, nr + 1, 2)
    hpot[even] = 4 * math.pi * (s1[even] + s2[even])
    hfr[even] = -(4 * math.pi) * s1[even] / even / dr
    if nr >= 6:
        hpot[0] = 3 * (hpot[2] - hpot[4]) + hpot[6]
    odd = np.arange(1, nr, 2)
    if odd.size:
        hpot[odd] = 0.5 * (hpot[odd - 1] + hpot[odd + 1])
        hfr[odd] = 0.5 * (hfr[odd - 1] + hfr[odd + 1])
    return hpot, hfr
