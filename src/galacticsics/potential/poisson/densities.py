"""
Component density models for the Python Poisson solver.

Densities follow the legacy Fortran routines in ``legacy/fortran/``:

* halo — ``halodenspsi.f``, ``nfwprofiles.f``
* disk — ``diskdens.f``, ``appdiskdens.f``, ``diskdensestimate`` in ``dpolardens.f``
* bulge — ``bulgedenspsi.f``, ``sersicprofiles.f``

Approximate disk potential subtraction (``totdens - appdisk``)
----------------------------------------------------------------
During harmonic synthesis (:func:`total_density_harmonic`), the Poisson
iteration adds the self-consistent disk density from the solved potential
(:func:`disk_density_psi`, legacy ``diskdens``) and **subtracts** the
analytic approximate disk density (:func:`disk_density_estimate`, legacy
``appdiskdens``).  This matches ``polardens.f`` / ``totdens.f``.

The approximate disk potential is already folded into the multipole sum via
:func:`~galacticsics.potential.evaluate.approximate_disk_potential` (legacy
``appdiskpot``).  Without the subtraction, the disk would be double-counted:
once through the approximate potential harmonics and again through
``diskdens``.  The residual multipoles then represent departures of the true
disk from the softened ``log(cosh)`` vertical ansatz.

**Important:** monopole seeding uses a *different* estimate,
``diskdensestimate`` integrated by ``diskpotentialestimate`` — not
``appdiskdens``.  See :mod:`galacticsics.potential.poisson.appdisk`.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.special import erfc

from galacticsics.models import GalaxyModel, NFWHalo, SersicBulge
from galacticsics.numerics import sech2_stable
from galacticsics.potential.poisson.sersic import (
    SersicParams,
    bulge_density_spherical,
    sersic_params_from_bulge,
)


def _truncated_exponential(
    r: float,
    sigma0: float,
    scale_length: float,
    outer_radius: float,
    trunc_width: float,
) -> float:
    """Exponentially declining surface density with complementary error-function truncation."""
    if r <= 0.0:
        return 0.0
    t = (r - outer_radius) / trunc_width
    if t < -4.0:
        trunc = 1.0
    elif t > 4.0:
        return 0.0
    else:
        trunc = 0.5 * math.erfc(t)
    return sigma0 * math.exp(-r / scale_length) * trunc


def _halo_density_normalization(halo: NFWHalo) -> float:
    """NFW amplitude ``rho_0`` from ``v0``, ``a``, and ``cusp`` (legacy ``haloconst``)."""
    return (2.0 ** (1.0 - halo.cusp)) * halo.v0**2 / (4.0 * math.pi * halo.a**2)


def nfw_density(r: float, halo: NFWHalo) -> float:
    """
    Untruncated NFW density at radius ``r``.

    Parameters
    ----------
    r : float
        Spherical radius [kpc].
    halo : NFWHalo
        Halo parameters.

    Returns
    -------
    float
        Mass density [GalactICS units kpc\\ :sup:`-3`].
    """
    if r <= 0.0:
        return 0.0
    s = r / halo.a
    cusp = halo.cusp
    rho0 = _halo_density_normalization(halo)
    return rho0 / (s**cusp) / ((1.0 + s) ** (3.0 - cusp))


def nfw_density_array(r: np.ndarray, halo: NFWHalo) -> np.ndarray:
    """
    Vectorized untruncated NFW density on a radial grid.

    Parameters
    ----------
    r : array_like, shape (N,)
        Spherical radii [kpc].  Non-positive entries evaluate to zero.
    halo : NFWHalo
        Halo parameters.

    Returns
    -------
    rho : ndarray, shape (N,)
        Mass density [GalactICS units kpc\\ :sup:`-3`].
    """
    radii = np.asarray(r, dtype=float)
    rho = np.zeros_like(radii, dtype=float)
    positive = radii > 0.0
    if not np.any(positive):
        return rho
    rp = radii[positive]
    s = rp / halo.a
    cusp = halo.cusp
    rho0 = _halo_density_normalization(halo)
    rho[positive] = rho0 / (s**cusp) / ((1.0 + s) ** (3.0 - cusp))
    return rho


def halo_trunc_factor(r: float, halo: NFWHalo) -> float:
    """
    Outer halo truncation factor in ``(0, 1]``.

    Parameters
    ----------
    r : float
        Spherical radius [kpc].
    halo : NFWHalo
        Halo truncation parameters ``r_outer`` and ``dr_trunc``.

    Returns
    -------
    float
        Complementary error-function taper applied to the NFW profile.
    """
    t = math.sqrt(0.5) * (r - halo.r_outer) / halo.dr_trunc
    if t < -4.0:
        return 1.0
    if t > 4.0:
        return 0.0
    return 0.5 * math.erfc(t)


def halo_trunc_factor_array(r: np.ndarray, halo: NFWHalo) -> np.ndarray:
    """
    Vectorized outer halo truncation factor.

    Parameters
    ----------
    r : array_like, shape (N,)
        Spherical radii [kpc].
    halo : NFWHalo
        Halo truncation parameters.

    Returns
    -------
    factor : ndarray, shape (N,)
        Values in ``[0, 1]`` multiplying the untruncated NFW density.
    """
    radii = np.asarray(r, dtype=float)
    t = math.sqrt(0.5) * (radii - halo.r_outer) / halo.dr_trunc
    factor = np.zeros_like(radii, dtype=float)
    factor[t < -4.0] = 1.0
    mid = (t >= -4.0) & (t <= 4.0)
    if np.any(mid):
        factor[mid] = 0.5 * erfc(t[mid])
    return factor


def halo_density_spherical(r: float, halo: NFWHalo) -> float:
    """
    Truncated NFW halo density.

    Parameters
    ----------
    r : float
        Spherical radius [kpc].
    halo : NFWHalo
        Enabled halo component.

    Returns
    -------
    float
        Mass density [GalactICS units kpc\\ :sup:`-3`].
    """
    if not halo.enabled or r <= 0.0:
        return 0.0
    return nfw_density(r, halo) * halo_trunc_factor(r, halo)


def halo_density_spherical_array(r: np.ndarray, halo: NFWHalo) -> np.ndarray:
    """
    Vectorized truncated NFW halo density.

    Parameters
    ----------
    r : array_like, shape (N,)
        Spherical radii [kpc].  Values below ``1e-6`` kpc are clamped for stability.
    halo : NFWHalo
        Enabled halo component.

    Returns
    -------
    rho : ndarray, shape (N,)
        Mass density [GalactICS units kpc\\ :sup:`-3`].  Zero when the halo is disabled.
    """
    if not halo.enabled:
        return np.zeros_like(np.asarray(r, dtype=float), dtype=float)
    radii = np.asarray(r, dtype=float)
    rho = np.zeros_like(radii, dtype=float)
    positive = radii > 0.0
    if not np.any(positive):
        return rho
    rp = np.maximum(radii[positive], 1e-6)
    rho[positive] = nfw_density_array(rp, halo) * halo_trunc_factor_array(rp, halo)
    return rho


def disk_surface_density(r: float, model: GalaxyModel) -> float:
    """
    Midplane stellar disk surface density ``Sigma(R)``.

    Parameters
    ----------
    r : float
        Cylindrical radius [kpc].
    model : GalaxyModel
        Model containing an :class:`~galacticsics.models.ExponentialDisk`.

    Returns
    -------
    float
        Surface density [GalactICS units kpc\\ :sup:`-2`].
    """
    disk = model.disk
    if disk is None or not disk.enabled:
        return 0.0
    sigma0 = disk.mass / (2.0 * math.pi * disk.scale_length**2)
    return _truncated_exponential(
        r,
        sigma0,
        disk.scale_length,
        disk.outer_radius,
        disk.trunc_width,
    )


def disk_surface_density_array(r: np.ndarray, model: GalaxyModel) -> np.ndarray:
    """
    Vectorized midplane stellar disk surface density ``Sigma(R)``.

    Parameters
    ----------
    r : array_like, shape (N,)
        Cylindrical radii [kpc].
    model : GalaxyModel
        Model containing an :class:`~galacticsics.models.ExponentialDisk`.

    Returns
    -------
    sigma : ndarray, shape (N,)
        Surface density [GalactICS units kpc\\ :sup:`-2`].
    """
    disk = model.disk
    radii = np.asarray(r, dtype=float)
    if disk is None or not disk.enabled:
        return np.zeros_like(radii, dtype=float)
    sigma0 = disk.mass / (2.0 * math.pi * disk.scale_length**2)
    sigma = np.zeros_like(radii, dtype=float)
    positive = radii > 0.0
    if not np.any(positive):
        return sigma
    rp = radii[positive]
    t = (rp - disk.outer_radius) / disk.trunc_width
    trunc = np.zeros_like(rp)
    trunc[t < -4.0] = 1.0
    mid = (t >= -4.0) & (t <= 4.0)
    if np.any(mid):
        trunc[mid] = 0.5 * erfc(t[mid])
    sigma[positive] = sigma0 * np.exp(-rp / disk.scale_length) * trunc
    return sigma


def disk_density_psi_batch(
    s: np.ndarray,
    z: np.ndarray,
    psi: np.ndarray,
    psi_mid: np.ndarray,
    psi_at_3zd: np.ndarray,
    model: GalaxyModel,
) -> np.ndarray:
    """
    Vectorized disk density from the potential-based vertical profile (``diskdens``).

    Parameters
    ----------
    s, z : array_like, shape (N,)
        Cylindrical coordinates [kpc].
    psi, psi_mid, psi_at_3zd : array_like, shape (N,)
        Potential at ``(s, z)``, ``(s, 0)``, and ``(s, 3 z_d)`` [100 km/s]\\ :sup:`2`].
    model : GalaxyModel
        Galaxy configuration.

    Returns
    -------
    rho : ndarray, shape (N,)
        Mass density [GalactICS units kpc\\ :sup:`-3`].
    """
    disk = model.disk
    s_arr = np.asarray(s, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    if disk is None or not disk.enabled:
        return np.zeros_like(s_arr, dtype=float)
    zdisk = disk.scale_height
    rho = np.zeros_like(s_arr, dtype=float)
    active = np.abs(z_arr / zdisk) <= 30.0
    if not np.any(active):
        return rho
    dpsizh = psi_mid[active] - psi_at_3zd[active]
    dpsi = psi_mid[active] - psi[active]
    coeff = np.divide(
        dpsi,
        dpsizh,
        out=np.zeros_like(dpsi),
        where=np.abs(dpsizh) > 0.0,
    )
    con = np.zeros_like(coeff)
    on_midplane = z_arr[active] == 0.0
    con[on_midplane] = 1.0
    off = ~on_midplane
    if np.any(off):
        coeff_off = coeff[off]
        con_off = np.zeros_like(coeff_off)
        valid = (coeff_off <= 16.0) & (coeff_off >= 0.0)
        con_off[valid] = 0.009866**coeff_off[valid]
        con[off] = con_off
    surface = disk_surface_density_array(np.hypot(s_arr[active], z_arr[active]), model)
    rho[active] = 0.5 / zdisk * surface * con
    return rho


def bulge_density_psi_array(
    psi: np.ndarray,
    energies: np.ndarray,
    dens_psi: np.ndarray,
    *,
    psi0: float,
    psic: float,
    psid: float,
) -> np.ndarray:
    """
    Interpolate bulge density from a ``denspsibulge.dat`` table on an energy grid.

    Parameters
    ----------
    psi : array_like, shape (N,)
        Local potential values [100 km/s]\\ :sup:`2`].
    energies : ndarray, shape (npsi,)
        Monotonic energy grid.
    dens_psi : ndarray, shape (npsi,)
        Tabulated bulge density at each energy.
    psi0, psic, psid : float
        Reference, cutoff, and inner energies matching legacy log spacing.

    Returns
    -------
    rho : ndarray, shape (N,)
        Bulge mass density at each ``psi``.
    """
    psi_arr = np.asarray(psi, dtype=float)
    rho = np.zeros_like(psi_arr, dtype=float)
    above_cutoff = psi_arr >= psic
    if not np.any(above_cutoff):
        return rho
    npsi = len(energies)
    log_num = np.log((psi0 - psi_arr[above_cutoff]) / max(psi0 - psid, 1e-30))
    log_den = math.log((psi0 - psic) / max(psi0 - psid, 1e-30))
    rj = 1.0 + float(npsi - 1) * log_num / log_den
    j = np.clip(rj.astype(int), 1, npsi - 1)
    frac = rj - j.astype(float)
    at_ref = psi_arr[above_cutoff] >= psi0
    interp = dens_psi[j - 1] + frac * (dens_psi[j] - dens_psi[j - 1])
    rho_vals = np.where(at_ref, float(dens_psi[0]), interp)
    rho[above_cutoff] = rho_vals
    return rho


def disk_vertical_sech2(z: float, zdisk: float) -> float:
    """Vertical sech\\ :sup:`2` factor ``rho_z(z)`` normalized to unit integral."""
    if zdisk <= 0.0:
        return 0.0
    zz = abs(z) / zdisk
    return 0.5 / zdisk * sech2_stable(zz)


def disk_density_estimate(s: float, z: float, model: GalaxyModel) -> float:
    """
    Analytic disk density estimate subtracted during harmonic synthesis.

    This is the ``appdiskdens`` contribution removed in ``polardens.f`` so the
    remaining multipoles represent departures from the approximate disk potential.
    """
    from galacticsics.potential.poisson.appdisk import approximate_disk_density

    return approximate_disk_density(s, z, model)


def disk_density_psi(
    s: float,
    z: float,
    psi: float,
    psi_mid: float,
    psi_at_3zd: float,
    model: GalaxyModel,
) -> float:
    """
    Disk density from the potential-based vertical profile (``diskdens``).

    Parameters
    ----------
    s, z : float
        Cylindrical coordinates [kpc].
    psi, psi_mid, psi_at_3zd : float
        Potential at ``(s, z)``, ``(s, 0)``, and ``(s, 3 z_d)`` [100 km/s]\\ :sup:`2`].
    model : GalaxyModel
        Galaxy configuration.

    Returns
    -------
    float
        Mass density [GalactICS units kpc\\ :sup:`-3`].
    """
    disk = model.disk
    if disk is None or not disk.enabled:
        return 0.0
    if abs(z / disk.scale_height) > 30.0:
        return 0.0
    zdisk = disk.scale_height
    if z == 0.0:
        con = 1.0
    else:
        dpsizh = psi_mid - psi_at_3zd
        dpsi = psi_mid - psi
        coeff = dpsi / dpsizh if abs(dpsizh) > 0.0 else 0.0
        if coeff > 16.0 or coeff < 0.0:
            con = 0.0
        else:
            con = 0.009866**coeff
    f = disk_surface_density(math.hypot(s, z), model)
    return 0.5 / zdisk * f * con


def bulge_density_psi(
    psi: float,
    energies: np.ndarray,
    dens_psi: np.ndarray,
    *,
    psi0: float,
    psic: float,
    psid: float,
) -> float:
    """
  Interpolate bulge density from a ``denspsibulge.dat`` table.

    Parameters
    ----------
    psi : float
        Local potential value [100 km/s]\\ :sup:`2`].
    energies : ndarray, shape (npsi,)
        Monotonic energy grid.
    dens_psi : ndarray, shape (npsi,)
        Tabulated bulge density at each energy.
    psi0, psic, psid : float
        Reference, cutoff, and inner energies matching legacy log spacing.

    Returns
    -------
    float
        Bulge mass density at ``psi``.
    """
    if psi < psic:
        return 0.0
    if psi >= psi0:
        return float(dens_psi[0])
    npsi = len(energies)
    rj = 1.0 + float(npsi - 1) * math.log((psi0 - psi) / max(psi0 - psid, 1e-30)) / math.log(
        (psi0 - psic) / max(psi0 - psid, 1e-30)
    )
    j = int(rj)
    j = max(1, min(j, npsi - 1))
    frac = rj - float(j)
    return float(dens_psi[j - 1] + frac * (dens_psi[j] - dens_psi[j - 1]))


def halo_density_from_psi_array(
    psi: np.ndarray,
    energies: np.ndarray,
    dens_psi: np.ndarray,
    *,
    psi0: float,
) -> np.ndarray:
    """
    Interpolate halo density from a ``denspsihalo.dat`` table on an energy grid.

    Parameters
    ----------
    psi : array_like, shape (N,)
        Local potential values [100 km/s]\\ :sup:`2`].
    energies : ndarray, shape (npsi,)
        Monotonic energy grid (need not be sorted).
    dens_psi : ndarray, shape (npsi,)
        Tabulated halo density at each energy.
    psi0 : float
        Reference potential at the origin; ``psi >= psi0`` returns ``dens_psi[0]``.

    Returns
    -------
    rho : ndarray, shape (N,)
        Halo mass density at each ``psi``.
    """
    psi_arr = np.asarray(psi, dtype=float)
    order = np.argsort(energies)
    e_sorted = energies[order]
    d_sorted = dens_psi[order]
    rho = np.interp(
        psi_arr,
        e_sorted,
        d_sorted,
        left=float(d_sorted[0]),
        right=float(d_sorted[-1]),
    )
    at_ref = psi_arr >= psi0
    if np.any(at_ref):
        rho = np.where(at_ref, float(dens_psi[0]), rho)
    return rho


def total_density_harmonic_batch(
    s: np.ndarray,
    z: np.ndarray,
    psi: np.ndarray,
    psi_mid: np.ndarray,
    psi_at_3zd: np.ndarray,
    model: GalaxyModel,
    *,
    dens_psi_halo: Callable[[float], float] | None,
    dens_psi_bulge: Callable[[float], float] | None,
    psic: float,
    halo_psi_tables: tuple[np.ndarray, np.ndarray, float] | None = None,
    bulge_psi_tables: tuple[np.ndarray, np.ndarray, float, float] | None = None,
) -> np.ndarray:
    """
    Vectorized :func:`total_density_harmonic` for polar quadrature nodes.

    Parameters
    ----------
    s, z : array_like, shape (N,)
        Cylindrical coordinates [kpc].
    psi, psi_mid, psi_at_3zd : array_like, shape (N,)
        Potential samples used by the disk vertical ansatz [100 km/s]\\ :sup:`2`].
    model : GalaxyModel
        Galaxy with any subset of halo, disk, and bulge enabled.
    dens_psi_halo, dens_psi_bulge : callable or None
        Scalar interpolators from DF tables (fallback when batch tables omitted).
    psic : float
        Halo/bulge DF cutoff potential [100 km/s]\\ :sup:`2`].
    halo_psi_tables : tuple of ndarray, optional
        ``(energies, dens_psi, psi0)`` for fast halo lookup during shell integration.
    bulge_psi_tables : tuple, optional
        ``(energies, dens_psi, psi0, psid)`` for fast bulge lookup.

    Returns
    -------
    rho : ndarray, shape (N,)
        Non-negative density after subtracting the approximate disk term.
    """
    from galacticsics.potential.poisson.appdisk import approximate_disk_density

    s_arr = np.asarray(s, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    psi_arr = np.asarray(psi, dtype=float)
    psi_mid_arr = np.asarray(psi_mid, dtype=float)
    psi_3zd_arr = np.asarray(psi_at_3zd, dtype=float)
    rho = np.zeros_like(s_arr, dtype=float)
    above_cutoff = psi_arr >= psic
    if model.halo and model.halo.enabled and dens_psi_halo is not None and np.any(above_cutoff):
        if halo_psi_tables is not None:
            energies, dens_tab, psi0 = halo_psi_tables
            rho[above_cutoff] += halo_density_from_psi_array(
                psi_arr[above_cutoff],
                energies,
                dens_tab,
                psi0=psi0,
            )
        else:
            for idx in np.flatnonzero(above_cutoff):
                rho[idx] += float(dens_psi_halo(float(psi_arr[idx])))
    if model.bulge and model.bulge.enabled and dens_psi_bulge is not None and np.any(above_cutoff):
        if bulge_psi_tables is not None:
            energies, dens_tab, psi0, psid = bulge_psi_tables
            rho[above_cutoff] += bulge_density_psi_array(
                psi_arr[above_cutoff],
                energies,
                dens_tab,
                psi0=psi0,
                psic=psic,
                psid=psid,
            )
        else:
            for idx in np.flatnonzero(above_cutoff):
                rho[idx] += float(dens_psi_bulge(float(psi_arr[idx])))
    if model.disk and model.disk.enabled:
        rho += disk_density_psi_batch(
            s_arr, z_arr, psi_arr, psi_mid_arr, psi_3zd_arr, model
        )
        from galacticsics.potential.poisson.appdisk import approximate_disk_density_batch

        rho -= approximate_disk_density_batch(s_arr, z_arr, model)
    return np.maximum(rho, 0.0)


def total_density_harmonic(
    s: float,
    z: float,
    psi: float,
    psi_mid: float,
    psi_at_3zd: float,
    model: GalaxyModel,
    *,
    dens_psi_halo: Callable[[float], float] | None,
    dens_psi_bulge: Callable[[float], float] | None,
    psic: float,
) -> float:
    """
    Total density for harmonic synthesis (``totdens - appdisk``).

    Parameters
    ----------
    s, z : float
        Cylindrical coordinates [kpc].
    psi, psi_mid, psi_at_3zd : float
        Potential samples used by the disk vertical ansatz.
    model : GalaxyModel
        Galaxy with any subset of halo, disk, and bulge enabled.
    dens_psi_halo, dens_psi_bulge : callable or None
        Interpolators built from ``denspsihalo.dat`` / ``denspsibulge.dat``.
    psic : float
        Halo/bulge DF cutoff potential.

    Returns
    -------
    float
        Non-negative density after subtracting the approximate disk term.
    """
    rho = 0.0
    if model.halo and model.halo.enabled and dens_psi_halo is not None and psi >= psic:
        rho += float(dens_psi_halo(psi))
    if model.bulge and model.bulge.enabled and dens_psi_bulge is not None and psi >= psic:
        rho += float(dens_psi_bulge(psi))
    if model.disk and model.disk.enabled:
        rho += disk_density_psi(s, z, psi, psi_mid, psi_at_3zd, model)
    rho -= disk_density_estimate(s, z, model)
    return max(rho, 0.0)


def bulge_density_on_grid(radii: np.ndarray, bulge: SersicBulge) -> np.ndarray:
    """
    Spherical bulge density tabulated on a radial grid.

    Parameters
    ----------
    radii : ndarray, shape (nr + 1,)
        Grid radii [kpc].
    bulge : SersicBulge
        Bulge component.

    Returns
    -------
    ndarray, shape (nr + 1,)
        Density values [GalactICS units kpc\\ :sup:`-3`].
    """
    params = sersic_params_from_bulge(bulge)
    r_safe = np.maximum(np.asarray(radii, dtype=float), 1e-6)
    if not bulge.enabled:
        return np.zeros_like(r_safe, dtype=float)
    return np.fromiter(
        (bulge_density_spherical(float(r), bulge, params) for r in r_safe),
        dtype=float,
        count=r_safe.size,
    )
