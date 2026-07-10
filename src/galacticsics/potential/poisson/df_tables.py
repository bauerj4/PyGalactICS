"""
Distribution-function and ``dens(psi)`` tables for Python sampling.

Writes the auxiliary files consumed by ``genhalo``, ``genbulge``, and the
Poisson solver:

* ``dfnfw.dat``, ``denspsihalo.dat``, ``dfhalo.table``
* ``dfsersic.dat``, ``denspsibulge.dat``
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np

from galacticsics.models import GalaxyModel
from galacticsics.potential.poisson.densities import (
    bulge_density_on_grid,
    bulge_density_psi,
    halo_density_spherical,
    halo_density_spherical_array,
)
from galacticsics.potential.poisson.appdisk import disk_monopole_density_grid
from galacticsics.potential.poisson.integrals import (
    integrated_disk_density_on_cylinder,
    monopole_estimate_from_spherical_density,
)
from galacticsics.potential.poisson.sersic import (
    SersicParams,
    sersic_d2rho_dpsi2,
    sersic_density,
    sersic_force,
    sersic_params_from_bulge,
)


def _get_total_psi_estimate(
    r: float,
    hpot: np.ndarray,
    dpot: np.ndarray,
    bpot: np.ndarray,
    dr: float,
) -> float:
    """
    Linearly interpolate the monopole potential estimate at radius ``r``.

    Parameters
    ----------
    r : float
        Radius [kpc].
    hpot, dpot, bpot : ndarray, shape (nr + 1,)
        Halo, disk, and bulge monopole contributions [100 km/s]\\ :sup:`2`].
    dr : float
        Radial step [kpc].

    Returns
    -------
    float
        Total potential estimate.
    """
    if r <= 0.0:
        return float(hpot[0] + dpot[0] + bpot[0])
    ihi = int(r / dr)
    ihi = min(max(ihi, 0), len(hpot) - 2)
    r1 = dr * ihi
    r2 = dr * (ihi + 1)
    t = (r - r1) / (r2 - r1) if r2 > r1 else 0.0
    return float(
        t * (hpot[ihi + 1] + dpot[ihi + 1] + bpot[ihi + 1])
        + (1.0 - t) * (hpot[ihi] + dpot[ihi] + bpot[ihi])
    )


def build_monopole_estimates(
    model: GalaxyModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build halo, disk, and bulge monopole potential/force estimates on the model grid.

    Matches legacy ``halopotentialestimate`` / ``diskpotentialestimate`` layout.
    Disk density uses ``diskdensestimate`` quadrature (not ``appdiskdens``).

    Parameters
    ----------
    model : GalaxyModel
        Galaxy with grid and enabled components.

    Returns
    -------
    hpot : ndarray, shape (nr + 1,)
        Halo monopole potential [100 km/s]\\ :sup:`2`].
    hfr : ndarray, shape (nr + 1,)
        Halo monopole inward force.
    ddens : ndarray, shape (nr + 1,)
        Disk monopole density estimate from ``diskpotentialestimate``.
    dpot : ndarray, shape (nr + 1,)
        Disk monopole potential.
    dfr : ndarray, shape (nr + 1,)
        Disk monopole inward force.
    bpot : ndarray, shape (nr + 1,)
        Bulge monopole potential.
    bfr : ndarray, shape (nr + 1,)
        Bulge monopole inward force.
    """
    dr = model.grid.dr
    nr = model.grid.nr
    radii = np.arange(nr + 1, dtype=float) * dr

    halo = model.halo
    if halo is not None and halo.enabled:
        rho_h = halo_density_spherical_array(radii, halo)
        hpot, hfr = monopole_estimate_from_spherical_density(rho_h, dr=dr, nr=nr)
    else:
        hpot = np.zeros(nr + 1, dtype=float)
        hfr = np.zeros(nr + 1, dtype=float)

    if model.disk is not None and model.disk.enabled:
        ddens = disk_monopole_density_grid(model, ntheta=100)
        dpot, dfr = monopole_estimate_from_spherical_density(ddens, dr=dr, nr=nr)
    else:
        ddens = np.zeros(nr + 1, dtype=float)
        dpot = np.zeros(nr + 1, dtype=float)
        dfr = np.zeros(nr + 1, dtype=float)

    if model.bulge is not None and model.bulge.enabled:
        rho_b = bulge_density_on_grid(radii, model.bulge)
        bpot, bfr = monopole_estimate_from_spherical_density(rho_b, dr=dr, nr=nr)
    else:
        bpot = np.zeros(nr + 1, dtype=float)
        bfr = np.zeros(nr + 1, dtype=float)

    return hpot, hfr, ddens, dpot, dfr, bpot, bfr


def build_energy_table(
    model: GalaxyModel,
    hpot: np.ndarray,
    dpot: np.ndarray,
    bpot: np.ndarray,
    *,
    npsi: int,
    rmin: float | None = None,
) -> tuple[float, float, float, np.ndarray]:
    """
    Log-spaced binding-energy grid (``gentableE``).

    Returns
    -------
    psi0 : float
        Central potential [100 km/s]\\ :sup:`2`].
    psic : float
        Outer cutoff potential.
    psid : float
        Inner reference potential at ``rmin``.
    energies : ndarray, shape (npsi,)
        Monotonic energy samples.
    """
    dr = model.grid.dr
    if rmin is None:
        rmin = max(0.001, dr)
    psi0 = _get_total_psi_estimate(0.0, hpot, dpot, bpot, dr)
    halo = model.halo
    assert halo is not None
    psic = _get_total_psi_estimate(halo.r_outer + 5.0 * halo.dr_trunc, hpot, dpot, bpot, dr)
    psid = _get_total_psi_estimate(rmin, hpot, dpot, bpot, dr)
    energies = np.zeros(npsi, dtype=float)
    for i in range(npsi):
        frac = i / max(npsi - 1, 1)
        log_term = frac * math.log((psi0 - psic) / max(psi0 - psid, 1e-30))
        energies[i] = psi0 - math.exp(log_term) * (psi0 - psid)
    return psi0, psic, psid, energies


def _invert_psi_to_radius(
    psi: float,
    hpot: np.ndarray,
    dpot: np.ndarray,
    bpot: np.ndarray,
    dr: float,
    rmin: float,
    rmax: float,
) -> float:
    """
    Invert monopole ``Psi(r)`` to radius by bisection.

    Parameters
    ----------
    psi : float
        Target potential value [100 km/s]\\ :sup:`2`].
    hpot, dpot, bpot : ndarray
        Component monopole potentials.
    dr : float
        Radial grid step [kpc].
    rmin, rmax : float
        Bracketing radii for bisection [kpc].

    Returns
    -------
    float
        Radius where ``Psi(r) ≈ psi``.
    """
    lo, hi = rmin, rmax
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _get_total_psi_estimate(mid, hpot, dpot, bpot, dr) > psi:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _interp_monopole_row(r: float, values: np.ndarray, dr: float, nr: int) -> float:
    """
    Linear interpolation on legacy radial monopole tables.

    Parameters
    ----------
    r : float
        Radius [kpc].
    values : ndarray, shape (nr + 1,)
        Tabulated monopole quantity.
    dr : float
        Radial step [kpc].
    nr : int
        Number of radial bins.

    Returns
    -------
    float
        Interpolated value at ``r``.
    """
    if r <= 0.0:
        return float(values[0])
    ihi = min(max(int(r / dr) + 1, 1), nr)
    r1 = dr * (ihi - 1)
    r2 = dr * ihi
    t = (r - r1) / (r2 - r1) if r2 > r1 else 0.0
    return float(values[ihi] * t + values[ihi - 1] * (1.0 - t))


def _d2_rho_dpsi2_nfw_legacy(
    r: float,
    model: GalaxyModel,
    *,
    hpot: np.ndarray,
    hfr: np.ndarray,
    ddens: np.ndarray,
    dpot: np.ndarray,
    dfr: np.ndarray,
    bpot: np.ndarray,
    bfr: np.ndarray,
    dr: float,
    nr: int,
) -> float:
    """
    Legacy ``getd2rhonfwdpsi2`` (``gendf.f``): include disk+bulge forces in denominator.

    Parameters
    ----------
    r : float
        Spherical radius [kpc].
    model : GalaxyModel
        Galaxy with enabled halo (and optional disk/bulge).
    hpot, hfr, ddens, dpot, dfr, bpot, bfr : ndarray
        Monopole estimates from :func:`build_monopole_estimates`.
    dr : float
        Radial step [kpc].
    nr : int
        Number of radial bins.

    Returns
    -------
    float
        ``d²ρ/dψ²`` for the NFW halo at radius ``r``.
    """
    halo = model.halo
    if halo is None or not halo.enabled or r <= 0.0:
        return 0.0
    s = r / halo.a
    den = halo_density_spherical(r, halo)
    denp = -den / halo.a * (3.0 * s + halo.cusp) / s / (1.0 + s)
    denpp = (
        den
        / halo.a**2
        * (halo.cusp * (halo.cusp + 1.0) + 8.0 * halo.cusp * s + 12.0 * s * s)
        / s**2
        / (1.0 + s) ** 2
    )
    force = _interp_monopole_row(r, hfr, dr, nr)
    totalden = den
    if model.disk is not None and model.disk.enabled:
        force += _interp_monopole_row(r, dfr, dr, nr)
        totalden += _interp_monopole_row(r, ddens, dr, nr)
    if model.bulge is not None and model.bulge.enabled:
        force += _interp_monopole_row(r, bfr, dr, nr)
        from galacticsics.potential.poisson.densities import bulge_density_on_grid

        totalden += float(bulge_density_on_grid(np.array([r]), model.bulge)[0])
    force = -abs(force) if force > 0 else force
    if abs(force) < 1e-30:
        return 0.0
    bbb = 4.0 * math.pi * totalden * denp / force
    ccc = 2.0 * denp / r
    ddd = denpp
    return (bbb + ccc + ddd) / (force * force)


def _d2_rho_dpsi2_nfw(r: float, halo) -> float:
    """
    Spherical NFW ``d²ρ/dψ²`` without disk/bulge force corrections.

    Parameters
    ----------
    r : float
        Spherical radius [kpc].
    halo : NFWHalo
        Halo parameters.

    Returns
    -------
    float
        Second derivative of density with respect to potential.
    """
    s = r / halo.a
    if s <= 0.0:
        return 0.0
    cusp = halo.cusp
    haloconst = (2.0 ** (1.0 - cusp)) * halo.v0**2 / (4.0 * math.pi * halo.a**2)
    num = cusp * (cusp + 1.0) + 8.0 * cusp * s + 12.0 * s * s
    den = s * s * (1.0 + s) ** 2
    return haloconst / halo.a**2 * num / den


def _spherical_total_at_radius(
    r: float,
    model: GalaxyModel,
    *,
    sersic: SersicParams | None,
) -> tuple[float, float]:
    """
    Combined spherical density and inward force for DF inversion.

    Parameters
    ----------
    r : float
        Spherical radius [kpc].
    model : GalaxyModel
        Galaxy configuration.
    sersic : SersicParams or None
        Bulge parameters when bulge is enabled.

    Returns
    -------
    total_den : float
        Combined mass density [kpc\\ :sup:`-3`].
    total_force : float
        Inward gravitational force magnitude.
    """
    total_den = 0.0
    total_force = 0.0
    halo = model.halo
    if halo is not None and halo.enabled and r > 0.0:
        total_den += halo_density_spherical(r, halo)
        s = r / halo.a
        haloconst = (2.0 ** (1.0 - halo.cusp)) * halo.v0**2 / (4.0 * math.pi * halo.a**2)
        l2 = haloconst * halo.a * (s ** (1.0 - halo.cusp)) / ((1.0 + s) ** (2.0 - halo.cusp))
        total_force += abs(-4.0 * math.pi * l2 / (r * r))
    if model.bulge is not None and model.bulge.enabled and sersic is not None:
        total_den += sersic_density(r, sersic)
        total_force += abs(sersic_force(r, sersic))
    if model.disk is not None and model.disk.enabled and r > 0.0:
        dd = integrated_disk_density_on_cylinder(r, model, ntheta=16)
        total_den += dd
        # Spherical approximation: M_enc ~ 4 pi int_0^r rho(r') r'^2 dr'
        total_force += abs(4.0 * math.pi * dd / max(r, 1e-6))
    return total_den, max(total_force, 1e-30)


def compute_nfw_df_table(
    energies: np.ndarray,
    model: GalaxyModel,
    hpot: np.ndarray,
    dpot: np.ndarray,
    bpot: np.ndarray,
    *,
    hfr: np.ndarray | None = None,
    ddens: np.ndarray | None = None,
    dfr: np.ndarray | None = None,
    bfr: np.ndarray | None = None,
    nint: int = 20,
    rmin: float = 0.001,
) -> np.ndarray:
    """
    Eddington-inverted halo DF values (log stored like ``dfnfw.dat``).

    Parameters
    ----------
    energies : ndarray, shape (npsi,)
        Binding-energy grid.
    model : GalaxyModel
        Galaxy model with enabled halo.
    hpot, dpot, bpot : ndarray
        Monopole potential estimates.
    nint : int, optional
        Trapezoidal points in the velocity integral.
    rmin : float, optional
        Inner radius bracket for ``Psi(r)`` inversion.

    Returns
    -------
    log_df : ndarray, shape (npsi,)
        Natural logarithm of the DF, holding the last positive value when zero.
    """
    halo = model.halo
    assert halo is not None
    dr = model.grid.dr
    nr = model.grid.nr
    if hfr is None or ddens is None or dfr is None or bfr is None:
        hpot, hfr, ddens, dpot, dfr, bpot, bfr = build_monopole_estimates(model)
    psic = _get_total_psi_estimate(halo.r_outer + 5.0 * halo.dr_trunc, hpot, dpot, bpot, dr)
    log_df = np.zeros(len(energies), dtype=float)
    last = -30.0
    for i, energy in enumerate(energies):
        if energy < psic:
            log_df[i] = last
            continue
        tmax = math.sqrt(max(energy - psic, 0.0))
        if tmax <= 0.0:
            log_df[i] = last
            continue
        dt = tmax / max(nint - 1, 1)
        rpsi = _invert_psi_to_radius(
            energy, hpot, dpot, bpot, dr, rmin, halo.r_outer + halo.dr_trunc
        )
        d2 = _d2_rho_dpsi2_nfw_legacy(
            rpsi,
            model,
            hpot=hpot,
            hfr=hfr,
            ddens=ddens,
            dpot=dpot,
            dfr=dfr,
            bpot=bpot,
            bfr=bfr,
            dr=dr,
            nr=nr,
        )
        total = dt * d2
        for j in range(1, nint - 1):
            t = dt * j
            psi_j = energy - t * t
            r_j = _invert_psi_to_radius(
                psi_j, hpot, dpot, bpot, dr, rmin, halo.r_outer + halo.dr_trunc
            )
            total += 2.0 * dt * _d2_rho_dpsi2_nfw_legacy(
                r_j,
                model,
                hpot=hpot,
                hfr=hfr,
                ddens=ddens,
                dpot=dpot,
                dfr=dfr,
                bpot=bpot,
                bfr=bfr,
                dr=dr,
                nr=nr,
            )
        df = total / (math.sqrt(8.0) * math.pi**2)
        if df > 0.0:
            last = math.log(df)
        log_df[i] = last
    return log_df


def compute_sersic_df_table(
    energies: np.ndarray,
    model: GalaxyModel,
    hpot: np.ndarray,
    dpot: np.ndarray,
    bpot: np.ndarray,
    *,
    nint: int = 20,
    rmin: float = 0.001,
) -> np.ndarray:
    """
    Eddington-inverted bulge DF values (log stored like ``dfsersic.dat``).

    Returns
    -------
    log_df : ndarray, shape (npsi,)
        Natural logarithm of the bulge DF.
    """
    halo = model.halo
    bulge = model.bulge
    assert halo is not None and bulge is not None
    dr = model.grid.dr
    sersic = sersic_params_from_bulge(bulge)
    psic = _get_total_psi_estimate(halo.r_outer + 5.0 * halo.dr_trunc, hpot, dpot, bpot, dr)
    log_df = np.zeros(len(energies), dtype=float)
    last = -30.0
    for i, energy in enumerate(energies):
        if energy < psic:
            log_df[i] = last
            continue
        tmax = math.sqrt(max(energy - psic, 0.0))
        if tmax <= 0.0:
            log_df[i] = last
            continue
        dt = tmax / max(nint - 1, 1)
        rpsi = _invert_psi_to_radius(
            energy, hpot, dpot, bpot, dr, rmin, halo.r_outer + halo.dr_trunc
        )
        total_den, total_force = _spherical_total_at_radius(rpsi, model, sersic=sersic)
        d2 = sersic_d2rho_dpsi2(
            rpsi, sersic, total_density=total_den, total_force=total_force
        )
        total = dt * d2
        for j in range(1, nint - 1):
            t = dt * j
            psi_j = energy - t * t
            r_j = _invert_psi_to_radius(
                psi_j, hpot, dpot, bpot, dr, rmin, halo.r_outer + halo.dr_trunc
            )
            td, tf = _spherical_total_at_radius(r_j, model, sersic=sersic)
            total += 2.0 * dt * sersic_d2rho_dpsi2(r_j, sersic, total_density=td, total_force=tf)
        df = total / (math.sqrt(8.0) * math.pi**2)
        if df > 0.0:
            last = math.log(df)
        log_df[i] = last
    return log_df


def df_interp(energy: float, energies: np.ndarray, log_df: np.ndarray) -> float:
    """
    Log-linear interpolation of a DF table on a monotonic energy grid.

    Parameters
    ----------
    energy : float
        Binding energy [100 km/s]\\ :sup:`2`].
    energies : ndarray, shape (npsi,)
        Energy abscissas (need not be sorted).
    log_df : ndarray, shape (npsi,)
        Natural logarithm of DF values.

    Returns
    -------
    float
        Interpolated DF value (always positive via ``exp``).
    """
    order = np.argsort(energies)
    e = energies[order]
    ld = log_df[order]
    if energy <= e[0]:
        return math.exp(ld[0])
    if energy >= e[-1]:
        return math.exp(ld[-1])
    idx = int(np.searchsorted(e, energy) - 1)
    idx = max(0, min(idx, len(e) - 2))
    frac = (energy - e[idx]) / (e[idx + 1] - e[idx])
    logv = ld[idx] + frac * (ld[idx + 1] - ld[idx])
    return math.exp(logv)


def build_dens_psi_from_df(
    energies: np.ndarray,
    log_df: np.ndarray,
    *,
    psic: float,
    fcut: float | None = None,
) -> np.ndarray:
    """
    Integrate a lowered DF over velocity to obtain ``rho(psi)``.

    Parameters
    ----------
    energies : ndarray, shape (npsi,)
        Potential-energy grid.
    log_df : ndarray, shape (npsi,)
        Log DF samples.
    psic : float
        Lower potential cutoff.
    fcut : float or None, optional
        DF floor subtracted before integration (legacy uses ``0`` for bulge).

    Returns
    -------
    ndarray, shape (npsi,)
        Mass density at each energy.
    """
    if fcut is None:
        fcut = df_interp(psic, energies, log_df)
    rho = np.zeros(len(energies), dtype=float)
    coef = (17.0 / 48.0, 59.0 / 48.0, 43.0 / 48.0, 49.0 / 48.0)
    for i, psi in enumerate(energies):
        if i == len(energies) - 1:
            rho[i] = 0.0
            continue
        vmin = -10.0
        vmax = math.log(math.sqrt(max(2.0 * (psi - psic), 1e-30)))
        m = 40
        dlogv = (vmax - vmin) / max(m - 1, 1)
        acc = 0.0
        for j in range(1, 5):
            v = math.exp(vmin + dlogv * (j - 1))
            e = psi - 0.5 * v * v
            df = max(df_interp(e, energies, log_df) - fcut, 0.0)
            acc += 4.0 * math.pi * dlogv * v**3 * coef[j - 1] * df
        for j in range(5, m - 3):
            v = math.exp(vmin + dlogv * (j - 1))
            e = psi - 0.5 * v * v
            df = max(df_interp(e, energies, log_df) - fcut, 0.0)
            acc += 4.0 * math.pi * dlogv * v**3 * df
        for jj in range(4, 1, -1):
            j = m - jj + 1
            v = math.exp(vmin + dlogv * (j - 1))
            e = psi - 0.5 * v * v
            df = max(df_interp(e, energies, log_df) - fcut, 0.0)
            acc += 4.0 * math.pi * dlogv * v**3 * coef[jj - 1] * df
        rho[i] = acc
    return rho


def _dens_psi_factory(
    energies: np.ndarray,
    dens_psi: np.ndarray,
    *,
    psi0: float,
    psic: float,
    psid: float,
) -> Callable[[float], float]:
    """
    Build a log-spaced ``rho(psi)`` interpolator from tabulated arrays.

    Parameters
    ----------
    energies : ndarray, shape (npsi,)
        Energy grid.
    dens_psi : ndarray, shape (npsi,)
        Density samples.
    psi0, psic, psid : float
        Reference potentials for log spacing.

    Returns
    -------
    callable
        Function ``psi -> rho(psi)`` delegating to :func:`bulge_density_psi`.
    """

    def interp(psi: float) -> float:
        return bulge_density_psi(psi, energies, dens_psi, psi0=psi0, psic=psic, psid=psid)

    return interp


def write_df_artifacts(
    work_dir: Path,
    model: GalaxyModel,
    *,
    npsi: int = 1000,
    nint: int = 20,
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Write halo and bulge DF auxiliary files for sampling.

    Parameters
    ----------
    work_dir : Path
        Run directory.
    model : GalaxyModel
        Galaxy with enabled halo and optional disk / bulge.
    npsi, nint : int, optional
        Energy-grid resolution and Eddington quadrature count.

    Returns
    -------
    psi0, psic, psid : float
        Reference potentials.
    energies : ndarray, shape (npsi,)
        Energy grid shared by all tables.
    dens_psi_halo : ndarray or None
        Halo ``rho(psi)`` table; ``None`` when halo disabled.
    dens_psi_bulge : ndarray or None
        Bulge ``rho(psi)`` table; ``None`` when bulge disabled.
    """
    dr = model.grid.dr
    nr = model.grid.nr
    eff_npsi = min(npsi, max(128, nr * 2))
    eff_nint = min(nint, max(8, nr // 20 + 4))

    hpot, hfr, ddens, dpot, dfr, bpot, bfr = build_monopole_estimates(model)
    psi0, psic, psid, energies = build_energy_table(
        model, hpot, dpot, bpot, npsi=eff_npsi
    )

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    dens_psi_halo = None
    if model.halo is not None and model.halo.enabled:
        log_df_h = compute_nfw_df_table(
            energies,
            model,
            hpot,
            dpot,
            bpot,
            hfr=hfr,
            ddens=ddens,
            dfr=dfr,
            bfr=bfr,
            nint=eff_nint,
        )
        dens_psi_halo = build_dens_psi_from_df(energies, log_df_h, psic=psic)
        with (work_dir / "dfnfw.dat").open("w", encoding="utf-8") as f:
            for e, ld in zip(energies, log_df_h):
                f.write(f"{e:.8E} {ld:.8E}\n")
        with (work_dir / "denspsihalo.dat").open("w", encoding="utf-8") as f:
            for e, rho in zip(energies, dens_psi_halo):
                f.write(f"{e:.8E} {rho:.8E}\n")
        with (work_dir / "dfhalo.table").open("w", encoding="utf-8") as f:
            for e, ld in zip(energies, log_df_h):
                f.write(f"{e:.8E} {math.exp(ld):.8E}\n")

    dens_psi_bulge = None
    if model.bulge is not None and model.bulge.enabled:
        log_df_b = compute_sersic_df_table(
            energies, model, hpot, dpot, bpot, nint=eff_nint
        )
        dens_psi_bulge = build_dens_psi_from_df(energies, log_df_b, psic=psic, fcut=0.0)
        with (work_dir / "dfsersic.dat").open("w", encoding="utf-8") as f:
            for e, ld in zip(energies, log_df_b):
                f.write(f"{e:.8E} {ld:.8E}\n")
        with (work_dir / "denspsibulge.dat").open("w", encoding="utf-8") as f:
            for e, rho in zip(energies, dens_psi_bulge):
                f.write(f"{e:.8E} {rho:.8E}\n")

    return psi0, psic, psid, energies, dens_psi_halo, dens_psi_bulge


def write_halo_df_artifacts(
    work_dir: Path,
    model: GalaxyModel,
    *,
    npsi: int = 1000,
    nint: int = 20,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Backward-compatible wrapper writing halo tables only.

    Returns
    -------
    psi0, psic, psid : float
        Reference potentials.
    energies : ndarray, shape (npsi,)
        Energy grid.
    dens_psi_halo : ndarray, shape (npsi,)
        Halo ``rho(psi)`` samples.
    """
    psi0, psic, psid, energies, dens_h, _ = write_df_artifacts(
        work_dir, model, npsi=npsi, nint=nint
    )
    if dens_h is None:
        raise ValueError("write_halo_df_artifacts requires an enabled halo")
    return psi0, psic, psid, energies, dens_h
