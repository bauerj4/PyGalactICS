"""Python port of legacy ``diskdf``."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import integrate

from galacticsics.distribution.diskdf import DiskCorrectionTable
from galacticsics.io import read_harmonic_potential
from galacticsics.io.formats import read_frequency_table
from galacticsics.models import GalaxyModel
from galacticsics.numerics import natural_cubic_spline
from galacticsics.potential.evaluate import evaluate_potential


class _RcircInterpolator:
    """
    Spline inversion of specific angular momentum to circular radius (``rcirc.f``).

    Parameters
    ----------
    pot
        Harmonic potential with attached frequency table context.
    freq
        Tabulated epicycle frequencies on the DBH radial grid.
    """

    def __init__(self, pot, freq) -> None:
        dr = pot.model.grid.dr
        nr = pot.model.grid.nr
        amtab = np.zeros(nr, dtype=float)
        rtab = np.zeros(nr, dtype=float)
        for i in range(nr):
            r = i * dr
            omega = freq.omega(max(r, 1e-10))
            amtab[i] = omega * r * r
            rtab[i] = 1.0 / math.sqrt(max(omega, 1e-30))
        self._amtab = amtab
        self._rtab = rtab
        slope_inf = 1.5 * rtab[-1] / max(amtab[-1], 1e-30)
        self._inv = natural_cubic_spline(amtab, rtab)
        self._slope_inf = slope_inf

    def __call__(self, am: float) -> float:
        """
        Return the circular radius for specific angular momentum ``am = R v_phi``.

        Parameters
        ----------
        am : float
            Specific angular momentum [kpc * 100 km/s].

        Returns
        -------
        float
            Circular radius [kpc].
        """
        aam = abs(am)
        if aam <= 0.0:
            return 0.0
        if aam > self._amtab[-1]:
            return float(self._rtab[-1] * (aam / self._amtab[-1]) ** 2)
        rc_over_sqrt_am = float(self._inv(aam))
        return max(rc_over_sqrt_am * math.sqrt(aam), 0.0)


def _fnamidden(r: float, pot, f_d: float) -> float:
    """
    Midplane disk density with radial DF correction (``fnamidden.f``).

    Parameters
    ----------
    r : float
        Cylindrical radius [kpc].
    pot
        Harmonic potential.
    f_d : float
        Radial correction factor from the ``cordbh`` spline.

    Returns
    -------
    float
        Corrected midplane mass density [kpc\\ :sup:`-3`].
    """
    return _disk_midplane_density(r, 0.0, pot) * max(f_d, 1e-6)


def _sigma_r2_base(r: float, model: GalaxyModel) -> float:
    """
    Radial velocity dispersion squared without ``f_d`` correction (``sigr2.f``).

    Parameters
    ----------
    r : float
        Cylindrical radius [kpc].
    model : GalaxyModel
        Galaxy with disk kinematics.

    Returns
    -------
    float
        ``sigma_r^2`` [100 km/s]\\ :sup:`2`].
    """
    kin = model.disk_kinematics
    return kin.sigma_r0**2 * math.exp(-r / kin.sigma_r_scale)


def _sigma_r2(r: float, model: GalaxyModel, f_d: float) -> float:
    kin = model.disk_kinematics
    return _sigma_r2_base(r, model) * max(f_d, 1e-6)


def _sigma_z2_array(r: np.ndarray, pot, zdisk: float, f_sz: np.ndarray) -> np.ndarray:
    psizh = np.array([evaluate_potential(pot, float(ri), 3 * zdisk) for ri in r])
    psi00 = np.array([evaluate_potential(pot, float(ri), 0.0) for ri in r])
    base = (psizh - psi00) / math.log(0.419974)
    return np.maximum(base * f_sz, 1e-10)


def _sigma_z2(r: float, pot, zdisk: float, f_sz: float) -> float:
    return float(_sigma_z2_array(np.array([r]), pot, zdisk, np.array([f_sz]))[0])


def _disk_midplane_density(
    r: float,
    z: float,
    pot,
    *,
    psi_mid: float | None = None,
    psi_3zd: float | None = None,
) -> float:
    from galacticsics.potential.poisson.densities import disk_density_psi

    psi = evaluate_potential(pot, r, z)
    if psi_mid is None:
        psi_mid = evaluate_potential(pot, r, 0.0)
    if psi_3zd is None:
        psi_3zd = evaluate_potential(pot, r, 3 * pot.model.disk.scale_height)
    return disk_density_psi(r, z, psi, psi_mid, psi_3zd, pot.model)


def _diskdf3ez(
    ep: float,
    am: float,
    ez: float,
    pot,
    freq,
    rcirc_fn: _RcircInterpolator,
    spline_d,
    spline_sz,
) -> float:
    """
    3D epicycle DF (``diskdf3ez.f``) with legacy ``rcirc(am)`` evaluation.

    Parameters
    ----------
    ep, am, ez : float
        Epicycle actions / energies at the sampling point.
    pot, freq
        Potential and frequency tables.
    rcirc_fn : _RcircInterpolator
        Maps angular momentum to circular radius.
    spline_d, spline_sz
        Cubic splines for the running ``f_d`` / ``f_sz`` corrections.

    Returns
    -------
    float
        Distribution function value (non-negative).
    """
    disk = pot.model.disk
    assert disk is not None
    rc = rcirc_fn(am)
    if rc <= 0.0:
        return 0.0
    omega = freq.omega(rc)
    kappa = freq.kappa(rc)
    if kappa <= 0 or omega <= 0:
        return 0.0
    vc = rc * omega
    ec = -evaluate_potential(pot, rc, 0.0) + 0.5 * vc * vc
    # Legacy diskdf3ez.f: flip binding energy for counter-rotating orbits so the
    # epicycle DF strongly suppresses am < 0 (otherwise f is even in Lz).
    if am < 0.0:
        psi00 = evaluate_potential(pot, 0.0, 0.0)
        ec = -2.0 * psi00 - ec
    f_d = float(spline_d(rc))
    f_sz = float(spline_sz(rc))
    sr2 = _sigma_r2_base(rc, pot.model) * max(f_d, 1e-6)
    sz2 = _sigma_z2(rc, pot, disk.scale_height, f_sz)
    if sz2 <= 0 or sr2 <= 0:
        return 0.0
    fvert = _fnamidden(rc, pot, f_d) * math.exp(min(-ez / sz2, 700.0)) / math.sqrt(2 * math.pi * sz2)
    arg = (ep - ec) / sr2
    return max(
        omega / (math.pi * kappa) / sr2 * math.exp(min(-arg, 700.0)) * fvert,
        0.0,
    )


def _diskdf5ez(
    vr: float,
    vt: float,
    vz: float,
    r: float,
    z: float,
    pot,
    freq,
    rcirc_fn: _RcircInterpolator,
    spline_d,
    spline_sz,
) -> float:
    """Full 3D epicycle DF at velocity ``(vr, vt, vz)`` (``diskdf5ez.f``)."""
    psir0 = evaluate_potential(pot, r, 0.0)
    psirz = psir0 if z == 0.0 else evaluate_potential(pot, r, z)
    ep = 0.5 * (vr * vr + vt * vt) - psir0
    am = r * vt
    ez = 0.5 * vz * vz - psirz + psir0
    return _diskdf3ez(ep, am, ez, pot, freq, rcirc_fn, spline_d, spline_sz)


def _find_fmax(
    vpmax: float,
    r: float,
    z: float,
    pot,
    freq,
    rcirc_fn: _RcircInterpolator,
    spline_d,
    spline_sz,
    vsigp: float,
) -> float:
    """Approximate local DF maximum in ``v_phi`` (``FindMax1`` in ``gendisk.c``)."""
    dv = 0.1 * vsigp
    v0 = vpmax - dv
    v1 = vpmax + dv
    f0 = _diskdf5ez(0.0, v0, 0.0, r, z, pot, freq, rcirc_fn, spline_d, spline_sz)
    fmid = _diskdf5ez(0.0, vpmax, 0.0, r, z, pot, freq, rcirc_fn, spline_d, spline_sz)
    f1 = _diskdf5ez(0.0, v1, 0.0, r, z, pot, freq, rcirc_fn, spline_d, spline_sz)
    if fmid >= f0 and fmid >= f1 and fmid > 0.0:
        return fmid
    vmax = v0 + dv * np.argmax(
        [
            _diskdf5ez(0.0, v0 + i * 2 * dv, 0.0, r, z, pot, freq, rcirc_fn, spline_d, spline_sz)
            for i in range(101)
        ]
    )
    return _diskdf5ez(0.0, vmax, 0.0, r, z, pot, freq, rcirc_fn, spline_d, spline_sz)


def _diskdf3intez(
    ep: float,
    am: float,
    ez: float,
    pot,
    freq,
    rcirc_fn: _RcircInterpolator,
    spline_d,
    spline_sz,
) -> float:
    """
    Epicycle DF for ``diskdf`` quadrature (``diskdf3intez.f``).

    Uses ``rcirc(am)`` for circular-orbit quantities and ``fnamidden(rc)`` for
    the vertical factor.
    """
    disk = pot.model.disk
    assert disk is not None
    rc = rcirc_fn(am)
    if rc <= 0.0:
        return 0.0
    omega = freq.omega(rc)
    kappa = freq.kappa(rc)
    if kappa <= 0 or omega <= 0:
        return 0.0
    vc = rc * omega
    ec = -evaluate_potential(pot, rc, 0.0) + 0.5 * vc * vc
    # Legacy diskdf3intez.f: same counter-rotating energy flip as diskdf3ez.
    if am < 0.0:
        psi00 = evaluate_potential(pot, 0.0, 0.0)
        ec = -2.0 * psi00 - ec
    f_d = float(spline_d(rc))
    f_sz = float(spline_sz(rc))
    sr2 = _sigma_r2_base(rc, pot.model) * max(f_d, 1e-6)
    sz2 = _sigma_z2(rc, pot, disk.scale_height, f_sz)
    if sz2 <= 0 or sr2 <= 0:
        return 0.0
    fvert = _fnamidden(rc, pot, f_d) * math.exp(min(-ez / sz2, 700.0))
    arg = (ep - ec) / sr2
    val = omega / kappa * math.sqrt(2 / math.pi / sr2) * math.exp(min(-arg, 700.0)) * fvert
    return max(val, 0.0)


def _diskdf5intez(
    vt: float,
    r: float,
    z: float,
    pot,
    freq,
    rcirc_fn: _RcircInterpolator,
    spline_d,
    spline_sz,
) -> float:
    """Match ``diskdf5intez.f`` (epicycle DF at azimuthal speed ``vt``)."""
    psir0 = evaluate_potential(pot, r, 0.0)
    psirz = psir0 if z == 0.0 else evaluate_potential(pot, r, z)
    ep = 0.5 * vt * vt - psir0
    am = r * vt
    ez = -psirz + psir0
    return _diskdf3intez(ep, am, ez, pot, freq, rcirc_fn, spline_d, spline_sz)


def _diskdf5intez_batch(
    vt: np.ndarray,
    r: float,
    z: float,
    pot,
    freq,
    rcirc_fn: _RcircInterpolator,
    spline_d,
    spline_sz,
) -> np.ndarray:
    return np.array(
        [_diskdf5intez(float(v), r, z, pot, freq, rcirc_fn, spline_d, spline_sz) for v in vt]
    )


def _integrate_vphi(
    vt: np.ndarray,
    pot,
    freq,
    rcirc_fn: _RcircInterpolator,
    spline_d,
    spline_sz,
    r: float,
    z: float,
) -> float:
    """Simpson integration of ``diskdf5intez`` over azimuthal velocity."""
    df = np.clip(_diskdf5intez_batch(vt, r, z, pot, freq, rcirc_fn, spline_d, spline_sz), 0.0, 1e100)
    return float(integrate.simpson(df, x=vt))


def _diskdf_relaxation(model: GalaxyModel) -> float:
    """
    Under-relaxation factor for ``diskdf`` on coarse grids.

    Toomre-Q scaling of ``sigma_r0`` makes the correction loop oscillate when
    ``nr`` is capped for laptop screening; damped updates keep ``f_d`` in range.
    """
    kin = model.disk_kinematics
    if model.grid.nr <= 4000 and kin.toomre_q_target is not None:
        return 0.4
    return 1.0


def _diskdf_f_d_cap(model: GalaxyModel) -> float:
    """Clip ``f_d`` during iteration so coarse grids stay within ``cordbh`` validity."""
    if model.grid.nr <= 4000:
        return 1.6
    return 1e3


def solve_diskdf_python(
    model: GalaxyModel,
    work_dir: Path,
    *,
    n_radial_steps: int | None = None,
    n_iterations: int | None = None,
    relax: float | None = None,
) -> DiskCorrectionTable:
    """
    Iterative disk DF correction producing ``cordbh.dat`` (Python ``diskdf`` port).

    Parameters
    ----------
    model : GalaxyModel
        Galaxy with enabled disk and kinematics parameters.
    work_dir : Path
        Directory containing ``dbh.dat`` and ``freqdbh.dat``.
    n_radial_steps, n_iterations : int or None, optional
        Override ``DiskKinematics`` iteration controls.

    Returns
    -------
    DiskCorrectionTable
        Radial ``f_d`` and ``f_sz`` correction splines written to ``cordbh.dat``.
    """
    work_dir = Path(work_dir)
    pot = read_harmonic_potential(work_dir / "dbh.dat")
    freq = read_frequency_table(work_dir / "freqdbh.dat")
    disk = model.disk
    assert disk is not None
    kin = model.disk_kinematics
    nrspl = n_radial_steps or kin.n_radial_steps
    niter = n_iterations or kin.n_iterations
    relax = _diskdf_relaxation(model) if relax is None else float(relax)
    f_d_cap = _diskdf_f_d_cap(model)
    drspl = (disk.outer_radius + 2 * disk.trunc_width) / nrspl
    rr = np.linspace(0.0, drspl * nrspl, nrspl + 1)
    fdrat = np.ones(nrspl + 1, dtype=float)
    fszrat = np.ones(nrspl + 1, dtype=float)
    spline_d = natural_cubic_spline(rr, fdrat)
    spline_sz = natural_cubic_spline(rr, fszrat)
    rcirc_fn = _RcircInterpolator(pot, freq)

    zdisk = disk.scale_height
    active = rr > 0.0
    psi_mid = np.array([evaluate_potential(pot, float(r), 0.0) for r in rr])
    psi_3zd = np.array([evaluate_potential(pot, float(r), 3 * zdisk) for r in rr])
    rho0 = np.array(
        [
            _disk_midplane_density(float(r), 0.0, pot, psi_mid=psi_mid[i], psi_3zd=psi_3zd[i])
            for i, r in enumerate(rr)
        ]
    )
    rhoz = np.array(
        [
            _disk_midplane_density(float(r), zdisk, pot, psi_mid=psi_mid[i], psi_3zd=psi_3zd[i])
            for i, r in enumerate(rr)
        ]
    )

    for _ in range(niter):
        f_d = spline_d(rr)
        f_sz = spline_sz(rr)
        drat = np.ones(nrspl + 1, dtype=float)
        dz2rat = np.ones(nrspl + 1, dtype=float)

        for ir in np.where(active)[0]:
            r = float(rr[ir])
            omega = freq.omega(r)
            sigr = math.sqrt(_sigma_r2_base(r, model))
            vc = r * omega
            dvr = 0.1 * sigr
            vt = vc + (np.arange(1, 102) - 51) * dvr
            d0 = _integrate_vphi(vt, pot, freq, rcirc_fn, spline_d, spline_sz, r, 0.0)
            dz2 = _integrate_vphi(vt, pot, freq, rcirc_fn, spline_d, spline_sz, r, zdisk)
            drat[ir] = min(d0 / max(rho0[ir], 1e-30), 1e6)
            dz2rat[ir] = min(dz2 / max(rhoz[ir], 1e-30), 1e6)

        ratio = np.maximum(drat / np.maximum(dz2rat, 1e-30), 1e-30)
        dens_ratio = np.maximum(rho0, 1e-30) / np.maximum(rhoz, 1e-30)
        fzrat = np.ones_like(drat)
        fzrat[active] = np.log(dens_ratio[active]) / np.maximum(np.log(ratio[active]), 1e-30)
        old_fd = fdrat.copy()
        old_sz = fszrat.copy()
        target_fd = old_fd.copy()
        target_sz = old_sz.copy()
        target_fd[active] /= np.maximum(drat[active], 1e-30)
        target_sz[active] /= np.maximum(np.abs(fzrat[active]), 1e-30)
        if relax < 1.0:
            fdrat = (1.0 - relax) * old_fd + relax * target_fd
            fszrat = (1.0 - relax) * old_sz + relax * target_sz
        else:
            fdrat = target_fd
            fszrat = target_sz
        fdrat = np.clip(fdrat, 1e-3, f_d_cap)
        fszrat = np.clip(fszrat, 1e-3, 1e3)
        fdrat[0] = 1.0
        fszrat[0] = 1.0
        spline_d = natural_cubic_spline(rr, fdrat)
        spline_sz = natural_cubic_spline(rr, fszrat)

    cordbh = work_dir / "cordbh.dat"
    lines = [f"# {kin.sigma_r0:17.7f} {kin.sigma_r_scale:17.7f} {nrspl}"]
    for i in range(nrspl + 1):
        lines.append(f" {rr[i]:17.7f} {fdrat[i]:17.7f} {fszrat[i]:17.7f}")
    cordbh.write_text("\n".join(lines) + "\n")
    return DiskCorrectionTable(
        sigma_r0=kin.sigma_r0,
        sigma_r_scale=kin.sigma_r_scale,
        radius=rr,
        f_d=fdrat,
        f_sz=fszrat,
    )
