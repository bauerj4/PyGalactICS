"""Pack harmonic potential tables for the OpenMP C samplers."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from galacticsics.io import read_harmonic_potential
from galacticsics.io.formats import (
    cordbh_is_valid,
    read_disk_correction,
    read_frequency_table,
)
from galacticsics.potential.harmonics import HarmonicPotential
from galacticsics.sampling.equilibrium import read_equilibrium_spatial_profile


def _plcon_array(pot: HarmonicPotential) -> np.ndarray:
    n_harm = pot.n_harmonics
    return np.array([pot.plcon(2 * i) for i in range(n_harm)], dtype=np.float64)


def _dense_freq_table(freq, r_max: float, n: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius = np.linspace(0.0, r_max, n, dtype=np.float64)
    omega = np.array([freq.omega(float(r)) for r in radius], dtype=np.float64)
    kappa = np.array([freq.kappa(float(r)) for r in radius], dtype=np.float64)
    return radius, omega, kappa


def _dense_corr_table(corr, r_max: float, n: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius = np.linspace(0.0, r_max, n, dtype=np.float64)
    f_d = np.array([corr.f_d_at(float(r)) for r in radius], dtype=np.float64)
    f_sz = np.array([corr.f_sz_at(float(r)) for r in radius], dtype=np.float64)
    return radius, f_d, f_sz


def _pack_rcirc_table(pot: HarmonicPotential, freq, *, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Angular-momentum → circular-radius table matching ``_RcircInterpolator``."""
    grid = pot.model.grid
    nr = grid.nr
    dr = grid.dr
    am = np.zeros(nr, dtype=np.float64)
    inv_sqrt_am = np.zeros(nr, dtype=np.float64)
    for i in range(nr):
        r = i * dr
        omega = freq.omega(max(r, 1e-10))
        am[i] = omega * r * r
        inv_sqrt_am[i] = 1.0 / math.sqrt(max(omega, 1e-30))
    return am, inv_sqrt_am


def pack_potential(pot: HarmonicPotential) -> dict:
    disk = pot.model.disk
    has_disk = int(disk is not None and disk.enabled)
    disk_const = (
        disk.disk_const
        if disk is not None and has_disk and disk.scale_length > 0
        else 0.0
    )
    disk_rd = disk.scale_length if disk is not None else 1.0
    disk_zd = disk.scale_height if disk is not None else 1.0
    disk_rtrunc = disk.outer_radius if disk is not None else 0.0
    disk_trunc_width = disk.trunc_width if disk is not None else 1.0
    if has_disk:
        from galacticsics.potential.poisson.disk_vertical import (
            cached_disk_vertical_normalization,
        )

        disk_norm_radii, disk_norm_values = cached_disk_vertical_normalization(pot)
    else:
        disk_norm_radii = np.array([0.0], dtype=np.float64)
        disk_norm_values = np.array([0.0], dtype=np.float64)
    return {
        "apot": np.ascontiguousarray(pot.apot[: pot.n_harmonics, :], dtype=np.float64),
        "plcon": _plcon_array(pot),
        "nr": int(pot.nr),
        "n_harm": int(pot.n_harmonics),
        "lmax": int(pot.lmax),
        "dr": float(pot.dr),
        "psic": float(pot.psic),
        "has_disk": has_disk,
        "disk_const": float(disk_const),
        "disk_rd": float(disk_rd),
        "disk_zd": float(disk_zd),
        "disk_scale_height": float(disk_zd),
        "disk_rtrunc": float(disk_rtrunc),
        "disk_trunc_width": float(disk_trunc_width),
        "disk_vertical_norm_radii": np.ascontiguousarray(
            disk_norm_radii, dtype=np.float64
        ),
        "disk_vertical_norm_values": np.ascontiguousarray(
            disk_norm_values, dtype=np.float64
        ),
    }


def _pack_spatial_profile(work_dir: Path, component: str) -> dict:
    profile = read_equilibrium_spatial_profile(
        work_dir / f"{component}_spatial_profile.npz"
    )
    if profile.component != component:
        raise ValueError(
            f"spatial profile component mismatch: expected {component}, "
            f"found {profile.component}"
        )
    return {
        "spatial_radius": profile.radius,
        "spatial_envelope": profile.envelope,
        "spatial_cdf": profile.radial_cdf,
        "spatial_psi": profile.psi,
        "spatial_density": profile.density,
        "spatial_mass": profile.mass,
    }


def pack_halo_context(work_dir: Path) -> dict:
    work_dir = Path(work_dir)
    pot = read_harmonic_potential(work_dir / "dbh.dat")
    halo = pot.model.halo
    if halo is None:
        raise ValueError("halo component missing")

    energies = []
    log_df = []
    for line in (work_dir / "dfnfw.dat").read_text().splitlines():
        e, ld = line.split()[:2]
        energies.append(float(e))
        log_df.append(float(ld))
    energies_arr = np.asarray(energies, dtype=np.float64)
    log_df_arr = np.asarray(log_df, dtype=np.float64)
    # dfnfw.dat energies are descending; the C sampler's interp1d (and scipy
    # with assume_sorted=True) require ascending abscissas.
    order = np.argsort(energies_arr)
    energies_arr = np.ascontiguousarray(energies_arr[order])
    log_df_arr = np.ascontiguousarray(log_df_arr[order])
    log_interp = interp1d(
        energies_arr,
        log_df_arr,
        kind="linear",
        bounds_error=False,
        fill_value=(float(log_df_arr[0]), float(log_df_arr[-1])),
        assume_sorted=True,
    )
    # Lowered-DF cutoff at the truncation potential (legacy genhalo convention).
    fcut = float(np.exp(log_interp(pot.psic)))
    f_sorted = np.maximum(np.exp(log_df_arr) - fcut, 0.0)
    fmax_cum = np.ascontiguousarray(np.maximum.accumulate(f_sorted), dtype=np.float64)

    ctx = pack_potential(pot)
    ctx.update(
        {
            "df_energy": energies_arr,
            "df_log_df": log_df_arr,
            "df_fmax_cum": fmax_cum,
            "df_fcut": fcut,
        }
    )
    ctx.update(_pack_spatial_profile(work_dir, "halo"))
    ctx["halomass"] = ctx["spatial_mass"]
    ctx["haloedge"] = float(ctx["spatial_radius"][-1])
    return ctx


def pack_disk_context(work_dir: Path) -> dict:
    work_dir = Path(work_dir)
    if not cordbh_is_valid(work_dir / "cordbh.dat"):
        raise RuntimeError(
            "invalid disk DF correction table; rerun diskdf before sampling"
        )
    pot = read_harmonic_potential(work_dir / "dbh.dat")
    freq = read_frequency_table(work_dir / "freqdbh.dat")
    corr = read_disk_correction(work_dir / "cordbh.dat")
    disk = pot.model.disk
    if disk is None:
        raise ValueError("disk component missing")

    rd = 1.2 * disk.scale_length
    zd = 1.2 * disk.scale_height
    rtrunc = disk.outer_radius + 2 * disk.trunc_width

    from galacticsics.sampling.disk_envelope import disk_rejection_envelope

    rhomax, rhomin = disk_rejection_envelope(
        pot,
        radial_limit=rtrunc,
        proposal_scale_length=rd,
        proposal_scale_height=zd,
    )

    # A velocity proposal at R <= rtrunc can have a guiding-centre radius well
    # outside rtrunc.  Packing frequencies only to the position cutoff clamps
    # Omega/kappa there and can manufacture enormous, spurious outer-disk DF
    # peaks.  Cover the full solved frequency grid, at no coarser than its
    # native radial sampling.
    freq_limit = float(freq.radius[-1])
    freq_n = max(512, int(freq.radius.size))
    freq_r, freq_o, freq_k = _dense_freq_table(freq, freq_limit, n=freq_n)
    corr_r, corr_fd, corr_fsz = _dense_corr_table(corr, rtrunc)
    rcirc_am, rcirc_inv = _pack_rcirc_table(pot, freq)

    ctx = pack_potential(pot)
    ctx.update(
        {
            "corr_radius": corr_r,
            "corr_f_d": corr_fd,
            "corr_f_sz": corr_fsz,
            "sigma_r0": float(corr.sigma_r0),
            "sigma_r_scale": float(corr.sigma_r_scale),
            "freq_radius": freq_r,
            "freq_omega": freq_o,
            "freq_kappa": freq_k,
            "rcirc_am": rcirc_am,
            "rcirc_inv_sqrt_am": rcirc_inv,
            "rd": rd,
            "zd": zd,
            "rtrunc": rtrunc,
            "rhomax": rhomax,
            "rhomin": rhomin,
            "disk_mass": float(disk.mass),
        }
    )
    return ctx


def pack_bulge_context(work_dir: Path) -> dict:
    """Pack final-potential spatial and velocity tables for OpenMP ``genbulge``."""

    work_dir = Path(work_dir)
    pot = read_harmonic_potential(work_dir / "dbh.dat")
    bulge = pot.model.bulge
    if bulge is None or not bulge.enabled:
        raise ValueError("bulge component missing")

    energies = []
    log_df = []
    for line in (work_dir / "dfsersic.dat").read_text().splitlines():
        e, ld = line.split()[:2]
        energies.append(float(e))
        log_df.append(float(ld))
    energies_arr = np.asarray(energies, dtype=np.float64)
    log_df_arr = np.asarray(log_df, dtype=np.float64)
    order = np.argsort(energies_arr)
    energies_arr = np.ascontiguousarray(energies_arr[order])
    log_df_arr = np.ascontiguousarray(log_df_arr[order])
    log_interp = interp1d(
        energies_arr,
        log_df_arr,
        kind="linear",
        bounds_error=False,
        fill_value=(float(log_df_arr[0]), float(log_df_arr[-1])),
        assume_sorted=True,
    )
    fcut = float(np.exp(log_interp(pot.psic)))
    f_sorted = np.maximum(np.exp(log_df_arr) - fcut, 0.0)
    fmax_cum = np.ascontiguousarray(np.maximum.accumulate(f_sorted), dtype=np.float64)

    ctx = pack_potential(pot)
    ctx.update(
        {
            "df_energy": energies_arr,
            "df_log_df": log_df_arr,
            "df_fmax_cum": fmax_cum,
            "df_fcut": fcut,
        }
    )
    ctx.update(_pack_spatial_profile(work_dir, "bulge"))
    ctx["bulgemass"] = ctx["spatial_mass"]
    ctx["bulgeedge"] = float(ctx["spatial_radius"][-1])
    return ctx


def flat_to_particle_dtype(flat: np.ndarray, n_particles: int) -> np.ndarray:
    from galacticsics.sampling.particles import PARTICLE_DTYPE

    data = np.zeros(n_particles, dtype=PARTICLE_DTYPE)
    flat = flat.reshape(n_particles, 7)
    for i, name in enumerate(PARTICLE_DTYPE.names):
        data[name] = flat[:, i]
    return data
