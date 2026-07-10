"""Python particle samplers (``gendisk`` / ``genhalo`` / ``genbulge``)."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from galacticsics.io import read_harmonic_potential
from galacticsics.io.formats import read_disk_correction, read_frequency_table
from galacticsics.potential.evaluate import evaluate_potential
from galacticsics.sampling.particles import PARTICLE_DTYPE, ParticleSet
from galacticsics.sampling.sampler import SampleConfig


def _rng_from_legacy_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(abs(seed) if seed != 0 else 42)


def _log_sample_progress(
    progress_log: Callable[[str], None] | None,
    n_accepted: int,
    n_target: int,
    label: str,
    *,
    interval: int = 5_000,
) -> None:
    if progress_log and n_accepted > 0 and n_accepted % interval == 0:
        progress_log(f"  {label}: {n_accepted:,}/{n_target:,} particles")


def _center_particles(data: np.ndarray) -> np.ndarray:
    mass = data["mass"]
    total = mass.sum()
    com = np.array(
        [
            (mass * data["x"]).sum() / total,
            (mass * data["y"]).sum() / total,
            (mass * data["z"]).sum() / total,
        ]
    )
    for name, idx in zip(("x", "y", "z"), range(3)):
        data[name] -= com[idx]
    vcom = np.array(
        [
            (mass * data["vx"]).sum() / total,
            (mass * data["vy"]).sum() / total,
            (mass * data["vz"]).sum() / total,
        ]
    )
    for name, idx in zip(("vx", "vy", "vz"), range(3)):
        data[name] -= vcom[idx]
    return data


def sample_disk_python(
    work_dir: Path,
    *,
    n_particles: int,
    seed: int = -1,
    center: bool = True,
    max_attempts: int | None = None,
    progress_log: Callable[[str], None] | None = None,
    config: SampleConfig | None = None,
) -> ParticleSet:
    """Rejection sampling for stellar disk particles."""
    from galacticsics.sampling.openmp import (
        openmp_sampler_status,
        sample_disk_openmp,
        warn_python_sampler_fallback,
    )

    sampler = openmp_sampler_status(config)
    if sampler.is_openmp:
        n_threads = 0 if config is None else config.n_openmp_threads
        try:
            return sample_disk_openmp(
                work_dir,
                n_particles=n_particles,
                seed=seed,
                center=center,
                max_attempts=max_attempts,
                n_threads=n_threads,
                progress_log=progress_log,
            )
        except RuntimeError as exc:
            warn_python_sampler_fallback("gendisk", f"OpenMP failed: {exc}", progress_log=progress_log)
    elif config is None or config.use_openmp:
        warn_python_sampler_fallback("gendisk", sampler.reason, progress_log=progress_log)

    from galacticsics.distribution.diskdf_solve import (
        _RcircInterpolator,
        _disk_midplane_density,
        _diskdf5ez,
        _find_fmax,
        _sigma_r2,
        _sigma_z2,
    )
    from galacticsics.numerics import natural_cubic_spline

    work_dir = Path(work_dir)
    pot = read_harmonic_potential(work_dir / "dbh.dat")
    freq = read_frequency_table(work_dir / "freqdbh.dat")
    corr = read_disk_correction(work_dir / "cordbh.dat")
    rcirc_fn = _RcircInterpolator(pot, freq)
    spline_d = natural_cubic_spline(corr.radius, corr.f_d)
    spline_sz = natural_cubic_spline(corr.radius, corr.f_sz)
    disk = pot.model.disk
    assert disk is not None
    rng = _rng_from_legacy_seed(seed)
    rd = 1.2 * disk.scale_length
    zd = 1.2 * disk.scale_height
    rtrunc = disk.outer_radius + 2 * disk.trunc_width
    mass = disk.mass / n_particles
    eps = 0.01

    r_grid = np.linspace(0.0, rtrunc, 50)
    rhoguess = np.exp(-r_grid / rd)
    rhotst = np.array([_disk_midplane_density(float(r), 0.0, pot) / max(g, 1e-30) for r, g in zip(r_grid, rhoguess)])
    rhomax = float(rhotst.max()) * 1.2
    rhomin = 1e-10 * rhomax
    attempt_limit = max_attempts if max_attempts is not None else max(500_000, 15 * n_particles)

    parts: list[tuple] = []
    attempts = 0
    while len(parts) < n_particles and attempts < attempt_limit:
        attempts += 1
        r_try = 2 * rtrunc
        while r_try > rtrunc:
            u1 = max(rng.random(), 1e-30)
            v1 = rng.random() * 2 - 1
            r_try = rd * (-math.log(u1))
            z_try = zd * math.atanh(max(min(v1, 0.999), -0.999))
        rhoguess = math.exp(-r_try / rd) / math.cosh(z_try / zd) ** 2
        rhotst = _disk_midplane_density(r_try, z_try, pot) / max(rhoguess, 1e-30)
        if rhotst < rhomin or (rhomax - rhomin) * rng.random() > rhotst:
            continue

        phi = rng.uniform(0, 2 * math.pi)
        x = r_try * math.cos(phi)
        y = r_try * math.sin(phi)
        omega = freq.omega(r_try)
        kappa = freq.kappa(r_try)
        vphimax = omega * r_try
        f_d = corr.f_d_at(r_try)
        f_sz = corr.f_sz_at(r_try)
        vsigR = math.sqrt(_sigma_r2(r_try, pot.model, f_d))
        vsigp = kappa / (2 * omega) * vsigR if omega > 0 else 0.0
        vsigz = math.sqrt(_sigma_z2(r_try, pot, disk.scale_height, f_sz))
        fmax = 1.1 * _find_fmax(vphimax, r_try, z_try, pot, freq, rcirc_fn, spline_d, spline_sz, vsigp)
        if fmax <= 0.0:
            continue

        accepted = False
        for _ in range(200):
            gr = 8 * (rng.random() - 0.5)
            gp = 16 * (rng.random() - 0.5)
            gz = 8 * (rng.random() - 0.5)
            if gr * gr / 16 + gp * gp / 64 + gz * gz / 16 > 1:
                continue
            vR = vsigR * gr
            vp = vphimax + vsigp * gp
            vz = vsigz * gz
            f0 = _diskdf5ez(vR, vp, vz, r_try, z_try, pot, freq, rcirc_fn, spline_d, spline_sz)
            if fmax * rng.random() <= f0:
                accepted = True
                break
        if not accepted:
            continue
        parts.append((mass, x, y, z_try, vR, vp, vz))
        _log_sample_progress(progress_log, len(parts), n_particles, "gendisk")

    if len(parts) < n_particles:
        raise RuntimeError(f"disk sampling failed after {attempts} attempts ({len(parts)}/{n_particles})")

    data = np.zeros(n_particles, dtype=PARTICLE_DTYPE)
    for i, row in enumerate(parts):
        for j, name in enumerate(PARTICLE_DTYPE.names):
            data[name][i] = row[j]
    if center:
        data = _center_particles(data)
    return ParticleSet(data, component="disk")


def sample_halo_python(
    work_dir: Path,
    *,
    n_particles: int,
    seed: int = -1,
    center: bool = True,
    streaming: float = 0.5,
    max_attempts: int | None = None,
    progress_log: Callable[[str], None] | None = None,
    config: SampleConfig | None = None,
) -> ParticleSet:
    """Rejection sampling for halo particles."""
    from galacticsics.sampling.openmp import (
        openmp_sampler_status,
        sample_halo_openmp,
        warn_python_sampler_fallback,
    )

    sampler = openmp_sampler_status(config)
    if sampler.is_openmp:
        n_threads = 0 if config is None else config.n_openmp_threads
        try:
            return sample_halo_openmp(
                work_dir,
                n_particles=n_particles,
                seed=seed,
                center=center,
                streaming=streaming,
                max_attempts=max_attempts,
                n_threads=n_threads,
                progress_log=progress_log,
            )
        except RuntimeError as exc:
            warn_python_sampler_fallback("genhalo", f"OpenMP failed: {exc}", progress_log=progress_log)
    elif config is None or config.use_openmp:
        warn_python_sampler_fallback("genhalo", sampler.reason, progress_log=progress_log)

    work_dir = Path(work_dir)
    pot = read_harmonic_potential(work_dir / "dbh.dat")
    masses = (work_dir / "mr.dat").read_text().splitlines()
    halomass = float(masses[2].split()[0])
    haloedge = float(masses[2].split()[1])
    if halomass <= 0 or haloedge <= 0:
        halomass = max(halomass, pot.model.halo.v0**2 * pot.model.halo.r_outer if pot.model.halo else 1.0)
        haloedge = max(haloedge, pot.model.halo.r_outer if pot.model.halo else 100.0)
    psic = pot.psic
    rng = _rng_from_legacy_seed(seed)
    mass = halomass / n_particles
    eps = 0.05

    energies = []
    log_df = []
    for line in (work_dir / "dfnfw.dat").read_text().splitlines():
        e, ld = line.split()[:2]
        energies.append(float(e))
        log_df.append(float(ld))
    energies = np.asarray(energies, dtype=float)
    log_df = np.asarray(log_df, dtype=float)
    log_interp = interp1d(
        energies,
        log_df,
        kind="linear",
        bounds_error=False,
        fill_value=(float(log_df[0]), float(log_df[-1])),
        assume_sorted=True,
    )

    def df_halo(psi: float) -> float:
        if psi <= psic:
            return 0.0
        return float(np.exp(log_interp(psi)))

    fcut = df_halo(psic)
    from galacticsics.potential.poisson.densities import halo_density_spherical

    halo = pot.model.halo
    assert halo is not None
    r_grid = np.linspace(0.0, haloedge, 50)
    rhocur = np.array([halo_density_spherical(float(r), halo) for r in r_grid]) * r_grid * r_grid
    rhomax = float(rhocur.max()) * 1.5
    rhomin = 1e-10 * rhomax
    attempt_limit = max_attempts if max_attempts is not None else max(500_000, 15 * n_particles)

    parts: list[tuple] = []
    attempts = 0
    while len(parts) < n_particles and attempts < attempt_limit:
        attempts += 1
        u1 = haloedge * rng.random()
        v1 = math.pi * (rng.random() * 2 - 1)
        r_cyl = u1
        z = r_cyl * math.tan(v1)
        if abs(z) > 2 * haloedge:
            continue
        rhotst = halo_density_spherical(math.hypot(r_cyl, z), halo) * (r_cyl * r_cyl + z * z)
        if rhotst < rhomin or (rhomax - rhomin) * rng.random() > rhotst:
            continue
        phi = rng.uniform(0, 2 * math.pi)
        x = r_cyl * math.cos(phi)
        y = r_cyl * math.sin(phi)
        psi = evaluate_potential(pot, r_cyl, z)
        if psi < psic:
            continue
        vmax2 = 2 * (psi - psic)
        vmax = math.sqrt(max(vmax2, 0.0))
        fmax = max(df_halo(psi) - fcut, 0.0)
        if fmax <= 0:
            continue
        accepted = False
        inner = 0
        while not accepted:
            inner += 1
            if inner > 10_000:
                break
            v2 = 1.1 * vmax2
            while v2 > vmax2:
                vR = 2 * vmax * (rng.random() - 0.5)
                vp = 2 * vmax * (rng.random() - 0.5)
                vz = 2 * vmax * (rng.random() - 0.5)
                v2 = vR * vR + vp * vp + vz * vz
            energy = psi - 0.5 * v2
            f0 = max(df_halo(energy) - fcut, 0.0)
            if fmax * rng.random() <= f0:
                accepted = True
        if not accepted:
            continue
        if rng.random() < streaming:
            vp = abs(vp)
        else:
            vp = -abs(vp)
        r_cyl_pos = math.hypot(x, y)
        if r_cyl_pos > 0:
            cph, sph = x / r_cyl_pos, y / r_cyl_pos
            vx = vR * cph - vp * sph
            vy = vR * sph + vp * cph
        else:
            vx, vy = vR, vp
        parts.append((mass, x, y, z, vx, vy, vz))
        _log_sample_progress(progress_log, len(parts), n_particles, "genhalo")

    if len(parts) < n_particles:
        raise RuntimeError(f"halo sampling failed after {attempts} attempts ({len(parts)}/{n_particles})")

    data = np.zeros(n_particles, dtype=PARTICLE_DTYPE)
    for i, row in enumerate(parts):
        for j, name in enumerate(PARTICLE_DTYPE.names):
            data[name][i] = row[j]
    if center:
        data = _center_particles(data)
    return ParticleSet(data, component="halo")


def sample_bulge_python(
    work_dir: Path,
    *,
    n_particles: int,
    seed: int = -1,
    center: bool = True,
    streaming: float = 0.0,
    max_attempts: int = 500_000,
) -> ParticleSet:
    """
    Rejection sampling for Sersic bulge particles (``genbulge``).

    Parameters
    ----------
    work_dir : Path
        Directory containing ``dbh.dat``, ``dfsersic.dat``, and ``mr.dat``.
    n_particles : int
        Target particle count.
    seed : int, optional
        Legacy-compatible RNG seed.
    center : bool, optional
        Subtract centre-of-mass position and velocity after acceptance.
    streaming : float, optional
        Fraction of particles with inverted azimuthal velocity.
    max_attempts : int, optional
        Upper bound on rejection trials.

    Returns
    -------
    ParticleSet
        Accepted bulge particles with component tag ``"bulge"``.
    """
    from galacticsics.potential.poisson.sersic import (
        bulge_density_spherical,
        sersic_params_from_bulge,
    )

    work_dir = Path(work_dir)
    pot = read_harmonic_potential(work_dir / "dbh.dat")
    masses = (work_dir / "mr.dat").read_text().splitlines()
    bulgemass = float(masses[1].split()[0])
    bulgeedge = float(masses[1].split()[1])
    bulge = pot.model.bulge
    if bulgemass <= 0.0 or bulgeedge <= 0.0:
        if bulge is not None:
            bulgemass = max(bulgemass, bulge.v0**2 * bulge.a)
            bulgeedge = max(bulgeedge, 3.0 * bulge.a)
    psic = pot.psic
    rng = _rng_from_legacy_seed(seed)
    mass = bulgemass / n_particles
    assert bulge is not None

    energies = []
    log_df = []
    for line in (work_dir / "dfsersic.dat").read_text().splitlines():
        e, ld = line.split()[:2]
        energies.append(float(e))
        log_df.append(float(ld))
    energies = np.asarray(energies, dtype=float)
    log_df = np.asarray(log_df, dtype=float)
    log_interp = interp1d(
        energies,
        log_df,
        kind="linear",
        bounds_error=False,
        fill_value=(float(log_df[0]), float(log_df[-1])),
        assume_sorted=True,
    )

    def df_bulge(psi: float) -> float:
        if psi <= psic:
            return 0.0
        return float(np.exp(log_interp(psi)))

    fcut = df_bulge(psic)
    params = sersic_params_from_bulge(bulge)
    r_grid = np.linspace(0.0, bulgeedge, 50)
    rhocur = np.array(
        [bulge_density_spherical(float(r), bulge, params) for r in r_grid]
    ) * r_grid * r_grid
    rhomax = float(rhocur.max()) * 1.5
    rhomin = 1e-10 * rhomax

    parts: list[tuple] = []
    attempts = 0
    while len(parts) < n_particles and attempts < max_attempts:
        attempts += 1
        u1 = bulgeedge * rng.random()
        v1 = math.pi * (rng.random() * 2 - 1)
        r_cyl = u1
        z = r_cyl * math.tan(v1)
        if abs(z) > 2 * bulgeedge:
            continue
        rad = math.hypot(r_cyl, z)
        rhotst = bulge_density_spherical(rad, bulge, params) * (r_cyl * r_cyl + z * z)
        if rhotst < rhomin or (rhomax - rhomin) * rng.random() > rhotst:
            continue
        phi = rng.uniform(0, 2 * math.pi)
        x = r_cyl * math.cos(phi)
        y = r_cyl * math.sin(phi)
        psi = evaluate_potential(pot, r_cyl, z)
        if psi < psic:
            continue
        vmax2 = 2 * (psi - psic)
        vmax = math.sqrt(max(vmax2, 0.0))
        fmax = max(df_bulge(psi) - fcut, 0.0)
        if fmax <= 0:
            continue
        accepted = False
        while not accepted:
            v2 = 1.1 * vmax2
            while v2 > vmax2:
                vR = 2 * vmax * (rng.random() - 0.5)
                vp = 2 * vmax * (rng.random() - 0.5)
                vz = 2 * vmax * (rng.random() - 0.5)
                v2 = vR * vR + vp * vp + vz * vz
            energy = psi - 0.5 * v2
            f0 = max(df_bulge(energy) - fcut, 0.0)
            if fmax * rng.random() <= f0:
                accepted = True
        if rng.random() < streaming:
            vp = abs(vp)
        else:
            vp = -abs(vp)
        r_cyl_pos = math.hypot(x, y)
        if r_cyl_pos > 0:
            cph, sph = x / r_cyl_pos, y / r_cyl_pos
            vx = vR * cph - vp * sph
            vy = vR * sph + vp * cph
        else:
            vx, vy = vR, vp
        parts.append((mass, x, y, z, vx, vy, vz))

    if len(parts) < n_particles:
        raise RuntimeError(f"bulge sampling failed after {attempts} attempts ({len(parts)}/{n_particles})")

    data = np.zeros(n_particles, dtype=PARTICLE_DTYPE)
    for i, row in enumerate(parts):
        for j, name in enumerate(PARTICLE_DTYPE.names):
            data[name][i] = row[j]
    if center:
        data = _center_particles(data)
    return ParticleSet(data, component="bulge")
