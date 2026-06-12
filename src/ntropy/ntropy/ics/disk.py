"""Exponential disk initial conditions (GalactICS / legacy prescription)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import erfc

from ntropy.particles import ParticleState


@dataclass
class ExponentialDiskParams:
    """
    Truncated exponential disk parameters (GalactICS units).

    Parameters
    ----------
    mass : float
        Disk mass [GalactICS mass units].
    scale_length : float
        Radial scale length ``R_d`` [kpc].
    outer_radius : float
        Truncation radius ``R_out`` [kpc].
    scale_height : float
        Vertical scale height ``z_d`` [kpc].
    trunc_width : float
        Truncation width ``ΔR`` [kpc].
    eps : float
        Gravitational softening length.
    n_particles : int
        Number of disk particles.
    """

    mass: float = 20.0
    scale_length: float = 3.0
    outer_radius: float = 30.0
    scale_height: float = 0.3
    trunc_width: float = 2.0
    eps: float = 0.02
    n_particles: int = 256


def disk_surface_density(
    r: np.ndarray,
    *,
    sigma0: float,
    scale_length: float,
    outer_radius: float,
    trunc_width: float,
) -> np.ndarray:
    """
    Truncated exponential surface density (legacy ``gendisk`` / ``dbh.f``).

    .. math::

       \\Sigma(R) = \\Sigma_0 \\exp(-R/R_d)\\,
       \\tfrac{1}{2}\\mathrm{erfc}\\!\\left(\\frac{R-R_{out}}{\\Delta R}\\right)

    Parameters
    ----------
    r : ndarray
        Cylindrical radii ``R`` [kpc].
    sigma0 : float
        Central surface density ``\\Sigma_0 = M / (2\\pi R_d^2)``.
    scale_length : float
        Radial scale ``R_d`` [kpc].
    outer_radius : float
        Outer truncation ``R_out`` [kpc].
    trunc_width : float
        Truncation width ``ΔR`` [kpc].

    Returns
    -------
    sigma : ndarray
        Surface density at each radius.
    """
    r = np.asarray(r, dtype=float)
    t = (r - outer_radius) / trunc_width
    trunc = 0.5 * erfc(t)
    return sigma0 * np.exp(-r / scale_length) * trunc


def exponential_disk_density(
    r: np.ndarray,
    z: np.ndarray,
    params: ExponentialDiskParams,
) -> np.ndarray:
    """
    Three-dimensional exponential disk mass density.

    .. math::

       \\rho(R,z) = \\frac{\\Sigma(R)}{2 z_d}\\mathrm{sech}^2(z/z_d)

    Parameters
    ----------
    r : ndarray
        Cylindrical radius ``R`` [kpc].
    z : ndarray
        Height ``z`` [kpc].
    params : ExponentialDiskParams
        Disk parameters.

    Returns
    -------
    rho : ndarray
        Mass density [GalactICS units / kpc³].
    """
    sigma0 = params.mass / (2.0 * np.pi * params.scale_length**2)
    sigma = disk_surface_density(
        r,
        sigma0=sigma0,
        scale_length=params.scale_length,
        outer_radius=params.outer_radius,
        trunc_width=params.trunc_width,
    )
    z_scaled = z / params.scale_height
    vertical = 0.5 / params.scale_height / np.cosh(z_scaled) ** 2
    return sigma * vertical


def sample_exponential_disk(
    params: ExponentialDiskParams | None = None,
    *,
    seed: int = 42,
) -> ParticleState:
    """
    Sample disk particles from the exponential + sech² density law.

    Radial positions are drawn from the marginal surface-density profile;
    vertical positions from the sech² vertical distribution; azimuth is
    uniform.  Velocities are set to zero (pressure-supported disks are
    evolved self-consistently by ntropy).

    Parameters
    ----------
    params : ExponentialDiskParams or None
        Disk configuration.  Defaults are used when ``None``.
    seed : int
        NumPy random seed.

    Returns
    -------
    state : ParticleState
        Disk particle set in the GalactICS unit system.
    """
    p = params or ExponentialDiskParams()
    rng = np.random.default_rng(seed)
    sigma0 = p.mass / (2.0 * np.pi * p.scale_length**2)

    r_grid = np.linspace(0.0, p.outer_radius + 4.0 * p.trunc_width, 2000)
    sigma_grid = disk_surface_density(
        r_grid,
        sigma0=sigma0,
        scale_length=p.scale_length,
        outer_radius=p.outer_radius,
        trunc_width=p.trunc_width,
    )
    radial_pdf = 2.0 * np.pi * r_grid * sigma_grid
    cdf = cumulative_trapezoid(radial_pdf, r_grid, initial=0.0)
    cdf /= cdf[-1]
    r_samples = np.interp(rng.random(p.n_particles), cdf, r_grid)

    u = rng.random(p.n_particles)
    z_samples = p.scale_height * np.arctanh(2.0 * u - 1.0)
    phi = rng.uniform(0.0, 2.0 * np.pi, p.n_particles)
    x = r_samples * np.cos(phi)
    y = r_samples * np.sin(phi)

    mass_per = np.full(p.n_particles, p.mass / p.n_particles)
    return ParticleState.from_arrays(
        pos=np.column_stack([x, y, z_samples]),
        vel=np.zeros((p.n_particles, 3)),
        mass=mass_per,
        eps=p.eps,
    )
