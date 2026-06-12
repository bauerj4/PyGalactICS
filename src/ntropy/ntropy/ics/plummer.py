"""Plummer sphere initial conditions via Abel DF."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

from ntropy.ics.spherical import abel_df_plummer, build_spherical_potential, eddington_df
from ntropy.particles import ParticleState


@dataclass
class PlummerParams:
    """Plummer sphere parameters (GalactICS units)."""

    mass: float = 50.0
    a: float = 1.0
    eps: float = 0.02
    n_particles: int = 128


def plummer_density(r: np.ndarray, mass: float, a: float) -> np.ndarray:
    return (3.0 * mass / (4.0 * np.pi * a**3)) * (1.0 + (r / a) ** 2) ** (-2.5)


def plummer_df(mass: float, a: float, e_grid: np.ndarray) -> np.ndarray:
    """Analytic Plummer distribution function (bound E < 0 convention)."""
    return abel_df_plummer(mass, a, e_grid)


def sample_plummer(
    params: PlummerParams | None = None,
    *,
    seed: int = 42,
) -> ParticleState:
    """Sample isotropic Plummer equilibrium using Abel DF."""
    p = params or PlummerParams()
    rng = np.random.default_rng(seed)
    r_grid = np.logspace(-2, 2, 500) * p.a
    rho_grid = plummer_density(r_grid, p.mass, p.a)
    _, psi_r = build_spherical_potential(r_grid, rho_grid)
    e_grid, f_e_num, _ = eddington_df(r_grid, rho_grid)
    f_e_ana = plummer_df(p.mass, p.a, e_grid)

    mass_shell = 4.0 * np.pi * r_grid**2 * rho_grid
    cdf_r = cumulative_trapezoid(mass_shell, r_grid, initial=0.0)
    cdf_r /= cdf_r[-1]
    r_samples = np.interp(rng.random(p.n_particles), cdf_r, r_grid)

    f_use = np.maximum(f_e_num, 1e-30)
    f_cdf = cumulative_trapezoid(f_use, e_grid, initial=0.0)
    f_cdf /= f_cdf[-1]
    e_samples = np.interp(rng.random(p.n_particles), f_cdf, e_grid)
    psi_at_r = interp1d(r_grid, psi_r, kind="linear", fill_value="extrapolate")(r_samples)
    v_mag = np.sqrt(2.0 * np.maximum(psi_at_r - e_samples, 0.0))

    costheta = rng.uniform(-1.0, 1.0, p.n_particles)
    phi = rng.uniform(0.0, 2.0 * np.pi, p.n_particles)
    sintheta = np.sqrt(1.0 - costheta**2)
    vx = v_mag * sintheta * np.cos(phi)
    vy = v_mag * sintheta * np.sin(phi)
    vz = v_mag * costheta

    costheta_pos = rng.uniform(-1.0, 1.0, p.n_particles)
    phi_pos = rng.uniform(0.0, 2.0 * np.pi, p.n_particles)
    sintheta_pos = np.sqrt(1.0 - costheta_pos**2)
    x = r_samples * sintheta_pos * np.cos(phi_pos)
    y = r_samples * sintheta_pos * np.sin(phi_pos)
    z = r_samples * costheta_pos

    mass_per = np.full(p.n_particles, p.mass / p.n_particles)
    state = ParticleState.from_arrays(
        pos=np.column_stack([x, y, z]),
        vel=np.column_stack([vx, vy, vz]),
        mass=mass_per,
        eps=p.eps,
    )
    state.remove_center_of_mass()
    return state
