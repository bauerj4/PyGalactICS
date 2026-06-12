"""Truncated NFW halo initial conditions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ntropy.ics.spherical import sample_spherical_equilibrium
from ntropy.particles import ParticleState


@dataclass
class NFWParams:
    """Truncated NFW halo parameters (GalactICS units)."""

    mass: float = 100.0
    a: float = 10.0
    r_trunc: float = 80.0
    rho0: float | None = None
    eps: float = 0.05
    n_particles: int = 256


def nfw_density(r: np.ndarray, rho0: float, a: float, r_trunc: float) -> np.ndarray:
    """Truncated NFW density with exponential cutoff."""
    r_safe = np.maximum(r, 1e-12)
    rho = rho0 / (r_safe / a * (1.0 + r_safe / a) ** 2)
    cutoff = np.exp(-r_safe / r_trunc)
    return rho * cutoff


def sample_nfw(
    params: NFWParams | None = None,
    *,
    seed: int = 42,
) -> ParticleState:
    """Sample self-gravitating truncated NFW equilibrium particles."""
    p = params or NFWParams()
    rng = np.random.default_rng(seed)
    r_grid = np.logspace(-2, np.log10(p.r_trunc), 500)
    if p.rho0 is None:
        rho0 = p.mass / (4.0 * np.pi * p.a**3 * (np.log(1.0 + p.r_trunc / p.a) - p.r_trunc / (p.r_trunc + p.a)))
    else:
        rho0 = p.rho0
    rho_grid = nfw_density(r_grid, rho0, p.a, p.r_trunc)
    state = sample_spherical_equilibrium(
        r_grid,
        rho_grid,
        p.n_particles,
        rng,
        total_mass=p.mass,
        eps=p.eps,
    )
    state.remove_center_of_mass()
    return state
