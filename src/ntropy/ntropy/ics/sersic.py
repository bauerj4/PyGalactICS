"""Truncated spherical Sersic bulge initial conditions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gamma

from ntropy.ics.spherical import sample_spherical_equilibrium
from ntropy.particles import ParticleState


@dataclass
class SersicParams:
    """Spherical Sersic bulge parameters (GalactICS units)."""

    mass: float = 10.0
    n: float = 4.0
    r_e: float = 0.5
    r_trunc: float = 20.0
    eps: float = 0.02
    n_particles: int = 128


def sersic_density(r: np.ndarray, mass: float, n: float, r_e: float, r_trunc: float) -> np.ndarray:
    """Truncated spherical Sersic density profile."""
    r_safe = np.maximum(r, 1e-12)
    b_n = 2.0 * n - 1.0 / 3.0 + 0.009877 / n
    rho0 = mass / (
        4.0
        * np.pi
        * r_e**3
        * n
        * gamma(3.0 * n)
        / (b_n ** (3.0 * n))
    )
    rho = rho0 * r_safe ** (-1.0 / n) * np.exp(-(r_safe / r_e) ** (1.0 / n))
    cutoff = np.exp(-r_safe / r_trunc)
    return rho * cutoff


def sample_sersic(
    params: SersicParams | None = None,
    *,
    seed: int = 42,
) -> ParticleState:
    """Sample self-gravitating truncated Sersic equilibrium particles."""
    p = params or SersicParams()
    rng = np.random.default_rng(seed)
    r_grid = np.logspace(-2, np.log10(p.r_trunc), 500)
    rho_grid = sersic_density(r_grid, p.mass, p.n, p.r_e, p.r_trunc)
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
