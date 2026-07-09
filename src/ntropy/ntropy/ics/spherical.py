"""Shared spherical equilibrium sampling utilities."""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.interpolate import interp1d

from ntropy.particles import ParticleState


def build_spherical_potential(
    r_grid: np.ndarray,
    rho_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute enclosed mass and potential on a radial grid."""
    r = np.asarray(r_grid, dtype=float)
    rho = np.asarray(rho_grid, dtype=float)
    m_enc = cumulative_trapezoid(4.0 * np.pi * r**2 * rho, r, initial=0.0)
    psi = np.zeros_like(r)
    for i in range(len(r)):
        integrand = 4.0 * np.pi * r[i:] ** 2 * rho[i:] / np.maximum(r[i:], 1e-30)
        if len(integrand) > 1:
            psi[i] = trapezoid(integrand, r[i:])
    return m_enc, psi


def eddington_df(
    r_grid: np.ndarray,
    rho_grid: np.ndarray,
    n_psi: int = 500,
) -> tuple[np.ndarray, np.ndarray, interp1d]:
    """Numerical Eddington inversion f(E) from spherical rho(r)."""
    r = np.asarray(r_grid, dtype=float)
    rho = np.asarray(rho_grid, dtype=float)
    _, psi_r = build_spherical_potential(r, rho)
    psi0 = psi_r[0]
    psi_grid = np.linspace(0.0, psi0, n_psi)
    rho_of_psi = interp1d(psi_r[::-1], rho[::-1], kind="cubic", fill_value=0.0, bounds_error=False)
    rho_psi = rho_of_psi(psi_grid)
    drho_dpsi = np.gradient(rho_psi, psi_grid)
    d2rho_dpsi2 = np.gradient(drho_dpsi, psi_grid)

    e_grid = psi_grid[1:]
    f_e = np.zeros_like(e_grid)
    for k, e_val in enumerate(e_grid):
        mask = psi_grid <= e_val
        integrand = d2rho_dpsi2[mask] / np.sqrt(np.maximum(e_val - psi_grid[mask], 1e-30))
        integral = trapezoid(integrand, psi_grid[mask]) if mask.sum() > 1 else 0.0
        f_e[k] = (1.0 / np.sqrt(8.0 * np.pi**2)) * integral
        f_e[k] += (1.0 / np.sqrt(8.0 * np.pi**2)) * drho_dpsi[0] / np.sqrt(max(e_val, 1e-30))
        f_e[k] = max(f_e[k], 0.0)
    return e_grid, f_e, rho_of_psi


def abel_df_plummer(mass: float, scale: float, e_grid: np.ndarray) -> np.ndarray:
    """Analytic Plummer DF from Abel transform (Binney & Tremaine eq. 4.46)."""
    a = scale
    m = mass
    q = np.maximum(-e_grid, 1e-30)
    coeff = 24.0 * m / (25.0 * np.pi**3 * a**3)
    return coeff * q ** (3.5)


def sample_spherical_equilibrium(
    r_grid: np.ndarray,
    rho_grid: np.ndarray,
    n_particles: int,
    rng: np.random.Generator,
    *,
    total_mass: float,
    eps: float = 0.02,
) -> ParticleState:
    """Sample isotropic equilibrium particles from tabulated spherical rho."""
    r = np.asarray(r_grid, dtype=float)
    rho = np.asarray(rho_grid, dtype=float)
    m_enc, psi_r = build_spherical_potential(r, rho)
    e_grid, f_e, _ = eddington_df(r, rho)

    mass_shell = 4.0 * np.pi * r**2 * rho
    mass_shell = np.maximum(mass_shell, 0.0)
    cdf_r = cumulative_trapezoid(mass_shell, r, initial=0.0)
    if cdf_r[-1] <= 0:
        raise ValueError("Zero total mass in density profile")
    cdf_r /= cdf_r[-1]
    r_samples = np.interp(rng.random(n_particles), cdf_r, r)

    f_cdf = cumulative_trapezoid(f_e, e_grid, initial=0.0)
    if f_cdf[-1] <= 0:
        raise ValueError("DF integral is zero")
    f_cdf /= f_cdf[-1]
    e_samples = np.interp(rng.random(n_particles), f_cdf, e_grid)
    psi_at_r = interp1d(r, psi_r, kind="linear", fill_value="extrapolate")(r_samples)
    v2 = 2.0 * np.maximum(psi_at_r - e_samples, 0.0)
    v_mag = np.sqrt(v2)

    dir_costheta = rng.uniform(-1.0, 1.0, n_particles)
    dir_phi = rng.uniform(0.0, 2.0 * np.pi, n_particles)
    dir_sintheta = np.sqrt(1.0 - dir_costheta**2)
    vx = v_mag * dir_sintheta * np.cos(dir_phi)
    vy = v_mag * dir_sintheta * np.sin(dir_phi)
    vz = v_mag * dir_costheta

    pos_costheta = rng.uniform(-1.0, 1.0, n_particles)
    pos_phi = rng.uniform(0.0, 2.0 * np.pi, n_particles)
    pos_sintheta = np.sqrt(1.0 - pos_costheta**2)
    x = r_samples * pos_sintheta * np.cos(pos_phi)
    y = r_samples * pos_sintheta * np.sin(pos_phi)
    z = r_samples * pos_costheta

    mass_per = np.full(n_particles, total_mass / n_particles)
    return ParticleState.from_arrays(
        pos=np.column_stack([x, y, z]),
        vel=np.column_stack([vx, vy, vz]),
        mass=mass_per,
        eps=eps,
    )
