"""Plummer softened gravity kernels."""

from __future__ import annotations

import numpy as np

from ntropy.units import G

# Above this particle count, skip O(N²) potential energy (use KE-only drift proxy).
LARGE_N_ENERGY_THRESHOLD = 16_384


def pairwise_softening(eps_i: np.ndarray, eps_j: np.ndarray) -> np.ndarray:
    """
    Symmetric pairwise softening length (Gadget-style mean).

    Parameters
    ----------
    eps_i : ndarray, shape (N,)
        Softening lengths for target particles.
    eps_j : ndarray, shape (N,)
        Softening lengths for source particles.

    Returns
    -------
    h_ij : ndarray, shape (N, N)
        Pairwise softening with ``h_ij = 0.5 * (eps_i + eps_j)``.
    """
    return 0.5 * (eps_i[:, None] + eps_j[None, :])


def softened_acceleration_vectorized(
    pos: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
) -> np.ndarray:
    """
    Vectorized Plummer-softened gravitational acceleration.

    .. math::

       \\mathbf{a}_i = \\sum_j G m_j
       \\frac{\\mathbf{r}_j - \\mathbf{r}_i}
       {(|\\mathbf{r}_j-\\mathbf{r}_i|^2 + h_{ij}^2)^{3/2}}

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Particle positions [kpc].
    mass : ndarray, shape (N,)
        Particle masses.
    eps : ndarray, shape (N,)
        Per-particle softening lengths [kpc].

    Returns
    -------
    acc : ndarray, shape (N, 3)
        Accelerations [kpc / (100 km/s)²] in GalactICS units.

    Notes
    -----
    Complexity is O(N²) in memory and time.  Self-interactions are excluded.
    """
    dr = pos[:, None, :] - pos[None, :, :]
    r2 = np.sum(dr * dr, axis=2)
    h_ij = pairwise_softening(eps, eps)
    h2 = h_ij**2
    denom = (r2 + h2) ** 1.5
    np.fill_diagonal(denom, np.inf)
    factor = G * mass[None, :] / denom
    acc = -(factor[..., None] * dr).sum(axis=1)
    return acc


def softened_acceleration_targets(
    pos: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
    target_indices: np.ndarray,
) -> np.ndarray:
    """
    Vectorized softened acceleration for a subset of target particles.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Particle positions.
    mass : ndarray, shape (N,)
        Particle masses.
    eps : ndarray, shape (N,)
        Softening lengths.
    target_indices : ndarray, shape (N_t,)
        Indices of particles to evaluate.

    Returns
    -------
    acc : ndarray, shape (N_t, 3)
        Accelerations on the requested targets only.

    Notes
    -----
    Complexity is O(N_t × N).  Self-interactions are excluded.
    """
    targets = np.asarray(target_indices, dtype=int)
    t_pos = pos[targets]
    t_eps = eps[targets]
    dr = t_pos[:, None, :] - pos[None, :, :]
    r2 = np.sum(dr * dr, axis=2)
    h_ij = 0.5 * (t_eps[:, None] + eps[None, :])
    h2 = h_ij**2
    denom = (r2 + h2) ** 1.5
    rows = np.arange(len(targets))
    denom[rows, targets] = np.inf
    factor = G * mass[None, :] / denom
    return -(factor[..., None] * dr).sum(axis=1)


def softened_potential_energy(
    pos: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
) -> float:
    """
    Total pairwise softened potential energy.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Particle positions.
    mass : ndarray, shape (N,)
        Particle masses.
    eps : ndarray, shape (N,)
        Softening lengths.

    Returns
    -------
    energy : float
        Softened potential energy (negative for bound pairs).
    """
    n = len(mass)
    energy = 0.0
    for i in range(n - 1):
        mi = mass[i]
        if mi == 0.0:
            continue
        dr = pos[i + 1 :] - pos[i]
        r2 = np.sum(dr * dr, axis=1)
        h = 0.5 * (eps[i] + eps[i + 1 :])
        mj = mass[i + 1 :]
        energy -= G * mi * np.sum(mj / np.sqrt(r2 + h * h))
    return float(energy)


def kinetic_energy(vel: np.ndarray, mass: np.ndarray) -> float:
    """
    Total kinetic energy.

    Parameters
    ----------
    vel : ndarray, shape (N, 3)
        Velocities.
    mass : ndarray, shape (N,)
        Masses.

    Returns
    -------
    energy : float
    """
    return float(0.5 * np.sum(mass * np.sum(vel * vel, axis=1)))


def total_energy(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
    *,
    allow_kinetic_only: bool = True,
) -> float:
    """
    Total energy (kinetic + softened potential).

    For ``N > LARGE_N_ENERGY_THRESHOLD`` (default 16384), returns kinetic
    energy only to avoid O(N²) memory and time.  Tiered diagnostics then
    report kinetic-energy drift, which is still a useful stability proxy.

    Parameters
    ----------
    pos, vel, mass, eps
        Particle state arrays.
    allow_kinetic_only : bool
        When ``True`` (default), large ``N`` uses the KE-only fast path.

    Returns
    -------
    energy : float
    """
    ke = kinetic_energy(vel, mass)
    if allow_kinetic_only and len(mass) > LARGE_N_ENERGY_THRESHOLD:
        return ke
    return ke + softened_potential_energy(pos, mass, eps)
