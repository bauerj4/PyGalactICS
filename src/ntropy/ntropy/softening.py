"""Plummer softened gravity kernels."""

from __future__ import annotations

import numpy as np

from ntropy.units import G


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
    h_ij = pairwise_softening(eps, eps)
    for i in range(n - 1):
        dr = pos[i + 1 :] - pos[i]
        r = np.sqrt(np.sum(dr * dr, axis=1))
        h = h_ij[i, i + 1 :]
        energy -= G * mass[i] * np.sum(mass[i + 1 :] / np.sqrt(r * r + h * h))
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
) -> float:
    """
    Total energy (kinetic + softened potential).

    Parameters
    ----------
    pos, vel, mass, eps
        Particle state arrays.

    Returns
    -------
    energy : float
    """
    return kinetic_energy(vel, mass) + softened_potential_energy(pos, mass, eps)
