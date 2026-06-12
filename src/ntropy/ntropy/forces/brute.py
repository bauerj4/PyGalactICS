"""Brute-force softened gravity."""

from __future__ import annotations

import numpy as np

from ntropy.softening import (
    softened_acceleration_targets,
    softened_acceleration_vectorized,
)


def compute_forces_brute(
    pos: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
    target_indices: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute Plummer-softened accelerations by direct summation.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Particle positions.
    mass : ndarray, shape (N,)
        Particle masses.
    eps : ndarray, shape (N,)
        Softening lengths.
    target_indices : ndarray, optional
        When provided, only compute accelerations for these particle indices.
        All particles still contribute as interaction sources.

    Returns
    -------
    acc : ndarray, shape (N, 3) or (len(target_indices), 3)
        Accelerations on target particles.

    Notes
    -----
    Time complexity is O(N²) or O(N × N_targets).
    """
    if target_indices is None:
        return softened_acceleration_vectorized(pos, mass, eps)

    return softened_acceleration_targets(pos, mass, eps, target_indices)
