"""Brute-force softened gravity."""

from __future__ import annotations

import numpy as np

from ntropy.softening import (
    softened_acceleration_targets,
    softened_acceleration_vectorized,
)

# Above this N, evaluate the full pairwise sum in target chunks so peak memory
# stays O(CHUNK × N) instead of O(N²) with the (N, N, 3) broadcast.
_BRUTE_CHUNK_THRESHOLD = 4096
_BRUTE_CHUNK_SIZE = 2048


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
    Time complexity is O(N²) or O(N × N_targets). For ``N`` above ~4096 the
    all-targets path is evaluated in chunks to bound peak memory.
    """
    if target_indices is None:
        n = len(mass)
        if n <= _BRUTE_CHUNK_THRESHOLD:
            return softened_acceleration_vectorized(pos, mass, eps)
        acc = np.empty_like(pos, dtype=float)
        for start in range(0, n, _BRUTE_CHUNK_SIZE):
            chunk = np.arange(start, min(start + _BRUTE_CHUNK_SIZE, n))
            acc[chunk] = softened_acceleration_targets(pos, mass, eps, chunk)
        return acc

    return softened_acceleration_targets(pos, mass, eps, target_indices)
