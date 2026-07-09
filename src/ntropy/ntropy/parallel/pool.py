"""Parallel force dispatch (MPI backend)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ntropy.parallel.mpi import compute_forces_mpi


def compute_forces_parallel(
    pos: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
    *,
    method: Literal["brute", "bh", "bh_c"] = "bh",
    theta: float = 0.5,
    n_workers: int = 1,
) -> np.ndarray:
    """
    Compute forces using MPI domain decomposition.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Particle positions.
    mass : ndarray, shape (N,)
        Particle masses.
    eps : ndarray, shape (N,)
        Softening lengths.
    method : {'brute', 'bh', 'bh_c'}
        Force backend (``bh_c`` uses the C Barnes–Hut extension).
    theta : float
        Barnes–Hut opening angle.
    n_workers : int
        Retained for JSON compatibility. When running under ``mpirun``,
        the MPI communicator size overrides this value.

    Returns
    -------
    acc : ndarray, shape (N, 3)
        Accelerations.
    """
    del n_workers  # MPI communicator size defines worker count
    return compute_forces_mpi(pos, mass, eps, method=method, theta=theta)
