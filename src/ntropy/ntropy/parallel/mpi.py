"""MPI domain-decomposed force computation (mpi4py)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ntropy.forces.brute import compute_forces_brute
from ntropy.forces.bhtree import BarnesHutTree, compute_forces_bh
from ntropy.forces.bhtree_c import BarnesHutTreeC, extension_available
from ntropy.parallel.domains import domain_slices, sort_by_peano

_MPI_COMM = None
_MPI_AVAILABLE = False

try:
    from mpi4py import MPI as _MPI

    _MPI_COMM = _MPI.COMM_WORLD
    _MPI_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    _MPI = None
    _MPI_COMM = None
    _MPI_AVAILABLE = False


def mpi_available() -> bool:
    """Return True when mpi4py is installed."""
    return _MPI_AVAILABLE


def get_comm():
    """Return the world MPI communicator, or None if mpi4py is missing."""
    return _MPI_COMM


def compute_forces_mpi(
    pos: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
    *,
    method: Literal["brute", "bh", "bh_c"] = "bh",
    theta: float = 0.5,
    comm=None,
) -> np.ndarray:
    """
    Compute accelerations using MPI domain decomposition.

    Particles are sorted by Morton (Z-order) key and split into contiguous
    domains across MPI ranks, matching the Gadget-2 assignment strategy.
    Each rank evaluates forces on its domain targets; results are assembled
    with ``allgather``.

    When ``comm.Get_size() == 1`` (plain ``python`` or ``mpirun -n 1``),
    this falls back to the serial force kernel.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Particle positions.
    mass : ndarray, shape (N,)
        Particle masses.
    eps : ndarray, shape (N,)
        Per-particle softening lengths.
    method : {'brute', 'bh', 'bh_c'}
        Force evaluation backend (``bh_c`` uses packed C tree broadcast).
    theta : float
        Barnes–Hut opening angle (ignored for brute force).
    comm : MPI communicator, optional
        Defaults to ``MPI.COMM_WORLD``.

    Returns
    -------
    acc : ndarray, shape (N, 3)
        Acceleration on every particle (identical on all ranks).

    Raises
    ------
    ImportError
        If mpi4py is not installed and ``comm.Get_size() > 1``.
    """
    if comm is None:
        if not _MPI_AVAILABLE:
            return _serial_forces(pos, mass, eps, method=method, theta=theta)
        comm = _MPI_COMM

    size = comm.Get_size()
    rank = comm.Get_rank()

    if size == 1:
        return _serial_forces(pos, mass, eps, method=method, theta=theta)

    if not _MPI_AVAILABLE:
        raise ImportError(
            "mpi4py is required for MPI parallel runs (mpirun -n N). "
            "Install with: pip install 'ntropy[mpi]'"
        )

    n = len(mass)
    order = sort_by_peano(pos)
    slices = domain_slices(n, size)
    local_targets = order[slices[rank]]

    if method == "bh":
        tree = BarnesHutTree(pos, mass, eps) if rank == 0 else None
        tree = comm.bcast(tree, root=0)
        local_acc = compute_forces_bh(
            pos, mass, eps, theta=theta, tree=tree, target_indices=local_targets
        )
    elif method == "bh_c":
        if not extension_available():
            raise ImportError(
                "force.method 'bh_c' requires the C Barnes–Hut extension. "
                "Reinstall with: pip install -e src/ntropy"
            )
        packed = None
        if rank == 0:
            tree_c = BarnesHutTreeC.build(pos, mass, eps)
            packed = tree_c.pack_buffers()
        packed = comm.bcast(packed, root=0)
        tree_c = BarnesHutTreeC.from_packed(packed)
        local_acc = tree_c.accel_targets(
            local_targets, theta, pos=pos, eps=eps
        )
    else:
        local_acc = compute_forces_brute(
            pos, mass, eps, target_indices=local_targets
        )

    gathered_indices = comm.allgather(local_targets)
    gathered_acc = comm.allgather(local_acc)

    acc = np.zeros((n, 3), dtype=float)
    for indices, piece in zip(gathered_indices, gathered_acc):
        acc[indices] = piece
    return acc


def _serial_forces(
    pos: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
    *,
    method: Literal["brute", "bh", "bh_c"],
    theta: float,
) -> np.ndarray:
    if method == "brute":
        return compute_forces_brute(pos, mass, eps)
    if method == "bh_c":
        if not extension_available():
            raise ImportError(
                "force.method 'bh_c' requires the C Barnes–Hut extension. "
                "Reinstall with: pip install -e src/ntropy"
            )
        return BarnesHutTreeC.build(pos, mass, eps).accel_all(theta, pos=pos, eps=eps)
    tree = BarnesHutTree(pos, mass, eps)
    return compute_forces_bh(pos, mass, eps, theta=theta, tree=tree)
