"""MPI domain-decomposed force computation (mpi4py)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ntropy.config import BhOptimizationsConfig
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


def mpi_rank0(comm=None) -> bool:
    """Return True on serial runs or MPI rank 0 (for filesystem I/O)."""
    if comm is None:
        if not _MPI_AVAILABLE:
            return True
        comm = _MPI_COMM
    if comm is None:
        return True
    return comm.Get_rank() == 0


@dataclass
class MpiForceCache:
    """
    Reusable Barnes–Hut state for MPI force evaluation.

    Avoids rebuilding and rebroadcasting the tree on every substep when
    ``rebuild=False`` (honours ``force.rebuild_every`` via :class:`ForceContext`).
    """

    method: Literal["brute", "bh", "bh_c"] | None = None
    bh_tree: BarnesHutTree | None = field(default=None, repr=False)
    bh_c_tree: BarnesHutTreeC | None = field(default=None, repr=False)
    packed: dict[str, np.ndarray] | None = field(default=None, repr=False)

    def clear(self) -> None:
        self.method = None
        self.bh_tree = None
        self.bh_c_tree = None
        self.packed = None


def compute_forces_mpi(
    pos: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
    *,
    method: Literal["brute", "bh", "bh_c"] = "bh",
    theta: float = 0.5,
    comm=None,
    bh_opts: BhOptimizationsConfig | None = None,
    cache: MpiForceCache | None = None,
    rebuild: bool = True,
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
            return _serial_forces(
                pos, mass, eps, method=method, theta=theta, bh_opts=bh_opts,
                cache=cache, rebuild=rebuild,
            )
        comm = _MPI_COMM

    size = comm.Get_size()
    rank = comm.Get_rank()

    if size == 1:
        return _serial_forces(
            pos, mass, eps, method=method, theta=theta, bh_opts=bh_opts,
            cache=cache, rebuild=rebuild,
        )

    if not _MPI_AVAILABLE:
        raise ImportError(
            "mpi4py is required for MPI parallel runs (mpirun -n N). "
            "Install with: pip install 'ntropy[mpi]'"
        )

    n = len(mass)
    order = sort_by_peano(pos)
    slices = domain_slices(n, size)
    local_targets = order[slices[rank]]

    use_cached = (
        not rebuild
        and cache is not None
        and cache.method == method
        and (
            (method == "bh" and cache.bh_tree is not None)
            or (method == "bh_c" and cache.bh_c_tree is not None)
        )
    )

    if method == "bh":
        if use_cached:
            tree = cache.bh_tree
        else:
            tree = BarnesHutTree(pos, mass, eps) if rank == 0 else None
            tree = comm.bcast(tree, root=0)
            if cache is not None:
                cache.method = method
                cache.bh_tree = tree
                cache.bh_c_tree = None
                cache.packed = None
        local_acc = compute_forces_bh(
            pos, mass, eps, theta=theta, tree=tree, target_indices=local_targets
        )
    elif method == "bh_c":
        if not extension_available():
            raise ImportError(
                "force.method 'bh_c' requires the C Barnes–Hut extension. "
                "Reinstall with: pip install -e src/ntropy"
            )
        if use_cached:
            tree_c = cache.bh_c_tree
        else:
            packed = None
            if rank == 0:
                tree_c_root = BarnesHutTreeC.build(pos, mass, eps, bh_opts=bh_opts)
                packed = tree_c_root.pack_buffers()
            packed = comm.bcast(packed, root=0)
            tree_c = BarnesHutTreeC.from_packed(packed, bh_opts=bh_opts)
            if cache is not None:
                cache.method = method
                cache.bh_c_tree = tree_c
                cache.packed = packed
                cache.bh_tree = None
        local_acc = tree_c.accel_targets(
            local_targets, theta, pos=pos, eps=eps
        )
    else:
        if cache is not None:
            cache.clear()
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
    bh_opts: BhOptimizationsConfig | None = None,
    cache: MpiForceCache | None = None,
    rebuild: bool = True,
) -> np.ndarray:
    if method == "brute":
        if cache is not None:
            cache.clear()
        return compute_forces_brute(pos, mass, eps)
    if method == "bh_c":
        if not extension_available():
            raise ImportError(
                "force.method 'bh_c' requires the C Barnes–Hut extension. "
                "Reinstall with: pip install -e src/ntropy"
            )
        use_cached = not rebuild and cache is not None and cache.bh_c_tree is not None
        if use_cached:
            tree = cache.bh_c_tree
        else:
            tree = BarnesHutTreeC.build(pos, mass, eps, bh_opts=bh_opts)
            if cache is not None:
                cache.method = method
                cache.bh_c_tree = tree
                cache.bh_tree = None
                cache.packed = None
        return tree.accel_all(theta, pos=pos, eps=eps)
    use_cached = not rebuild and cache is not None and cache.bh_tree is not None
    if use_cached:
        tree = cache.bh_tree
    else:
        tree = BarnesHutTree(pos, mass, eps)
        if cache is not None:
            cache.method = method
            cache.bh_tree = tree
            cache.bh_c_tree = None
            cache.packed = None
    return compute_forces_bh(pos, mass, eps, theta=theta, tree=tree)
