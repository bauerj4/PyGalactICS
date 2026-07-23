"""MPI domain-decomposed force computation (mpi4py)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ntropy.config import BhOptimizationsConfig
from ntropy.forces.brute import compute_forces_brute
from ntropy.forces.bhtree import BarnesHutTree, compute_forces_bh
from ntropy.forces.bhtree_c import BarnesHutTreeC, extension_available
from ntropy.parallel.domains import domain_slices, sort_by_peano
from ntropy.parallel.local_essential_tree import (
    LetPayload,
    domain_bbox,
    export_let_from_packed,
    export_let_from_python_tree,
    merge_let_payloads,
    root_monopole_from_packed,
)

_MPI_COMM = None
_MPI_AVAILABLE = False
_MPI = None


def _launched_under_mpi() -> bool:
    """True when this process was started by ``mpirun`` / ``mpiexec`` / PMI."""
    return any(
        key in os.environ
        for key in (
            "OMPI_COMM_WORLD_SIZE",
            "OMPI_COMM_WORLD_RANK",
            "PMI_SIZE",
            "PMI_RANK",
            "PMIX_RANK",
            "MPI_LOCALNRANKS",
            "I_MPI_INFO_NUMA_NODE_NUM",
        )
    )


try:
    import mpi4py

    # OpenMPI 5 / mpi4py auto-Init outside mpirun can hang forever (PMIx /
    # interface probing). Notebooks and serial runs must import without Init;
    # mpirun workers keep the default auto-init path.
    if not _launched_under_mpi():
        mpi4py.rc.initialize = False
        mpi4py.rc.finalize = False
    from mpi4py import MPI as _MPI

    _MPI_AVAILABLE = True
    if _MPI.Is_initialized():
        _MPI_COMM = _MPI.COMM_WORLD
except (ImportError, RuntimeError, OSError):
    _MPI = None
    _MPI_COMM = None
    _MPI_AVAILABLE = False


def mpi_available() -> bool:
    """Return True when mpi4py is installed (Init not required)."""
    return _MPI_AVAILABLE


def get_comm():
    """
    Return the world MPI communicator, or None if mpi4py is missing / not Init.

    Serial notebooks never call ``MPI.Init()`` (it can hang outside ``mpirun``).
    Workers launched under ``mpirun`` auto-init on import and get ``COMM_WORLD``.
    """
    if not _MPI_AVAILABLE or _MPI is None:
        return None
    if not _MPI.Is_initialized():
        return None
    return _MPI.COMM_WORLD


def mpi_rank0(comm=None) -> bool:
    """Return True on serial runs or MPI rank 0 (for filesystem I/O)."""
    if comm is None:
        if not _MPI_AVAILABLE:
            return True
        comm = get_comm()
    if comm is None:
        return True
    return int(comm.Get_rank()) == 0


@dataclass
class MpiForceCache:
    """
    Reusable Barnes–Hut state for MPI force evaluation.

    Avoids rebuilding the tree on every substep when ``rebuild=False``
    (honours ``force.rebuild_every`` via :class:`ForceContext`).

    For Gadget-style local trees, also caches the imported LET payload and
    the coarse global (domain-root) monopoles from the last rebuild.
    """

    method: Literal["brute", "bh", "bh_c"] | None = None
    bh_tree: BarnesHutTree | None = field(default=None, repr=False)
    bh_c_tree: BarnesHutTreeC | None = field(default=None, repr=False)
    packed: dict[str, np.ndarray] | None = field(default=None, repr=False)
    local_targets: np.ndarray | None = field(default=None, repr=False)
    let_import: LetPayload | None = field(default=None, repr=False)
    global_roots: np.ndarray | None = field(default=None, repr=False)
    local_trees: bool | None = None

    def clear(self) -> None:
        self.method = None
        self.bh_tree = None
        self.bh_c_tree = None
        self.packed = None
        self.local_targets = None
        self.let_import = None
        self.global_roots = None
        self.local_trees = None


def compute_forces_mpi(
    pos: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
    *,
    method: Literal["brute", "bh", "bh_c", "gpu_direct", "gpu_bh"] = "bh",
    theta: float = 0.5,
    comm=None,
    bh_opts: BhOptimizationsConfig | None = None,
    cache: MpiForceCache | None = None,
    rebuild: bool = True,
    mpi_local_trees: bool = True,
) -> np.ndarray:
    """
    Compute accelerations using MPI domain decomposition.

    Particles are sorted by Morton (Z-order) key and split into contiguous
    domains across MPI ranks, matching the Gadget-2 assignment strategy.
    Each rank evaluates forces on its domain targets; results are assembled
    with ``Allgatherv``.

    When ``mpi_local_trees`` is True (default) and ``method`` is a Barnes–Hut
    backend, each rank builds a **local** octree on its domain particles and
    exchanges a Local Essential Tree (LET) of monopoles / leaf particles with
    peers — the Gadget-style global + local tree model.  Set
    ``mpi_local_trees=False`` to rebuild a full replicated tree on every rank
    (previous behaviour).

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
    method : {'brute', 'bh', 'bh_c', 'gpu_direct', 'gpu_bh'}
        Force evaluation backend.  ``'gpu_direct'`` and ``'gpu_bh'`` require
        CuPy + a CUDA-capable GPU; they fall back to ``'bh_c'`` when unavailable.
    theta : float
        Barnes–Hut opening angle (ignored for brute force).
    comm : MPI communicator, optional
        Defaults to ``MPI.COMM_WORLD``.
    bh_opts : BhOptimizationsConfig, optional
        C Barnes–Hut kernel tuning flags (``bh_c`` only).
    cache : MpiForceCache, optional
        Reusable tree state honouring ``force.rebuild_every``; also holds
        the last LET import for diagnostics.
    rebuild : bool
        When False and a compatible cached tree exists, skip the tree
        build (replicated-tree path only; local trees rebuild every call).
    mpi_local_trees : bool
        Use Gadget-style local trees + LET (default True).

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
        # On single-rank, prefer GPU when available
        return _serial_forces(
            pos, mass, eps, method=method, theta=theta, bh_opts=bh_opts,
            cache=cache, rebuild=rebuild,
        )

    from ntropy.forces import gpu_available as _gpu_avail
    use_gpu = _gpu_avail() and method in ("gpu_direct", "gpu_bh")

    if not _MPI_AVAILABLE:
        raise ImportError(
            "mpi4py is required for MPI parallel runs (mpirun -n N). "
            "Install with: pip install 'ntropy[mpi]'"
        )

    n = len(mass)
    order = sort_by_peano(pos)
    slices = domain_slices(n, size)
    local_targets = order[slices[rank]]

    use_local_trees = bool(mpi_local_trees) and method in ("bh", "bh_c")

    # GPU backend uses the same domain decomposition; it just changes the
    # force-evaluation kernel (tree is still built on CPU, walk on GPU).
    if method in ("gpu_direct", "gpu_bh"):
        use_local_trees = False  # rely on replicated tree path for simplicity

    if use_local_trees:
        local_acc = _forces_local_essential(
            pos,
            mass,
            eps,
            local_targets=local_targets,
            method=method,  # type: ignore[arg-type]
            theta=theta,
            comm=comm,
            rank=rank,
            size=size,
            order=order,
            slices=slices,
            bh_opts=bh_opts,
            cache=cache,
            rebuild=rebuild,
        )
    elif method == "bh":
        local_acc = _forces_replicated_bh(
            pos, mass, eps, local_targets=local_targets, theta=theta,
            cache=cache, rebuild=rebuild,
        )
    elif method == "bh_c":
        local_acc = _forces_replicated_bh_c(
            pos, mass, eps, local_targets=local_targets, theta=theta,
            bh_opts=bh_opts, cache=cache, rebuild=rebuild,
        )
    elif method == "gpu_direct":
        from ntropy.forces.gpu_direct import compute_forces_gpu as _gpu_force
        if cache is not None:
            cache.clear()
        local_acc = _gpu_force(
            pos, mass, eps, target_indices=local_targets
        )
    elif method == "gpu_bh":
        from ntropy.forces.gpu_bh import (
            compute_forces_gpu_bh as _gpu_bh_force,
        )
        if cache is not None:
            cache.clear()
        local_acc = _gpu_bh_force(
            pos, mass, eps, theta=theta, target_indices=local_targets,
        )
    else:
        if cache is not None:
            cache.clear()
        local_acc = compute_forces_brute(
            pos, mass, eps, target_indices=local_targets
        )

    return _assemble_accelerations(comm, local_acc, order, slices, n)


def _assemble_accelerations(comm, local_acc, order, slices, n: int) -> np.ndarray:
    """
    Gather per-rank accelerations into the full array on every rank.

    Uses a buffer-based ``Allgatherv`` with ``MPI.DOUBLE`` (no pickling)
    and undoes the Morton sort so the result matches input particle order.

    Parameters
    ----------
    comm : MPI communicator
        World communicator for the collective.
    local_acc : ndarray, shape (n_local, 3)
        This rank's accelerations in Morton-sorted domain order.
    order : ndarray, shape (n,)
        Morton sort permutation applied to the particle arrays.
    slices : list of slice
        Contiguous domain slices per rank (in sorted order).
    n : int
        Total particle count.

    Returns
    -------
    acc : ndarray, shape (n, 3)
        Full acceleration array in original particle order, identical on
        all ranks.
    """
    counts = np.array(
        [3 * (s.stop - s.start) for s in slices], dtype=np.int64
    )
    displs = np.concatenate(([0], np.cumsum(counts[:-1])))
    acc_sorted = np.empty((n, 3), dtype=np.float64)
    comm.Allgatherv(
        np.ascontiguousarray(local_acc, dtype=np.float64),
        [acc_sorted, counts, displs, _MPI.DOUBLE],
    )
    acc = np.empty((n, 3), dtype=np.float64)
    acc[order] = acc_sorted
    return acc


def _forces_replicated_bh(
    pos, mass, eps, *, local_targets, theta, cache, rebuild,
) -> np.ndarray:
    """
    Replicated-tree fallback for the Python Barnes–Hut backend.

    Every rank builds the identical full tree locally from the replicated
    particle arrays (no broadcast) and walks only its ``local_targets``.
    Honours ``rebuild`` via ``cache`` since the full tree is position-frozen
    between rebuilds.

    Returns
    -------
    acc : ndarray, shape (len(local_targets), 3)
        Accelerations on this rank's domain targets.
    """
    use_cached = (
        not rebuild
        and cache is not None
        and cache.method == "bh"
        and cache.bh_tree is not None
        and cache.local_trees is False
    )
    if use_cached:
        tree = cache.bh_tree
    else:
        tree = BarnesHutTree(pos, mass, eps)
        if cache is not None:
            cache.method = "bh"
            cache.bh_tree = tree
            cache.bh_c_tree = None
            cache.packed = None
            cache.let_import = None
            cache.global_roots = None
            cache.local_trees = False
            cache.local_targets = None
    return compute_forces_bh(
        pos, mass, eps, theta=theta, tree=tree, target_indices=local_targets
    )


def _forces_replicated_bh_c(
    pos, mass, eps, *, local_targets, theta, bh_opts, cache, rebuild,
) -> np.ndarray:
    """
    Replicated-tree fallback for the C Barnes–Hut backend (``bh_c``).

    Same contract as :func:`_forces_replicated_bh` but building a
    :class:`BarnesHutTreeC` and walking with ``accel_targets``.

    Returns
    -------
    acc : ndarray, shape (len(local_targets), 3)
        Accelerations on this rank's domain targets.

    Raises
    ------
    ImportError
        When the C extension is not built.
    """
    if not extension_available():
        raise ImportError(
            "force.method 'bh_c' requires the C Barnes–Hut extension. "
            "Reinstall with: pip install -e src/ntropy"
        )
    use_cached = (
        not rebuild
        and cache is not None
        and cache.method == "bh_c"
        and cache.bh_c_tree is not None
        and cache.local_trees is False
    )
    if use_cached:
        tree_c = cache.bh_c_tree
    else:
        tree_c = BarnesHutTreeC.build(pos, mass, eps, bh_opts=bh_opts)
        if cache is not None:
            cache.method = "bh_c"
            cache.bh_c_tree = tree_c
            cache.packed = None
            cache.bh_tree = None
            cache.let_import = None
            cache.global_roots = None
            cache.local_trees = False
            cache.local_targets = None
    return tree_c.accel_targets(local_targets, theta, pos=pos, eps=eps)


def _forces_local_essential(
    pos: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
    *,
    local_targets: np.ndarray,
    method: Literal["bh", "bh_c"],
    theta: float,
    comm,
    rank: int,
    size: int,
    order: np.ndarray,
    slices: list[slice],
    bh_opts: BhOptimizationsConfig | None,
    cache: MpiForceCache | None,
    rebuild: bool,
) -> np.ndarray:
    """
    Gadget-style local tree walk + LET remote contribution.

    Builds an *export* octree over only this rank's domain particles,
    exchanges LET payloads with peers (one ``alltoall``), then builds one
    **combined force tree** over ``[local particles, imported leaf
    particles, imported monopoles-as-pseudo-particles]`` and evaluates the
    total acceleration in a single Barnes–Hut walk.

    The combined tree matters: with overlapping / adjacent domain AABBs
    (always the case for a centrally concentrated halo) the LET export
    degenerates to mostly leaf particles, and applying those pairwise
    would cost O(n_local × N) per force call — near-brute-force work
    duplicated on every rank.  Walking one combined tree keeps the cost
    at O(n_local log N) regardless of how many leaves the LET imports.

    Local trees are rebuilt every force call.  Domain membership can change
    whenever particles move, so reusing a cached local tree (built on a
    previous subset) would desynchronise MPI collectives across ranks.
    ``rebuild`` / ``force.rebuild_every`` still apply to the replicated-tree
    fallback (``mpi_local_trees=False``).

    Parameters
    ----------
    pos, mass, eps : ndarray
        Full replicated particle arrays.
    local_targets : ndarray, shape (n_local,)
        Global indices of this rank's domain particles.
    method : {'bh', 'bh_c'}
        Barnes–Hut backend for both trees.
    theta : float
        Opening angle for the walk and the LET export criterion.
    comm : MPI communicator
        World communicator.
    rank, size : int
        This rank's index and the total rank count.
    order, slices : ndarray, list of slice
        Unused here (domain assignment is recomputed by the caller);
        kept for signature parity with the replicated path.
    bh_opts : BhOptimizationsConfig, optional
        C kernel tuning flags (``bh_c`` only).
    cache : MpiForceCache, optional
        Updated in place with the latest force tree / LET import.
    rebuild : bool
        Ignored (see above).

    Returns
    -------
    acc : ndarray, shape (n_local, 3)
        Total (local + remote) accelerations on this rank's targets.

    Notes
    -----
    Imported monopoles enter the force tree as point pseudo-particles with
    zero softening; they were accepted at ``size / r_min < theta`` so the
    softening term is negligible at those distances.
    """
    del order, slices  # domain assignment is recomputed by the caller
    n_local = len(local_targets)

    local_pos = np.ascontiguousarray(pos[local_targets], dtype=np.float64)
    local_mass = np.ascontiguousarray(mass[local_targets], dtype=np.float64)
    local_eps = np.ascontiguousarray(eps[local_targets], dtype=np.float64)

    my_lo, my_hi = domain_bbox(pos, local_targets)
    boxes = comm.allgather((my_lo, my_hi))

    if method == "bh_c" and not extension_available():
        raise ImportError(
            "force.method 'bh_c' requires the C Barnes–Hut extension. "
            "Reinstall with: pip install -e src/ntropy"
        )

    # Cheap pre-check: if any peer domain is close enough that our domain's
    # bounding box would fail the opening criterion, the LET export will
    # open down to nearly every leaf (typical for a cuspy halo).  Skip the
    # export entirely and use the replicated full-tree walk.
    let_worth_trying = True
    if n_local > 0:
        my_com = 0.5 * (my_lo + my_hi)
        my_size = float(np.max(my_hi - my_lo))
        for j, (blo, bhi) in enumerate(boxes):
            if j == rank:
                continue
            r_min = float(
                np.linalg.norm(
                    np.maximum(blo - my_com, 0.0) + np.maximum(my_com - bhi, 0.0)
                )
            )
            if r_min <= 0.0 or my_size >= theta * r_min:
                let_worth_trying = False
                break
    let_worth_trying = bool(comm.allreduce(int(let_worth_trying), op=_MPI.LAND))

    if not let_worth_trying:
        if method == "bh_c":
            local_acc = _forces_replicated_bh_c(
                pos,
                mass,
                eps,
                local_targets=local_targets,
                theta=theta,
                bh_opts=bh_opts,
                cache=cache,
                rebuild=rebuild,
            )
        else:
            local_acc = _forces_replicated_bh(
                pos,
                mass,
                eps,
                local_targets=local_targets,
                theta=theta,
                cache=cache,
                rebuild=rebuild,
            )
        if cache is not None:
            cache.local_trees = True  # requested mode; fell back for this call
        return local_acc

    # --- Export tree over local domain particles + LET exchange ---
    local_to_global = np.asarray(local_targets, dtype=np.int64)
    export_tree_c = None
    export_tree_py = None
    root = np.zeros(5, dtype=np.float64)
    nodes = np.zeros((0, 19), dtype=np.float64)
    leaf_indices = np.zeros(0, dtype=np.int32)

    if n_local > 0:
        if method == "bh_c":
            export_tree_c = BarnesHutTreeC.build(
                local_pos, local_mass, local_eps, bh_opts=bh_opts
            )
            packed = export_tree_c.pack_buffers()
            nodes = np.asarray(packed["nodes"], dtype=np.float64)
            leaf_indices = np.asarray(packed["leaf_indices"])
            root = root_monopole_from_packed(nodes)
            # Release the C export tree now; only packed arrays are needed.
            del packed
            export_tree_c = None
        else:
            export_tree_py = BarnesHutTree(local_pos, local_mass, local_eps)
            root_com = export_tree_py.root.com
            root = np.array(
                [
                    root_com[0],
                    root_com[1],
                    root_com[2],
                    export_tree_py.root.mass,
                    export_tree_py.root.size,
                ],
                dtype=np.float64,
            )

    payloads: list[LetPayload] = []
    for j in range(size):
        if j == rank or n_local == 0:
            payloads.append(LetPayload.empty())
            continue
        box_lo, box_hi = boxes[j]
        if method == "bh_c":
            payloads.append(
                export_let_from_packed(
                    nodes, leaf_indices, local_to_global, box_lo, box_hi, theta
                )
            )
        else:
            payloads.append(
                export_let_from_python_tree(
                    export_tree_py, local_to_global, box_lo, box_hi, theta
                )
            )
    # Drop export trees before the collective so C buffers are freed promptly;
    # only the packed LET payloads are needed after this point.
    export_tree_c = None
    export_tree_py = None
    imported = merge_let_payloads(comm.alltoall(payloads))
    del payloads
    global_roots = np.asarray(comm.allgather(root), dtype=np.float64)

    if n_local == 0:
        if cache is not None:
            cache.clear()
        return np.zeros((0, 3), dtype=np.float64)

    # --- Combined force tree: local + imported leaves + monopole pseudo-particles ---
    # One O(n_local log N) walk instead of a dense O(n_local × K) pairwise sum
    # against the imported leaves (which duplicated near-brute-force work on
    # every rank whenever domain boxes overlapped).
    # When the LET failed to prune (imported leaves ≈ all remotes — typical for
    # a centrally concentrated halo with overlapping domain AABBs), skip the
    # near-full combined tree and walk a single replicated full tree on local
    # targets instead (one O(N) build, same cost as mpi_local_trees=False).
    k_leaf = int(imported.particle_indices.size)
    m_mono = int(imported.monopoles.shape[0])
    n_total = int(len(mass))
    let_unpruned = (n_local + k_leaf) >= int(0.9 * n_total)

    if let_unpruned:
        if method == "bh_c":
            force_tree = BarnesHutTreeC.build(pos, mass, eps, bh_opts=bh_opts)
            acc = force_tree.accel_targets(
                local_targets, theta, pos=pos, eps=eps
            )
        else:
            force_tree = BarnesHutTree(pos, mass, eps)
            acc = compute_forces_bh(
                pos,
                mass,
                eps,
                theta=theta,
                tree=force_tree,
                target_indices=local_targets,
            )
    else:
        if k_leaf or m_mono:
            parts_pos = [local_pos]
            parts_mass = [local_mass]
            parts_eps = [local_eps]
            if k_leaf:
                parts_pos.append(pos[imported.particle_indices])
                parts_mass.append(mass[imported.particle_indices])
                parts_eps.append(eps[imported.particle_indices])
            if m_mono:
                parts_pos.append(imported.monopoles[:, :3])
                parts_mass.append(imported.monopoles[:, 3])
                # θ-accepted monopoles are distant; softening is negligible there.
                parts_eps.append(np.zeros(m_mono, dtype=np.float64))
            comb_pos = np.ascontiguousarray(np.concatenate(parts_pos), dtype=np.float64)
            comb_mass = np.ascontiguousarray(np.concatenate(parts_mass), dtype=np.float64)
            comb_eps = np.ascontiguousarray(np.concatenate(parts_eps), dtype=np.float64)
        else:
            comb_pos, comb_mass, comb_eps = local_pos, local_mass, local_eps

        target_idx = np.arange(n_local, dtype=np.int64)
        if method == "bh_c":
            force_tree = BarnesHutTreeC.build(
                comb_pos, comb_mass, comb_eps, bh_opts=bh_opts
            )
            acc = force_tree.accel_targets(
                target_idx, theta, pos=comb_pos, eps=comb_eps
            )
        else:
            force_tree = BarnesHutTree(comb_pos, comb_mass, comb_eps)
            acc = compute_forces_bh(
                comb_pos,
                comb_mass,
                comb_eps,
                theta=theta,
                tree=force_tree,
                target_indices=target_idx,
            )

    if cache is not None:
        cache.method = method
        cache.bh_c_tree = force_tree if method == "bh_c" else None
        cache.bh_tree = force_tree if method == "bh" else None
        cache.packed = None
        cache.local_targets = local_to_global
        cache.let_import = imported
        cache.global_roots = global_roots
        cache.local_trees = True

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
    """
    Serial fallback used for one-rank runs or when mpi4py is missing.

    Honours the same ``cache`` / ``rebuild`` contract as the MPI paths so
    ``force.rebuild_every`` behaves identically with and without MPI.

    Returns
    -------
    acc : ndarray, shape (N, 3)
        Accelerations on every particle.
    """
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
                cache.local_trees = False
                cache.let_import = None
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
            cache.local_trees = False
            cache.let_import = None
    return compute_forces_bh(pos, mass, eps, theta=theta, tree=tree)
