"""
C-accelerated Barnes–Hut tree forces for ntropy.

This module is the **Python façade** over the compiled extension
``ntropy.forces._bh_c``, which implements the octree in ``forces/c/bh_tree.c``.
Use it when you need the same physics as :mod:`ntropy.forces.bhtree` but with
native build/walk kernels and MPI-friendly flat buffers.

Physics (identical to the Python reference)
-------------------------------------------
Monopole Barnes–Hut with Gadget-style Plummer softening (:math:`G = 1`):

* **Leaf opening** — pairwise softened sum over particles in an opened leaf,
  with :math:`h_{ij} = \\tfrac{1}{2}(\\varepsilon_i + \\varepsilon_j)`.
* **Cell opening** — if :math:`s / r < \\theta`, replace the subtree with its
  monopole at the center of mass, softened with the target's
  :math:`\\varepsilon_i` only.
* **Self-interaction** — excluded for each target.

MPI workflow (phase 1)
----------------------
See ``forces/c/PARALLEL.md``. Typical calls::

    tree = BarnesHutTreeC.build(pos, mass, eps)   # rank 0
    packed = tree.pack_buffers()
    packed = comm.bcast(packed, root=0)
    tree = BarnesHutTreeC.from_packed(packed)       # all ranks
    acc_local = tree.accel_targets(local_targets, theta)

Configuration
-------------
Select ``"bh_c"`` in ``ForceConfig.method`` or pass ``method="bh_c"`` to
:func:`ntropy.parallel.mpi.compute_forces_mpi`.

Build requirement
-----------------
The extension is compiled on ``pip install -e src/ntropy`` (requires a C
compiler and NumPy headers). Call :func:`extension_available` before relying
on the backend in notebooks or CI.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from ntropy.forces import _bh_c
except ImportError:  # pragma: no cover - extension not built
    _bh_c = None  # type: ignore[assignment]

# Width of each packed node row (see NODE_PACK_WIDTH in bh_module.c).
NODE_PACK_WIDTH = 19


def extension_available() -> bool:
    """
    Return whether the compiled ``_bh_c`` extension is importable.

    Returns
    -------
    available : bool
        ``True`` after a successful editable install that built the C sources.

    Notes
    -----
    When ``False``, every public entry point in this module raises
    :exc:`ImportError` with reinstall instructions. Tests skip via
    ``pytest.mark.skipif(not extension_available())``.
    """
    return _bh_c is not None


def _require_extension() -> Any:
    if _bh_c is None:
        raise ImportError(
            "ntropy C Barnes–Hut extension is not built. "
            "Reinstall with: pip install -e src/ntropy"
        )
    return _bh_c


class BarnesHutTreeC:
    """
    Barnes–Hut octree backed by the C extension.

    The tree stores particle positions used at **build** time in owned C
    buffers (immune to NumPy reallocations). Optional ``pos`` / ``eps``
    arguments on walk methods override trial coordinates for integrators
    that evaluate forces at shifted positions without rebuilding.

    Parameters
    ----------
    tree : _bh_c.Tree
        Low-level extension object (users should call :meth:`build` or
        :meth:`from_packed` instead of constructing directly).
    pos, mass, eps : ndarray
        Reference particle arrays kept alive for :meth:`pack_buffers` and
        default walk coordinates.

    Attributes
    ----------
    n_nodes : int
        Number of allocated octree nodes (internal + leaves).
    n_particles : int
        Number of source particles in the tree.

    See Also
    --------
    ntropy.forces.bhtree.BarnesHutTree : Pure-Python reference implementation.
    ntropy.forces.c.PARALLEL : MPI parallelization roadmap.

    Notes
    -----
    Packed node rows have shape ``(n_nodes, 19)``:

    +----------+------------------------------------------+
    | Index    | Field                                    |
    +==========+==========================================+
    | 0–2      | Cell center ``center[3]``                |
    | 3–5      | Center of mass ``com[3]``                |
    | 6        | Side length ``size``                     |
    | 7        | Aggregated mass                          |
    | 8        | ``is_leaf`` (1.0 or 0.0)                 |
    | 9–16     | Child node indices (``-1`` if absent)    |
    | 17       | ``leaf_start`` index into ``leaf_indices`` |
    | 18       | ``leaf_count``                           |
    +----------+------------------------------------------+
    """

    def __init__(self, tree: Any, *, pos: np.ndarray, mass: np.ndarray, eps: np.ndarray):
        self._tree = tree
        self._pos = np.ascontiguousarray(pos, dtype=float)
        self._mass = np.ascontiguousarray(mass, dtype=float)
        self._eps = np.ascontiguousarray(eps, dtype=float)

    @classmethod
    def build(
        cls,
        pos: np.ndarray,
        mass: np.ndarray,
        eps: np.ndarray | float,
    ) -> BarnesHutTreeC:
        """
        Build a new octree from particle arrays.

        Parameters
        ----------
        pos : ndarray, shape (N, 3)
            Source positions at build time [kpc]. Must be convertible to
            C-contiguous ``float64``.
        mass : ndarray, shape (N,)
            Source masses.
        eps : float or ndarray, shape (N,)
            Per-particle softening lengths [kpc]. A scalar is broadcast.

        Returns
        -------
        tree : BarnesHutTreeC
            Tree ready for :meth:`accel_all` or :meth:`accel_targets`.

        Notes
        -----
        Complexity is ``O(N log N)`` for typical halos. The C builder copies
        ``pos``, ``mass``, and ``eps`` into heap buffers owned by the tree so
        later NumPy reallocations cannot invalidate internal pointers.

        Examples
        --------
        >>> tree = BarnesHutTreeC.build(pos, mass, eps)  # doctest: +SKIP
        >>> acc = tree.accel_all(theta=0.5)  # doctest: +SKIP
        """
        bh = _require_extension()
        pos_a = np.ascontiguousarray(pos, dtype=float)
        mass_a = np.ascontiguousarray(mass, dtype=float)
        if isinstance(eps, (int, float)):
            eps_a = np.full(len(mass_a), float(eps), dtype=float)
        else:
            eps_a = np.ascontiguousarray(eps, dtype=float)
        tree = bh.build_tree(pos_a, mass_a, eps_a)
        return cls(tree, pos=pos_a, mass=mass_a, eps=eps_a)

    @classmethod
    def from_packed(cls, packed: dict[str, np.ndarray]) -> BarnesHutTreeC:
        """
        Reconstruct a tree from :meth:`pack_buffers` output.

        Parameters
        ----------
        packed : dict
            Mapping with keys ``nodes``, ``leaf_indices``, ``pos``, ``mass``,
            ``eps``. Typically received via ``MPI.bcast`` on non-root ranks.

        Returns
        -------
        tree : BarnesHutTreeC
            Tree with identical topology to the rank-0 build; **no rebuild**.

        Notes
        -----
        This is the unpack step in MPI phase 1 (see ``forces/c/PARALLEL.md``).
        Node topology is memcpy'd from the flat buffer; particle arrays are
        bound by reference (already replicated on every rank).
        """
        bh = _require_extension()
        tree = bh.tree_from_packed(
            np.ascontiguousarray(packed["nodes"], dtype=float),
            np.ascontiguousarray(packed["leaf_indices"], dtype=np.int32),
            np.ascontiguousarray(packed["pos"], dtype=float),
            np.ascontiguousarray(packed["mass"], dtype=float),
            np.ascontiguousarray(packed["eps"], dtype=float),
        )
        return cls(
            tree,
            pos=packed["pos"],
            mass=packed["mass"],
            eps=packed["eps"],
        )

    @property
    def n_nodes(self) -> int:
        """Number of octree nodes (internal cells + leaves)."""
        return int(self._tree.n_nodes)

    @property
    def n_particles(self) -> int:
        """Number of particles represented in the tree."""
        return int(self._tree.n_particles)

    def accel(
        self,
        target_index: int,
        theta: float,
        pos: np.ndarray | None = None,
        eps: np.ndarray | float | None = None,
    ) -> np.ndarray:
        """
        Softened acceleration on one target particle.

        Parameters
        ----------
        target_index : int
            Index into the particle arrays (0 .. N-1).
        theta : float
            Opening angle; smaller values are more accurate and slower.
        pos : ndarray, shape (N, 3), optional
            Trial positions for the walk (default: build-time positions).
        eps : float or ndarray, shape (N,), optional
            Trial softening lengths (default: build-time ``eps``).

        Returns
        -------
        acc : ndarray, shape (3,)
            Acceleration [kpc / (100 km/s)²] in GalactICS units.
        """
        targets = np.asarray([target_index], dtype=np.int32)
        return self.accel_targets(targets, theta, pos=pos, eps=eps)[0]

    def accel_targets(
        self,
        target_indices: np.ndarray,
        theta: float,
        pos: np.ndarray | None = None,
        eps: np.ndarray | float | None = None,
    ) -> np.ndarray:
        """
        Softened accelerations on a subset of targets.

        Parameters
        ----------
        target_indices : ndarray, shape (N_targets,)
            Particle indices to evaluate. ``int32`` preferred; ``int64`` is
            coerced in the extension.
        theta : float
            Barnes–Hut opening angle.
        pos : ndarray, shape (N, 3), optional
            Positions used during the tree walk. When ``None``, uses the
            positions stored at :meth:`build` (copied into C memory).
        eps : float or ndarray, shape (N,), optional
            Softening lengths for the walk.

        Returns
        -------
        acc : ndarray, shape (N_targets, 3)
            Accelerations in GalactICS units.

        Notes
        -----
        This is the MPI domain-decomposition entry point: each rank passes
        its Morton-sorted ``local_targets`` slice while all ranks share the
        same global tree from :meth:`from_packed`.

        Complexity is roughly ``O(N_targets log N)`` for balanced trees.
        """
        targets = np.ascontiguousarray(target_indices, dtype=np.int32)
        pos_use = self._pos if pos is None else np.ascontiguousarray(pos, dtype=float)
        if eps is None:
            eps_use = self._eps
        elif isinstance(eps, (int, float)):
            eps_use = np.full(len(pos_use), float(eps), dtype=float)
        else:
            eps_use = np.ascontiguousarray(eps, dtype=float)
        return self._tree.accel_targets(
            targets,
            float(theta),
            pos_use,
            eps_use,
        )

    def accel_all(
        self,
        theta: float,
        pos: np.ndarray | None = None,
        eps: np.ndarray | float | None = None,
    ) -> np.ndarray:
        """
        Softened accelerations on every particle.

        Parameters
        ----------
        theta : float
            Barnes–Hut opening angle.
        pos, eps : optional
            Same semantics as :meth:`accel_targets`.

        Returns
        -------
        acc : ndarray, shape (N, 3)
            Accelerations for all particles.
        """
        pos_use = self._pos if pos is None else np.ascontiguousarray(pos, dtype=float)
        if eps is None:
            eps_use = self._eps
        elif isinstance(eps, (int, float)):
            eps_use = np.full(len(pos_use), float(eps), dtype=float)
        else:
            eps_use = np.ascontiguousarray(eps, dtype=float)
        return self._tree.accel_all(float(theta), pos_use, eps_use)

    def pack_buffers(self) -> dict[str, np.ndarray]:
        """
        Export flat buffers for MPI broadcast.

        Returns
        -------
        packed : dict[str, ndarray]
            ``nodes`` : (n_nodes, 19) ``float64`` packed node rows.
            ``leaf_indices`` : (n_leaf,) ``int32`` particle index pool.
            ``meta`` : (2,) ``int64`` with ``[n_nodes, n_leaf_indices]``.
            ``pos``, ``mass``, ``eps`` : particle arrays (references).

        Notes
        -----
        Broadcast the dict with ``comm.bcast(packed, root=0)``; non-root ranks
        pass the result to :meth:`from_packed`. Buffer layout is documented in
        the class docstring and in ``forces/c/PARALLEL.md``.

        The packed format is versioned implicitly by ``NODE_PACK_WIDTH`` (19).
        """
        raw = self._tree.pack_buffers()
        return {
            "nodes": np.asarray(raw["nodes"], dtype=float),
            "leaf_indices": np.asarray(raw["leaf_indices"], dtype=np.int32),
            "meta": np.asarray(raw["meta"], dtype=np.int64),
            "pos": np.asarray(raw["pos"], dtype=float),
            "mass": np.asarray(raw["mass"], dtype=float),
            "eps": np.asarray(raw["eps"], dtype=float),
        }


def compute_forces_bh_c(
    pos: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
    theta: float = 0.5,
    tree: BarnesHutTreeC | None = None,
    target_indices: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute Barnes–Hut accelerations with the C backend.

    Convenience wrapper matching the signature of
    :func:`ntropy.forces.bhtree.compute_forces_bh`.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Particle positions.
    mass : ndarray, shape (N,)
        Particle masses.
    eps : ndarray, shape (N,)
        Per-particle softening lengths.
    theta : float, default 0.5
        Opening angle criterion (:math:`s/r < \\theta` opens a cell).
    tree : BarnesHutTreeC, optional
        Pre-built tree. Built automatically when ``None``.
    target_indices : ndarray, optional
        Subset of targets (default: all particles). Shape ``(N_targets,)``.

    Returns
    -------
    acc : ndarray, shape (N, 3) or (N_targets, 3)
        Accelerations in GalactICS units.

    Raises
    ------
    ImportError
        If the ``_bh_c`` extension was not compiled.

    See Also
    --------
    compute_forces_bh : Pure-Python equivalent (same physics, slower).
    ntropy.parallel.mpi.compute_forces_mpi : MPI driver with ``method="bh_c"``.

    Notes
    -----
    When ``tree`` is supplied, pass updated ``pos`` / ``eps`` each call if
    coordinates changed since the build (leapfrog drift-kick pattern).

    Examples
    --------
    >>> acc = compute_forces_bh_c(pos, mass, eps, theta=0.3)  # doctest: +SKIP
    """
    local_tree = tree or BarnesHutTreeC.build(pos, mass, eps)
    if target_indices is None:
        return local_tree.accel_all(theta, pos=pos, eps=eps)
    return local_tree.accel_targets(target_indices, theta, pos=pos, eps=eps)
