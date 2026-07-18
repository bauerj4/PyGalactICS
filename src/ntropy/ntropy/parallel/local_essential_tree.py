"""Gadget-style local trees with Local Essential Tree (LET) exchange.

Each MPI rank builds an octree over **only its domain particles**, then exports
a pruned set of monopole nodes and essential leaf particles to every other
rank (the Local Essential Tree).  The receiving rank folds the imported
monopoles (as pseudo-particles) and leaf particles into one **combined force
tree** together with its own particles and evaluates the total acceleration in
a single Barnes–Hut walk — never as a dense pairwise sum, which would
duplicate near-brute-force work on every rank whenever domain boxes overlap
(the norm for centrally concentrated systems).

The Allgathered domain-root monopoles form the coarse **global tree** skeleton;
LET nodes refine domains that fail the opening criterion for a peer's bounding
box.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ntropy.units import G


@dataclass
class LetPayload:
    """
    Local Essential Tree export from one rank to one peer.

    Attributes
    ----------
    monopoles : ndarray, shape (M, 5)
        Accepted pseudo-particle nodes as rows of
        ``(com_x, com_y, com_z, mass, size)`` [kpc, mass units, kpc].
    particle_indices : ndarray, shape (K,)
        Global indices (into the full replicated particle arrays) of leaf
        particles that must be applied pairwise because their node failed
        the opening criterion for the peer's bounding box.
    """

    monopoles: np.ndarray
    particle_indices: np.ndarray

    @staticmethod
    def empty() -> LetPayload:
        """
        Return a payload with zero monopoles and zero particle indices.

        Returns
        -------
        payload : LetPayload
            Empty payload with correctly shaped/dtyped arrays; used for the
            self-rank slot in ``Alltoall`` and for empty domains.
        """
        return LetPayload(
            monopoles=np.zeros((0, 5), dtype=np.float64),
            particle_indices=np.zeros(0, dtype=np.int64),
        )


def domain_bbox(pos: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Axis-aligned bounding box of ``pos[indices]``.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Full particle position array [kpc].
    indices : ndarray, shape (K,)
        Global indices of the domain particles owned by one rank.

    Returns
    -------
    lo, hi : ndarray, shape (3,)
        Per-axis minimum and maximum of the selected positions. Both are
        zero vectors when ``indices`` is empty.
    """
    if len(indices) == 0:
        z = np.zeros(3, dtype=np.float64)
        return z, z
    p = pos[indices]
    return p.min(axis=0).astype(np.float64), p.max(axis=0).astype(np.float64)


def min_dist_point_to_aabb(
    point: np.ndarray,
    box_lo: np.ndarray,
    box_hi: np.ndarray,
) -> float:
    """
    Euclidean distance from ``point`` to the closest point in an AABB.

    Parameters
    ----------
    point : ndarray, shape (3,)
        Query point (a node's centre of mass) [kpc].
    box_lo, box_hi : ndarray, shape (3,)
        Corners of the axis-aligned bounding box [kpc].

    Returns
    -------
    dist : float
        Minimum distance from the point to the box; ``0.0`` when the point
        lies inside (or on) the box.
    """
    delta = np.maximum(box_lo - point, 0.0) + np.maximum(point - box_hi, 0.0)
    return float(np.sqrt(np.dot(delta, delta)))


def _accept_monopole(
    com: np.ndarray,
    size: float,
    box_lo: np.ndarray,
    box_hi: np.ndarray,
    theta: float,
) -> bool:
    """
    Decide whether a node may be exported as a single monopole.

    Parameters
    ----------
    com : ndarray, shape (3,)
        Node centre of mass [kpc].
    size : float
        Node cell side length [kpc].
    box_lo, box_hi : ndarray, shape (3,)
        Peer domain bounding box [kpc].
    theta : float
        Barnes–Hut opening angle.

    Returns
    -------
    accept : bool
        True when ``size / r_min < theta`` holds for *every* point in the
        peer AABB (evaluated at the closest point, the worst case). False
        when the box touches or contains the node's centre of mass.
    """
    r_min = min_dist_point_to_aabb(com, box_lo, box_hi)
    if r_min <= 0.0:
        return False
    return size / r_min < theta


def export_let_from_packed(
    nodes: np.ndarray,
    leaf_indices: np.ndarray,
    local_to_global: np.ndarray,
    box_lo: np.ndarray,
    box_hi: np.ndarray,
    theta: float,
) -> LetPayload:
    """
    Walk a packed C octree and export the LET for one peer bounding box.

    Nodes that satisfy the opening criterion against the peer AABB are
    exported as monopoles; nodes that fail it are opened, and leaves that
    fail contribute their particles (as global indices) for exact pairwise
    evaluation on the peer.

    Parameters
    ----------
    nodes : ndarray, shape (n_nodes, 19)
        Flat node buffer from :meth:`BarnesHutTreeC.pack_buffers`.
    leaf_indices : ndarray
        Local particle indices into the *local* tree arrays.
    local_to_global : ndarray, shape (n_local,)
        Maps local tree particle index → global index in the full arrays.
    box_lo, box_hi : ndarray, shape (3,)
        Peer domain AABB.
    theta : float
        Barnes–Hut opening angle.

    Returns
    -------
    payload : LetPayload
        Monopoles and global leaf-particle indices essential for the peer.
        Empty when ``nodes`` is empty.

    Notes
    -----
    The walk is a vectorized breadth-first frontier over the flat node
    buffer (no per-node Python loop): each level classifies all frontier
    nodes at once into accepted monopoles, exported leaves, and internal
    nodes to open.
    """
    if nodes.size == 0:
        return LetPayload.empty()

    com = nodes[:, 3:6]
    size = nodes[:, 6]
    mass = nodes[:, 7]
    is_leaf = nodes[:, 8] > 0.5
    children = nodes[:, 9:17].astype(np.int64)
    leaf_start = nodes[:, 17].astype(np.int64)
    leaf_count = nodes[:, 18].astype(np.int64)

    # Vectorized min distance from every node COM to the peer AABB.
    delta = np.maximum(box_lo[None, :] - com, 0.0) + np.maximum(
        com - box_hi[None, :], 0.0
    )
    r_min = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    accept = (r_min > 0.0) & (size < theta * r_min)

    mono_sel: list[np.ndarray] = []
    leaf_sel: list[np.ndarray] = []
    frontier = np.array([0], dtype=np.int64)
    while frontier.size:
        f = frontier[mass[frontier] > 0.0]
        acc_f = accept[f]
        mono_sel.append(f[acc_f])
        rest = f[~acc_f]
        lf = is_leaf[rest]
        leaf_sel.append(rest[lf])
        ch = children[rest[~lf]].ravel()
        frontier = ch[ch >= 0]

    mono_nodes = np.concatenate(mono_sel) if mono_sel else np.zeros(0, dtype=np.int64)
    leaf_nodes = np.concatenate(leaf_sel) if leaf_sel else np.zeros(0, dtype=np.int64)

    monopoles = np.column_stack(
        [com[mono_nodes], mass[mono_nodes], size[mono_nodes]]
    ) if mono_nodes.size else np.zeros((0, 5), dtype=np.float64)

    if leaf_nodes.size:
        starts = leaf_start[leaf_nodes]
        counts = leaf_count[leaf_nodes]
        total = int(counts.sum())
        # Ragged arange: flat positions into ``leaf_indices`` for all leaves.
        offsets = np.repeat(np.cumsum(counts) - counts, counts)
        flat = np.arange(total, dtype=np.int64) - offsets + np.repeat(starts, counts)
        gidx = np.asarray(local_to_global, dtype=np.int64)[
            np.asarray(leaf_indices, dtype=np.int64)[flat]
        ]
    else:
        gidx = np.zeros(0, dtype=np.int64)

    return LetPayload(
        monopoles=np.ascontiguousarray(monopoles, dtype=np.float64),
        particle_indices=gidx,
    )


def export_let_from_python_tree(
    tree,
    local_to_global: np.ndarray,
    box_lo: np.ndarray,
    box_hi: np.ndarray,
    theta: float,
) -> LetPayload:
    """
    LET export from a pure-Python :class:`~ntropy.forces.bhtree.BarnesHutTree`.

    Recursive equivalent of :func:`export_let_from_packed` for the Python
    Barnes–Hut backend (``force.method: bh``).

    Parameters
    ----------
    tree : BarnesHutTree
        Local tree built over one rank's domain particles.
    local_to_global : ndarray, shape (n_local,)
        Maps local tree particle index → global index in the full arrays.
    box_lo, box_hi : ndarray, shape (3,)
        Peer domain AABB.
    theta : float
        Barnes–Hut opening angle.

    Returns
    -------
    payload : LetPayload
        Monopoles and global leaf-particle indices essential for the peer.
    """
    monos: list[list[float]] = []
    gidx: list[int] = []

    def walk(node) -> None:
        if node.mass <= 0.0:
            return
        if _accept_monopole(node.com, node.size, box_lo, box_hi, theta):
            monos.append(
                [
                    float(node.com[0]),
                    float(node.com[1]),
                    float(node.com[2]),
                    float(node.mass),
                    float(node.size),
                ]
            )
            return
        if node.is_leaf:
            for loc in node.leaf_particles:
                gidx.append(int(local_to_global[int(loc)]))
            return
        for child in node.children:
            if child is not None:
                walk(child)

    walk(tree.root)
    return LetPayload(
        monopoles=np.asarray(monos, dtype=np.float64).reshape(-1, 5),
        particle_indices=np.asarray(gidx, dtype=np.int64),
    )


def merge_let_payloads(payloads: list[LetPayload]) -> LetPayload:
    """
    Concatenate LET imports from every peer rank.

    Parameters
    ----------
    payloads : list of LetPayload
        One payload per rank as returned by ``comm.alltoall`` (the
        self-rank entry is expected to be empty).

    Returns
    -------
    merged : LetPayload
        Single payload with all monopole rows and particle indices
        stacked; empty payload when the list is empty.
    """
    if not payloads:
        return LetPayload.empty()
    monos = [p.monopoles for p in payloads if p.monopoles.size]
    parts = [p.particle_indices for p in payloads if p.particle_indices.size]
    return LetPayload(
        monopoles=(
            np.concatenate(monos, axis=0)
            if monos
            else np.zeros((0, 5), dtype=np.float64)
        ),
        particle_indices=(
            np.concatenate(parts, axis=0)
            if parts
            else np.zeros(0, dtype=np.int64)
        ),
    )


def apply_remote_accelerations(
    target_pos: np.ndarray,
    target_eps: np.ndarray,
    *,
    monopoles: np.ndarray,
    source_pos: np.ndarray,
    source_mass: np.ndarray,
    source_eps: np.ndarray,
    source_indices: np.ndarray,
) -> np.ndarray:
    """
    Dense-pairwise reference for the remote LET contribution (tests only).

    The production MPI path folds imported monopoles and leaves into a
    combined Barnes–Hut tree instead (see ``_forces_local_essential`` in
    :mod:`ntropy.parallel.mpi`); this O(n_targets × K) evaluation is kept
    as an exact reference for unit tests and small-K diagnostics.

    Monopoles use the target softening only (matching the C cell interaction).
    Leaf particles use Gadget mean pairwise softening
    ``h = (eps_i + eps_j) / 2``.

    Parameters
    ----------
    target_pos : ndarray, shape (n_targets, 3)
        Positions of this rank's domain particles [kpc].
    target_eps : ndarray, shape (n_targets,)
        Softening lengths of the targets [kpc].
    monopoles : ndarray, shape (M, 5)
        Imported pseudo-particles ``(com_x, com_y, com_z, mass, size)``.
    source_pos, source_mass, source_eps : ndarray
        Full replicated particle arrays; leaf sources are gathered from
        these via ``source_indices``.
    source_indices : ndarray, shape (K,)
        Global indices of imported LET leaf particles.

    Returns
    -------
    acc : ndarray, shape (n_targets, 3)
        Remote acceleration contribution [code units, G = 1]; add to the
        local-tree self-contribution for the total.
    """
    n_tgt = len(target_pos)
    acc = np.zeros((n_tgt, 3), dtype=np.float64)
    if n_tgt == 0:
        return acc

    if monopoles.size:
        com = monopoles[:, :3]
        m = monopoles[:, 3]
        dr = com[None, :, :] - target_pos[:, None, :]
        r2 = np.sum(dr * dr, axis=2)
        h2 = target_eps[:, None] ** 2
        denom = (r2 + h2) ** 1.5
        good = denom > 0.0
        safe = np.where(good, denom, 1.0)
        contrib = G * m[None, :, None] * dr / safe[:, :, None]
        acc += np.where(good[:, :, None], contrib, 0.0).sum(axis=1)

    if source_indices.size:
        sp = source_pos[source_indices]
        sm = source_mass[source_indices]
        se = source_eps[source_indices]
        dr = sp[None, :, :] - target_pos[:, None, :]
        r2 = np.sum(dr * dr, axis=2)
        h = 0.5 * (target_eps[:, None] + se[None, :])
        denom = (r2 + h * h) ** 1.5
        good = (r2 > 0.0) & (denom > 0.0)
        safe = np.where(good, denom, 1.0)
        contrib = G * sm[None, :, None] * dr / safe[:, :, None]
        acc += np.where(good[:, :, None], contrib, 0.0).sum(axis=1)

    return acc


def root_monopole_from_packed(nodes: np.ndarray) -> np.ndarray:
    """
    Extract the root-node monopole from a packed C octree.

    The Allgathered per-rank root monopoles form the coarse global-tree
    skeleton in the Gadget-style local-tree model.

    Parameters
    ----------
    nodes : ndarray, shape (n_nodes, 19)
        Flat node buffer from :meth:`BarnesHutTreeC.pack_buffers`.

    Returns
    -------
    root : ndarray, shape (5,)
        ``(com_x, com_y, com_z, mass, size)`` of the root node, or zeros
        when the tree is empty.
    """
    if nodes.size == 0:
        return np.zeros(5, dtype=np.float64)
    row = nodes[0]
    return np.array(
        [row[3], row[4], row[5], row[7], row[6]],
        dtype=np.float64,
    )
