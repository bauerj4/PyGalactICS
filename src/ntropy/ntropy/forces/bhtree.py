"""Barnes-Hut octree force computation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ntropy.softening import pairwise_softening
from ntropy.units import G


_MIN_LEAF_SIZE = 1e-12


@dataclass
class OctreeNode:
    center: np.ndarray
    size: float
    mass: float = 0.0
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))
    leaf_particles: list[int] = field(default_factory=list)
    children: list[OctreeNode | None] = field(default_factory=lambda: [None] * 8)
    is_leaf: bool = True

    def __getstate__(self):
        return {
            "center": self.center,
            "size": self.size,
            "mass": self.mass,
            "com": self.com,
            "leaf_particles": self.leaf_particles,
            "children": self.children,
            "is_leaf": self.is_leaf,
        }

    def __setstate__(self, state):
        self.center = state["center"]
        self.size = state["size"]
        self.mass = state["mass"]
        self.com = state["com"]
        self.leaf_particles = state["leaf_particles"]
        self.children = state["children"]
        self.is_leaf = state["is_leaf"]


def _child_index(center: np.ndarray, point: np.ndarray) -> int:
    idx = 0
    if point[0] >= center[0]:
        idx |= 1
    if point[1] >= center[1]:
        idx |= 2
    if point[2] >= center[2]:
        idx |= 4
    return idx


def _child_center(parent_center: np.ndarray, parent_size: float, child_idx: int) -> np.ndarray:
    offset = parent_size * 0.25
    signs = (
        (1.0 if child_idx & 1 else -1.0),
        (1.0 if child_idx & 2 else -1.0),
        (1.0 if child_idx & 4 else -1.0),
    )
    return parent_center + offset * np.array(signs)


class BarnesHutTree:
    """
    Monopole Barnes–Hut octree for softened gravity.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Particle positions at tree build time.
    mass : ndarray, shape (N,)
        Particle masses.
    eps : ndarray, shape (N,)
        Per-particle softening lengths.

    Notes
    -----
    Nodes store total mass and center-of-mass.  The opening criterion is
    ``size / distance < theta`` (open when true).
    """

    def __init__(self, pos: np.ndarray, mass: np.ndarray, eps: np.ndarray):
        self.pos = pos
        self.mass = mass
        self.eps = eps
        self.root = self._build_tree()

    def _build_tree(self) -> OctreeNode:
        n = len(self.mass)
        if n == 0:
            raise ValueError("Cannot build tree with zero particles")
        mins = self.pos.min(axis=0)
        maxs = self.pos.max(axis=0)
        center = 0.5 * (mins + maxs)
        half = 0.5 * (maxs - mins).max()
        if half == 0:
            half = 1.0
        size = 2.0 * half * 1.01
        root = OctreeNode(center=center.copy(), size=size)
        for i in range(n):
            self._insert(root, i)
        self._aggregate(root)
        return root

    def _update_leaf(self, node: OctreeNode) -> None:
        """Recompute leaf mass and center of mass from ``leaf_particles``."""
        if not node.leaf_particles:
            node.mass = 0.0
            node.com = node.center.copy()
            return
        masses = self.mass[node.leaf_particles]
        node.mass = float(masses.sum())
        if node.mass > 0:
            weighted = np.sum(
                masses[:, None] * self.pos[node.leaf_particles],
                axis=0,
            )
            node.com = weighted / node.mass
        else:
            node.com = self.pos[node.leaf_particles[0]].copy()

    def _coincident_in_leaf(self, node: OctreeNode, particle_index: int) -> bool:
        point = self.pos[particle_index]
        tol = max(1e-15, 1e-9 * node.size)
        for idx in node.leaf_particles:
            if np.linalg.norm(point - self.pos[idx]) <= tol:
                return True
        return False

    def _insert(self, node: OctreeNode, particle_index: int) -> None:
        point = self.pos[particle_index]
        if node.is_leaf:
            if not node.leaf_particles:
                node.leaf_particles.append(particle_index)
                self._update_leaf(node)
                return
            if self._coincident_in_leaf(node, particle_index) or node.size <= _MIN_LEAF_SIZE:
                node.leaf_particles.append(particle_index)
                self._update_leaf(node)
                return
            to_insert = list(node.leaf_particles) + [particle_index]
            node.leaf_particles = []
            node.is_leaf = False
            node.mass = 0.0
            node.com = np.zeros(3)
            for idx in to_insert:
                self._insert_child(node, idx)
            return

        self._insert_child(node, particle_index)

    def _insert_child(self, node: OctreeNode, particle_index: int) -> None:
        point = self.pos[particle_index]
        child_idx = _child_index(node.center, point)
        if node.children[child_idx] is None:
            node.children[child_idx] = OctreeNode(
                center=_child_center(node.center, node.size, child_idx),
                size=0.5 * node.size,
            )
        self._insert(node.children[child_idx], particle_index)

    def _aggregate(self, node: OctreeNode) -> None:
        if node.is_leaf:
            self._update_leaf(node)
            return
        total_mass = 0.0
        com = np.zeros(3)
        for child in node.children:
            if child is None:
                continue
            self._aggregate(child)
            total_mass += child.mass
            com += child.mass * child.com
        node.mass = total_mass
        if total_mass > 0:
            node.com = com / total_mass
        else:
            node.com = node.center.copy()

    def compute_acceleration(
        self,
        target_index: int,
        theta: float,
        pos: np.ndarray | None = None,
        eps: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute softened acceleration on one target particle."""
        if pos is None:
            pos = self.pos
        if eps is None:
            eps = self.eps
        acc = np.zeros(3, dtype=float)
        self._walk(self.root, target_index, theta, acc, pos, eps)
        return acc

    def _walk(
        self,
        node: OctreeNode,
        target_index: int,
        theta: float,
        acc: np.ndarray,
        pos: np.ndarray,
        eps: np.ndarray,
    ) -> None:
        if node.mass <= 0:
            return
        target_pos = pos[target_index]
        dr = node.com - target_pos
        dist = np.sqrt(np.sum(dr * dr))
        if node.is_leaf:
            for source in node.leaf_particles:
                if source == target_index:
                    continue
                r_vec = pos[source] - target_pos
                r2 = np.sum(r_vec * r_vec)
                if r2 == 0:
                    continue
                h = 0.5 * (eps[target_index] + eps[source])
                denom = (r2 + h * h) ** 1.5
                acc += G * self.mass[source] * r_vec / denom
            return

        if node.size / max(dist, 1e-30) < theta:
            h = eps[target_index]
            denom = (dist * dist + h * h) ** 1.5
            acc += G * node.mass * dr / denom
            return

        for child in node.children:
            if child is not None:
                self._walk(child, target_index, theta, acc, pos, eps)


def compute_forces_bh(
    pos: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
    theta: float = 0.5,
    tree: BarnesHutTree | None = None,
    target_indices: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute Barnes–Hut softened accelerations.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Particle positions.
    mass : ndarray, shape (N,)
        Particle masses.
    eps : ndarray, shape (N,)
        Softening lengths.
    theta : float
        Opening angle; smaller values are more accurate.
    tree : BarnesHutTree, optional
        Pre-built tree (built when ``None``).
    target_indices : ndarray, optional
        Subset of particles to evaluate (default: all).

    Returns
    -------
    acc : ndarray, shape (N, 3) or (len(target_indices), 3)
        Accelerations.
    """
    if tree is None:
        tree = BarnesHutTree(pos, mass, eps)
    n = len(mass)
    if target_indices is None:
        target_indices = np.arange(n)
    acc = np.zeros((len(target_indices), 3), dtype=float)
    for k, i in enumerate(target_indices):
        acc[k] = tree.compute_acceleration(int(i), theta, pos, eps)
    return acc
