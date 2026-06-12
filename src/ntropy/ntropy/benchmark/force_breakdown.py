"""Timing breakdown helpers for force-backend scaling investigations."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ntropy.forces.bhtree import BarnesHutTree, compute_forces_bh
from ntropy.forces.bhtree_c import BarnesHutTreeC, compute_forces_bh_c, extension_available
from ntropy.forces.brute import compute_forces_brute
from ntropy.parallel.domains import domain_slices, sort_by_peano


@dataclass
class BruteBreakdown:
    n_particles: int
    n_targets: int
    ms_per_force: float


@dataclass
class BhBreakdown:
    n_particles: int
    ms_build: float
    ms_walk: float
    ms_total: float
    build_fraction: float


@dataclass
class BhCBreakdown:
    """Timing split for the C Barnes–Hut extension (same fields as BhBreakdown)."""

    n_particles: int
    ms_build: float
    ms_walk: float
    ms_total: float
    build_fraction: float


def time_brute(
    pos: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
    *,
    target_indices: np.ndarray | None = None,
    n_warmup: int = 3,
    n_repeat: int = 15,
) -> float:
    """Seconds per brute-force evaluation."""
    for _ in range(n_warmup):
        compute_forces_brute(pos, mass, eps, target_indices=target_indices)
    start = time.perf_counter()
    for _ in range(n_repeat):
        compute_forces_brute(pos, mass, eps, target_indices=target_indices)
    return (time.perf_counter() - start) / n_repeat


def time_bh_components(
    pos: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
    *,
    theta: float = 0.5,
    target_indices: np.ndarray | None = None,
    n_warmup: int = 2,
    n_repeat: int = 10,
) -> BhBreakdown:
    """
    Split Barnes–Hut cost into tree build vs tree walk.

    The serial notebook benchmark times ``build + walk`` each call because
    it does not reuse a pre-built tree.
    """
    n = len(mass)
    targets = (
        np.arange(n, dtype=int)
        if target_indices is None
        else np.asarray(target_indices, dtype=int)
    )

    tree = BarnesHutTree(pos, mass, eps)
    for _ in range(n_warmup):
        compute_forces_bh(
            pos, mass, eps, theta=theta, tree=tree, target_indices=targets
        )

    build_times: list[float] = []
    walk_times: list[float] = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        local_tree = BarnesHutTree(pos, mass, eps)
        build_times.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        compute_forces_bh(
            pos, mass, eps, theta=theta, tree=local_tree, target_indices=targets
        )
        walk_times.append(time.perf_counter() - t1)

    ms_build = float(np.mean(build_times) * 1e3)
    ms_walk = float(np.mean(walk_times) * 1e3)
    ms_total = ms_build + ms_walk
    frac = ms_build / ms_total if ms_total > 0 else 0.0
    return BhBreakdown(
        n_particles=n,
        ms_build=ms_build,
        ms_walk=ms_walk,
        ms_total=ms_total,
        build_fraction=frac,
    )


def time_bh_c_components(
    pos: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
    *,
    theta: float = 0.5,
    target_indices: np.ndarray | None = None,
    n_warmup: int = 2,
    n_repeat: int = 10,
) -> BhCBreakdown:
    """
    Split C Barnes–Hut cost into tree build vs tree walk.

    Mirrors :func:`time_bh_components` but uses :class:`BarnesHutTreeC`.
    Raises :exc:`ImportError` when the ``_bh_c`` extension is missing.
    """
    if not extension_available():
        raise ImportError("C Barnes–Hut extension not built")

    n = len(mass)
    targets = (
        np.arange(n, dtype=int)
        if target_indices is None
        else np.asarray(target_indices, dtype=int)
    )

    tree = BarnesHutTreeC.build(pos, mass, eps)
    for _ in range(n_warmup):
        compute_forces_bh_c(
            pos, mass, eps, theta=theta, tree=tree, target_indices=targets
        )

    build_times: list[float] = []
    walk_times: list[float] = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        local_tree = BarnesHutTreeC.build(pos, mass, eps)
        build_times.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        compute_forces_bh_c(
            pos, mass, eps, theta=theta, tree=local_tree, target_indices=targets
        )
        walk_times.append(time.perf_counter() - t1)

    ms_build = float(np.mean(build_times) * 1e3)
    ms_walk = float(np.mean(walk_times) * 1e3)
    ms_total = ms_build + ms_walk
    frac = ms_build / ms_total if ms_total > 0 else 0.0
    return BhCBreakdown(
        n_particles=n,
        ms_build=ms_build,
        ms_walk=ms_walk,
        ms_total=ms_total,
        build_fraction=frac,
    )


def mpi_domain_target_count(n_particles: int, n_ranks: int) -> int:
    """Typical local target count for Morton-sorted domain decomposition."""
    slices = domain_slices(n_particles, n_ranks)
    return slices[0].stop - slices[0].start


def estimate_mpi_overhead_ms(
    pos: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
    *,
    n_ranks: int,
    n_repeat: int = 10,
) -> float:
    """
    Rough cost of Morton sort + index bookkeeping per force call.

    Does not include actual force computation or MPI collectives.
    """
    n = len(mass)
    order = sort_by_peano(pos)
    slices = domain_slices(n, n_ranks)
    local_targets = order[slices[0]]

    start = time.perf_counter()
    for _ in range(n_repeat):
        order = sort_by_peano(pos)
        slices = domain_slices(n, n_ranks)
        _ = order[slices[0]]
        acc = np.zeros((len(local_targets), 3))
        full_acc = np.zeros((n, 3))
        for indices, piece in zip([local_targets], [acc]):
            full_acc[indices] = piece
    return (time.perf_counter() - start) / n_repeat * 1e3
