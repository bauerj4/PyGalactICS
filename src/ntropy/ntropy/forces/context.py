"""Force evaluation context with optional Barnes–Hut tree persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ntropy.config import ForceConfig
from ntropy.forces.bhtree import BarnesHutTree, compute_forces_bh
from ntropy.forces.bhtree_c import BarnesHutTreeC, compute_forces_bh_c, extension_available
from ntropy.forces.brute import compute_forces_brute
from ntropy.parallel.pool import compute_forces_parallel
from ntropy.parallel.mpi import MpiForceCache

if TYPE_CHECKING:
    from ntropy.particles import ParticleState


@dataclass
class ForceContext:
    """
    Cached force backend state across timesteps.

    Holds optional Barnes–Hut tree instances between force evaluations so the
    octree is not rebuilt from scratch every substep when
    ``force.rebuild_every > 1``.

    Attributes
    ----------
    config : ForceConfig
        Force method, opening angle, and rebuild cadence.
    parallel_enabled : bool
        Whether to use MPI/domain parallel force evaluation.
    n_workers : int
        Worker count when ``parallel_enabled`` is true.
    """

    config: ForceConfig
    parallel_enabled: bool = False
    n_workers: int = 1
    _bh_tree: BarnesHutTree | None = field(default=None, repr=False)
    _bh_c_tree: BarnesHutTreeC | None = field(default=None, repr=False)
    _gpu_bh_state: object | None = field(default=None, repr=False)
    _step_count: int = field(default=0, repr=False)
    _mpi_cache: MpiForceCache = field(default_factory=MpiForceCache, repr=False)

    def _should_rebuild_tree(self) -> bool:
        """True when the cached tree should be rebuilt this evaluation."""
        every = max(1, self.config.rebuild_every)
        return self._step_count % every == 0

    def accel_at_pos(
        self,
        state: ParticleState,
        pos: np.ndarray,
        *,
        target_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Evaluate gravitational accelerations at ``pos``.

        Parameters
        ----------
        state : ParticleState
            Source masses and softening (and tree topology for BH).
        pos : ndarray, shape (N, 3)
            Trial positions [kpc].
        target_indices : ndarray, shape (N_targets,), optional
            Subset of particles to evaluate; default all.

        Returns
        -------
        acc : ndarray, shape (N, 3) or (N_targets, 3)
            Accelerations [code units].
        """
        mass = state.mass
        eps = state.eps
        method = self.config.method
        theta = self.config.theta

        if self.parallel_enabled:
            acc = compute_forces_parallel(
                pos,
                mass,
                eps,
                method=method,
                theta=theta,
                n_workers=self.n_workers,
                bh_opts=self.config.bh_optimizations,
                cache=self._mpi_cache,
                rebuild=self._should_rebuild_tree(),
                mpi_local_trees=self.config.mpi_local_trees,
            )
            if target_indices is not None:
                return acc[target_indices]
            return acc

        if method == "brute":
            return compute_forces_brute(
                pos, mass, eps, target_indices=target_indices
            )

        if method == "gpu_bh":
            from ntropy.forces.gpu_bh import GpuBhState, compute_forces_gpu_bh

            if self._gpu_bh_state is None:
                self._gpu_bh_state = GpuBhState()
            return compute_forces_gpu_bh(
                pos,
                mass,
                eps,
                theta=theta,
                target_indices=target_indices,
                state=self._gpu_bh_state,
                bh_opts=self.config.bh_optimizations,
            )

        if method == "gpu_direct":
            from ntropy.forces.gpu_direct import compute_forces_gpu

            return compute_forces_gpu(
                pos, mass, eps, target_indices=target_indices
            )

        if method == "bh_c":
            if not extension_available():
                raise ImportError(
                    "force.method='bh_c' requires the C extension; "
                    "pip install -e src/ntropy"
                )
            bh_opts = self.config.bh_optimizations
            if self._should_rebuild_tree() or self._bh_c_tree is None:
                self._bh_c_tree = BarnesHutTreeC.build(
                    pos, mass, eps, bh_opts=bh_opts
                )
            return compute_forces_bh_c(
                pos,
                mass,
                eps,
                theta=theta,
                tree=self._bh_c_tree,
                target_indices=target_indices,
                bh_opts=bh_opts,
            )

        if method != "bh":
            raise ValueError(
                f"Unknown force.method={method!r}; "
                "expected one of brute, bh, bh_c, gpu_bh, gpu_direct"
            )

        if self._should_rebuild_tree() or self._bh_tree is None:
            self._bh_tree = BarnesHutTree(pos, mass, eps)
        return compute_forces_bh(
            pos,
            mass,
            eps,
            theta=theta,
            tree=self._bh_tree,
            target_indices=target_indices,
        )

    def after_force_eval(self) -> None:
        """Advance internal counter after one force evaluation."""
        self._step_count += 1

    def reset(self) -> None:
        """Discard cached trees (e.g. after a large position change)."""
        self._bh_tree = None
        self._bh_c_tree = None
        if self._gpu_bh_state is not None:
            detach = getattr(self._gpu_bh_state, "detach", None)
            if callable(detach):
                detach()
        self._gpu_bh_state = None
        self._mpi_cache.clear()
        self._step_count = 0
