"""N-body simulation driver."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ntropy.config import RunConfig
from ntropy.forces.bhtree import BarnesHutTree, compute_forces_bh
from ntropy.forces.brute import compute_forces_brute
from ntropy.integrators.leapfrog import leapfrog_step
from ntropy.parallel.pool import compute_forces_parallel
from ntropy.particles import ParticleState
from ntropy.softening import total_energy


@dataclass
class SimulationResult:
    """
    Output of a completed simulation run.

    Attributes
    ----------
    initial_state : ParticleState
        State before integration (COM-removed).
    final_state : ParticleState
        State after all steps.
    energies : list of float
        Total energy after each step (including initial).
    output_dir : Path or None
        Directory where snapshots were written, if any.
    """

    initial_state: ParticleState
    final_state: ParticleState
    energies: list[float] = field(default_factory=list)
    output_dir: Path | None = None


class Simulation:
    """
    Self-gravitating N-body simulation with configurable force backend.

    Parameters
    ----------
    config : RunConfig
        JSON-derived run configuration.
    state : ParticleState, optional
        Initial particle state.  Loaded from ``config`` when ``None``.

    Notes
    -----
    Parallel force evaluation uses mpi4py domain decomposition when
    ``config.parallel.enabled`` is true.  Launch with ``mpirun -n N`` for
    multi-rank execution.
    """

    def __init__(self, config: RunConfig, state: ParticleState | None = None):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.state = state if state is not None else ParticleState.from_config(config)
        self._tree: BarnesHutTree | None = None

    def _compute_accelerations(self, state: ParticleState) -> np.ndarray:
        """
        Evaluate gravitational accelerations for the current state.

        Parameters
        ----------
        state : ParticleState
            Current particle state.

        Returns
        -------
        acc : ndarray, shape (N, 3)
        """
        cfg = self.config
        use_parallel = cfg.parallel.enabled
        if use_parallel:
            return compute_forces_parallel(
                state.pos,
                state.mass,
                state.eps,
                method=cfg.force.method,
                theta=cfg.force.theta,
                n_workers=cfg.parallel.n_workers,
            )
        if cfg.force.method == "brute":
            return compute_forces_brute(state.pos, state.mass, state.eps)
        self._tree = BarnesHutTree(state.pos, state.mass, state.eps)
        return compute_forces_bh(
            state.pos,
            state.mass,
            state.eps,
            theta=cfg.force.theta,
            tree=self._tree,
        )

    def step(self, dt: float | None = None) -> None:
        """
        Advance one leapfrog timestep.

        Parameters
        ----------
        dt : float, optional
            Timestep override (defaults to ``config.integrator.dt``).
        """
        dt = self.config.integrator.dt if dt is None else dt
        acc = self._compute_accelerations(self.state)
        pos_new, vel_half = leapfrog_step(self.state.pos, self.state.vel, acc, dt)
        self.state.pos = pos_new
        self.state.vel = vel_half
        acc_new = self._compute_accelerations(self.state)
        self.state.vel = vel_half + 0.5 * dt * acc_new

    def run(self) -> SimulationResult:
        """
        Run the full simulation loop from the current configuration.

        Returns
        -------
        SimulationResult
            Initial/final states, energy history, and output path.
        """
        cfg = self.config
        state = self.state.copy()
        state.remove_center_of_mass()
        energies: list[float] = []
        energies.append(
            total_energy(state.pos, state.vel, state.mass, state.eps)
        )

        output_dir = cfg.resolve_path(cfg.output.dir)
        if cfg.output.every > 0 or cfg.output.write_final:
            output_dir.mkdir(parents=True, exist_ok=True)

        initial = state.copy()
        if cfg.output.every > 0:
            state.write_ascii(output_dir / "snapshot_0000.dat")

        for step in range(1, cfg.integrator.n_steps + 1):
            self.state = state
            self.step()
            state = self.state
            energies.append(
                total_energy(state.pos, state.vel, state.mass, state.eps)
            )
            if cfg.output.every > 0 and step % cfg.output.every == 0:
                state.write_ascii(output_dir / f"snapshot_{step:04d}.dat")

        if cfg.output.write_final:
            state.write_ascii(output_dir / "final.dat")

        return SimulationResult(
            initial_state=initial,
            final_state=state,
            energies=energies,
            output_dir=output_dir if cfg.output.write_final or cfg.output.every > 0 else None,
        )


def run_simulation(config: RunConfig, state: ParticleState | None = None) -> SimulationResult:
    """
    Convenience wrapper to construct and run a :class:`Simulation`.

    Parameters
    ----------
    config : RunConfig
        Run configuration.
    state : ParticleState, optional
        Initial state override.

    Returns
    -------
    SimulationResult
    """
    return Simulation(config, state=state).run()
