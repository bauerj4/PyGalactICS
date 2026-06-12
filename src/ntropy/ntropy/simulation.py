"""N-body simulation driver."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ntropy.config import RunConfig
from ntropy.forces.bhtree import BarnesHutTree, compute_forces_bh
from ntropy.forces.brute import compute_forces_brute
from ntropy.integrators.euler import euler_step
from ntropy.integrators.leapfrog import leapfrog1_step, leapfrog_step
from ntropy.integrators.rk import rk2_step, rk3_step, rk4_step
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

    def _accel_at_pos(self, pos: np.ndarray) -> np.ndarray:
        """
        Evaluate gravitational accelerations at an arbitrary position array.

        Parameters
        ----------
        pos : ndarray, shape (N, 3)
            Trial particle positions (masses and softening from ``self.state``).

        Returns
        -------
        acc : ndarray, shape (N, 3)
        """
        cfg = self.config
        mass = self.state.mass
        eps = self.state.eps
        if cfg.parallel.enabled:
            return compute_forces_parallel(
                pos,
                mass,
                eps,
                method=cfg.force.method,
                theta=cfg.force.theta,
                n_workers=cfg.parallel.n_workers,
            )
        if cfg.force.method == "brute":
            return compute_forces_brute(pos, mass, eps)
        tree = BarnesHutTree(pos, mass, eps)
        return compute_forces_bh(
            pos,
            mass,
            eps,
            theta=cfg.force.theta,
            tree=tree,
        )

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
        return self._accel_at_pos(state.pos)

    def step(self, dt: float | None = None) -> None:
        """
        Advance one timestep with the configured integrator.

        Parameters
        ----------
        dt : float, optional
            Timestep override (defaults to ``config.integrator.dt``).
        """
        dt = self.config.integrator.dt if dt is None else dt
        integ = self.config.integrator
        pos, vel = self.state.pos, self.state.vel

        if integ.type == "euler":
            acc = self._accel_at_pos(pos)
            pos_new, vel_new = euler_step(pos, vel, acc, dt)
        elif integ.type == "rk2":
            pos_new, vel_new = rk2_step(pos, vel, self._accel_at_pos, dt)
        elif integ.type == "rk3":
            pos_new, vel_new = rk3_step(pos, vel, self._accel_at_pos, dt)
        elif integ.type == "rk4":
            pos_new, vel_new = rk4_step(pos, vel, self._accel_at_pos, dt)
        elif integ.type == "leapfrog":
            acc = self._accel_at_pos(pos)
            if integ.order == 1:
                pos_new, vel_new = leapfrog1_step(pos, vel, acc, dt)
            else:
                pos_new, vel_half = leapfrog_step(pos, vel, acc, dt)
                acc_new = self._accel_at_pos(pos_new)
                pos_new, vel_new = pos_new, vel_half + 0.5 * dt * acc_new
        else:
            raise ValueError(f"Unknown integrator type {integ.type!r}")

        self.state.pos = pos_new
        self.state.vel = vel_new

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
