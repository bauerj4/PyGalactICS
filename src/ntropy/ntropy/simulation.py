"""N-body simulation driver."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ntropy.analysis.tiered_diagnostics import (
    TieredDiagnosticsLog,
    write_diagnostics_log,
    write_particle_bin_dump,
)
from ntropy.config import RunConfig, format_run_config
from ntropy.forces.context import ForceContext
from ntropy.integrators.euler import euler_step
from ntropy.integrators.leapfrog import leapfrog1_step, leapfrog_step
from ntropy.integrators.rk import rk2_step, rk3_step, rk4_step
from ntropy.integrators.tiered import run_tiered_leapfrog
from ntropy.integrators.timestep import TimestepConfig
from ntropy.parallel.mpi import mpi_rank0
from ntropy.particle_types import TypeRegistry
from ntropy.particles import ParticleState
from ntropy.softening import total_energy
from ntropy.units import code_time_to_gyr, code_time_to_myr


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
    diagnostics : TieredDiagnosticsLog or None
        Per-substep bin and activity history from ntropy tiered runs.
    """

    initial_state: ParticleState
    final_state: ParticleState
    energies: list[float] = field(default_factory=list)
    output_dir: Path | None = None
    diagnostics: TieredDiagnosticsLog | None = None


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
        self._force_ctx = ForceContext(
            config=config.force,
            parallel_enabled=config.parallel.enabled,
            n_workers=config.parallel.n_workers,
        )
        self._accel_cache: np.ndarray | None = None

    def _accel_at_pos(
        self,
        pos: np.ndarray,
        *,
        target_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        use_subset = (
            target_indices is not None
            and self.config.force.active_subset
            and len(target_indices) < self.state.n
        )
        if not use_subset:
            acc = self._force_ctx.accel_at_pos(self.state, pos)
            self._accel_cache = acc
        else:
            if self._accel_cache is None or self._accel_cache.shape[0] != self.state.n:
                self._accel_cache = np.zeros((self.state.n, 3), dtype=float)
            partial = self._force_ctx.accel_at_pos(
                self.state, pos, target_indices=target_indices
            )
            self._accel_cache[target_indices] = partial
            acc = self._accel_cache
        self._force_ctx.after_force_eval()
        return acc

    def _compute_accelerations(self, state: ParticleState) -> np.ndarray:
        self.state = state
        return self._accel_at_pos(state.pos)

    def step(self, dt: float | None = None) -> None:
        """Advance one timestep with the configured integrator."""
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

    def _run_tiered(
        self,
        state: ParticleState,
        *,
        output_dir: Path | None = None,
        show_progress: bool = False,
        progress_desc: str | None = None,
    ) -> tuple[ParticleState, list[float], TieredDiagnosticsLog]:
        """
        Run ntropy tiered leapfrog on ``state``.

        Returns
        -------
        state : ParticleState
            Evolved state with updated ``timestep_bin``.
        energies : list of float
            Energy history.
        diagnostics : TieredDiagnosticsLog
            Per-substep bin and activity records.
        """
        registry = self.config.particle_types or TypeRegistry.default_galaxy()
        integ = self.config.integrator
        out = self.config.output
        ts_config = integ.timestep
        if integ.dt_base is not None:
            ts_config = TimestepConfig(
                eta=ts_config.eta,
                dt_base=integ.dt_base,
                max_bin=ts_config.max_bin,
                update_every=ts_config.update_every,
                accel_floor=ts_config.accel_floor,
            )
        end_gyr = integ.end_time_gyr if integ.end_time_gyr is not None else 1.0
        last_acc: np.ndarray | None = None
        particles_dir = output_dir / "particles" if output_dir else None
        dump_every = out.particle_dump_every or out.diagnostics_every
        progress_jsonl = None
        diagnostics_jsonl = None
        io_rank0 = mpi_rank0()
        if output_dir is not None and out.diagnostics_every > 0:
            progress_jsonl = str(output_dir / "diagnostics.progress.jsonl")
            if io_rank0:
                (output_dir / "diagnostics.progress.jsonl").write_text("")
                diagnostics_jsonl = output_dir / "diagnostics.jsonl"
                diagnostics_jsonl.write_text("")

        def _maybe_dump(step: int, snap: ParticleState, acc: np.ndarray) -> None:
            if not io_rank0:
                return
            if particles_dir is None or not out.write_particle_bins:
                return
            if dump_every <= 0 or step % dump_every != 0:
                return
            write_particle_bin_dump(
                particles_dir / f"step_{step:06d}.npz",
                snap,
                acc=acc,
            )

        def accel_fn(
            pos: np.ndarray,
            active_idx: np.ndarray | None = None,
        ) -> np.ndarray:
            nonlocal last_acc
            self.state.pos = pos
            last_acc = self._accel_at_pos(pos, target_indices=active_idx)
            return last_acc

        state, energies, diag_log = run_tiered_leapfrog(
            state,
            registry,
            accel_fn,
            ts_config=ts_config,
            end_time_gyr=end_gyr,
            order=integ.order,
            energy_every=max(0, out.every),
            diagnostics_every=out.diagnostics_every,
            diagnostics_jsonl=diagnostics_jsonl if io_rank0 else None,
            on_record=_maybe_dump if out.write_particle_bins else None,
            particle_dump_every=dump_every if out.write_particle_bins else None,
            show_progress=show_progress,
            progress_desc=progress_desc,
            progress_style="ntropy",
            progress_jsonl=progress_jsonl,
        )

        if output_dir is not None and out.diagnostics_every > 0 and io_rank0:
            write_diagnostics_log(diag_log, output_dir)

        return state, energies, diag_log

    def run(
        self,
        *,
        show_progress: bool = False,
        progress_desc: str | None = None,
        print_config: bool = False,
    ) -> SimulationResult:
        """Run the full simulation loop from the current configuration."""
        cfg = self.config
        if print_config or show_progress:
            print(format_run_config(cfg, label=progress_desc), file=sys.stderr, flush=True)

        state = self.state.copy()
        state.remove_center_of_mass()
        energies: list[float] = []
        energies.append(
            total_energy(state.pos, state.vel, state.mass, state.eps)
        )

        output_dir = cfg.resolve_path(cfg.output.dir)
        needs_output_dir = (
            cfg.output.every > 0
            or cfg.output.write_final
            or cfg.output.diagnostics_every > 0
        )
        if needs_output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        initial = state.copy()
        diag_log: TieredDiagnosticsLog | None = None
        if cfg.output.every > 0 and mpi_rank0():
            state.write_ascii(output_dir / "snapshot_0000.dat")

        if cfg.integrator.type == "tiered_leapfrog":
            state, tier_energies, diag_log = self._run_tiered(
                state,
                output_dir=output_dir if needs_output_dir else None,
                show_progress=show_progress,
                progress_desc=progress_desc,
            )
            energies = tier_energies if tier_energies else energies
            if cfg.output.write_final and mpi_rank0():
                state.write_ascii(output_dir / "final.dat")
            return SimulationResult(
                initial_state=initial,
                final_state=state,
                energies=energies,
                output_dir=output_dir if needs_output_dir else None,
                diagnostics=diag_log,
            )

        step_iter: range | object = range(1, cfg.integrator.n_steps + 1)
        if show_progress:
            try:
                from tqdm.auto import tqdm
            except ImportError as exc:
                raise ImportError(
                    "show_progress=True requires tqdm. Install with: pip install tqdm"
                ) from exc
            step_iter = tqdm(
                step_iter,
                desc=progress_desc or "ntropy simulation",
                unit="step",
                total=cfg.integrator.n_steps,
                mininterval=0.5,
            )

        dt = cfg.integrator.dt
        for step in step_iter:
            self.state = state
            self.step()
            state = self.state
            energy = total_energy(state.pos, state.vel, state.mass, state.eps)
            energies.append(energy)
            if show_progress and hasattr(step_iter, "set_postfix"):
                e0 = max(abs(energies[0]), 1e-30)
                t_code = step * dt
                step_iter.set_postfix(
                    t_Gyr=f"{code_time_to_gyr(t_code):.3f}",
                    t_Myr=f"{code_time_to_myr(t_code):.0f}",
                    dE=f"{abs(energy - energies[0]) / e0:.2e}",
                    refresh=False,
                )
            if cfg.output.every > 0 and step % cfg.output.every == 0 and mpi_rank0():
                state.write_ascii(output_dir / f"snapshot_{step:04d}.dat")

        if cfg.output.write_final and mpi_rank0():
            state.write_ascii(output_dir / "final.dat")

        return SimulationResult(
            initial_state=initial,
            final_state=state,
            energies=energies,
            output_dir=output_dir if cfg.output.write_final or cfg.output.every > 0 else None,
        )


def run_simulation(config: RunConfig, state: ParticleState | None = None) -> SimulationResult:
    """Convenience wrapper to construct and run a :class:`Simulation`."""
    return Simulation(config, state=state).run()
