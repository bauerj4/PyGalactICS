"""GADGET-style tiered leapfrog with dynamic per-particle timestep bins."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

import numpy as np

from ntropy.analysis.tiered_diagnostics import (
    StepDiagnostics,
    TieredDiagnosticsLog,
    collect_step_diagnostics,
)
from ntropy.integrators.leapfrog import leapfrog1_step, leapfrog_step
from ntropy.integrators.tiered_progress import NtropyProgressReporter
from ntropy.integrators.timestep import (
    TimestepConfig,
    active_mask_for_step,
    update_timestep_bins,
)
from ntropy.particle_types import TypeRegistry
from ntropy.particles import ParticleState
from ntropy.softening import total_energy
from ntropy.units import code_time_to_gyr


def run_tiered_leapfrog(
    state: ParticleState,
    registry: TypeRegistry,
    accel_fn: Callable[..., np.ndarray],
    *,
    ts_config: TimestepConfig,
    end_time_gyr: float,
    order: int = 2,
    energy_every: int = 0,
    diagnostics_every: int = 1,
    record_accel: bool = True,
    diagnostics_jsonl: str | Path | None = None,
    on_record: Callable[[int, ParticleState, np.ndarray], None] | None = None,
    particle_dump_every: int | None = None,
    show_progress: bool = False,
    progress_desc: str | None = None,
    progress_style: Literal["ntropy", "tqdm", "both"] = "ntropy",
    progress_print_every: int = 1,
    progress_jsonl: str | None = None,
) -> tuple[ParticleState, list[float], TieredDiagnosticsLog]:
    """
    Integrate with GADGET-style dynamic, quantized per-particle timesteps.

    Each particle carries an integer bin ``b``; its timestep is
    ``ts_config.dt_base * 2**b``.  Bins are recomputed from the local
    acceleration and softening (see :func:`~ntropy.integrators.timestep.update_timestep_bins`),
    clamped to per-type allowed ranges, and damped to change by at most one
    bin per update.

    The integrator advances a global fine substep of size ``dt_base``.
    Particle ``i`` is active on substep ``k`` when ``k % 2**b_i == 0``.

    Parameters
    ----------
    state : ParticleState
        Particle state (updated in place).  Requires ``type_id``; optional
        ``timestep_bin`` is initialized from the first force evaluation.
    registry : TypeRegistry
        Type metadata (softening defaults and bin limits per type).
    accel_fn : callable
        ``accel_fn(pos)`` or ``accel_fn(pos, active_idx)`` returning accelerations
        shaped ``(N, 3)`` [code units].  When ``active_idx`` is provided, only
        those particles need fresh force values (stale values elsewhere are OK).
    ts_config : TimestepConfig
        Base timestep, ``eta``, and bin hierarchy settings.
    end_time_gyr : float
        Simulation end time [Gyr].
    order : int
        Leapfrog variant: ``1`` (symplectic Euler) or ``2`` (velocity Verlet).
    energy_every : int
        Record total energy every this many fine substeps; ``0`` records
        initial and final energies only.
    diagnostics_every : int
        Record tiered bin/activity diagnostics every this many fine
        substeps.  ``0`` disables; ``1`` records every substep (default).
    record_accel : bool
        When true, include mean ``|a|`` in diagnostic records.
    on_record : callable, optional
        ``on_record(step, state_snapshot, acc)`` for per-particle ``.npz`` dumps.
    particle_dump_every : int, optional
        When set with ``on_record``, invoke the callback only on substeps where
        ``step % particle_dump_every == 0``.  ``None`` keeps legacy behaviour
        (callback on every diagnostic substep).

    Returns
    -------
    state : ParticleState
        Updated state including ``timestep_bin`` on each particle.
    energies : list of float
        Total energy history [code units].
    diagnostics : TieredDiagnosticsLog
        Per-substep bin histograms, active counts, and energy drift.

    Raises
    ------
    ValueError
        If ``state.type_id`` is missing.

    Notes
    -----
    Unlike a fixed per-type timestep, bins here respond to the dynamical
    time :math:`\\sqrt{\\varepsilon/|a|}` as in GADGET-2.  Component types only
    bound the allowed bin range (e.g. disk cannot use bins coarser than its max).
    """
    if state.type_id is None:
        raise ValueError("tiered integration requires ParticleState.type_id")

    t_end = end_time_gyr / code_time_to_gyr(1.0)
    n_steps = max(1, int(round(t_end / ts_config.dt_base)))

    ntropy_reporter: NtropyProgressReporter | None = None
    use_ntropy = show_progress and progress_style in ("ntropy", "both")
    use_tqdm = show_progress and progress_style in ("tqdm", "both")
    if use_ntropy:
        ntropy_reporter = NtropyProgressReporter(
            n_steps_total=n_steps,
            n_particles=state.n,
            end_time_gyr=end_time_gyr,
            label=progress_desc,
            print_every=progress_print_every,
            progress_jsonl=progress_jsonl,
        )
        ntropy_reporter.banner(dt_base=ts_config.dt_base)
        ntropy_reporter.note("initial force evaluation…")

    pos = state.pos
    vel = state.vel
    energies: list[float] = [total_energy(pos, vel, state.mass, state.eps)]
    e0 = energies[0]
    diag_log = TieredDiagnosticsLog(
        dt_base=ts_config.dt_base,
        max_bin=ts_config.max_bin,
        e0=e0,
        jsonl_path=Path(diagnostics_jsonl) if diagnostics_jsonl is not None else None,
    )

    acc = accel_fn(pos)
    bins = update_timestep_bins(
        acc,
        state.eps,
        state.type_id,
        registry,
        state.timestep_bin,
        ts_config,
    )
    state.timestep_bin = bins

    def _snapshot_state() -> ParticleState:
        return ParticleState.from_arrays(
            pos, vel, state.mass, state.eps,
            type_id=state.type_id,
            timestep_bin=bins.copy(),
            tags=state.tags,
        )

    def _maybe_record(step_idx: int, n_active: int, current_acc: np.ndarray) -> None:
        if diagnostics_every <= 0:
            return
        if step_idx % diagnostics_every != 0 and step_idx != n_steps:
            return
        energy = total_energy(pos, vel, state.mass, state.eps)
        record = collect_step_diagnostics(
            step=step_idx,
            bins=bins,
            type_id=state.type_id,
            registry=registry,
            energy=energy,
            e0=e0,
            n_active=n_active,
            dt_base=ts_config.dt_base,
            max_bin=ts_config.max_bin,
            acc=current_acc if record_accel else None,
        )
        diag_log.record(record)
        if ntropy_reporter is not None:
            ntropy_reporter.update(record)
        if on_record is not None:
            dump_step = (
                particle_dump_every is None
                or particle_dump_every <= 0
                or step_idx % particle_dump_every == 0
            )
            if dump_step:
                on_record(step_idx, _snapshot_state(), current_acc)

    _maybe_record(0, n_active=state.n, current_acc=acc)
    if ntropy_reporter is not None:
        ntropy_reporter.note("integrating…")

    step_range: range | object = range(1, n_steps + 1)
    if use_tqdm:
        try:
            from tqdm.auto import tqdm
        except ImportError as exc:
            raise ImportError(
                "show_progress=True requires tqdm. Install with: pip install tqdm"
            ) from exc
        step_range = tqdm(
            step_range,
            desc=progress_desc or "tiered leapfrog",
            unit="substep",
            total=n_steps,
            mininterval=0.5,
        )

    for step in step_range:
        active = active_mask_for_step(step, bins)
        active_idx = np.nonzero(active)[0]
        n_active = int(active_idx.size)
        if active_idx.size == 0:
            continue

        acc = accel_fn(pos, active_idx)

        if order == 1:
            pos_a, vel_a = leapfrog1_step(
                pos[active_idx], vel[active_idx], acc[active_idx], ts_config.dt_base
            )
            pos[active_idx] = pos_a
            vel[active_idx] = vel_a
        else:
            pos_half, vel_half = leapfrog_step(
                pos[active_idx], vel[active_idx], acc[active_idx], ts_config.dt_base
            )
            pos[active_idx] = pos_half
            acc = accel_fn(pos, active_idx)
            vel[active_idx] = vel_half + 0.5 * ts_config.dt_base * acc[active_idx]

        if step % ts_config.update_every == 0:
            bins = update_timestep_bins(
                acc,
                state.eps,
                state.type_id,
                registry,
                bins,
                ts_config,
            )
            state.timestep_bin = bins

        if energy_every > 0 and step % energy_every == 0:
            energies.append(total_energy(pos, vel, state.mass, state.eps))

        _maybe_record(step, n_active=n_active, current_acc=acc)

        if use_tqdm and hasattr(step_range, "set_postfix"):
            t_code = step * ts_config.dt_base
            energy = total_energy(pos, vel, state.mass, state.eps)
            e0_abs = max(abs(e0), 1e-30)
            step_range.set_postfix(
                t_Gyr=f"{code_time_to_gyr(t_code):.4f}",
                dE=f"{abs(energy - e0) / e0_abs:.2e}",
                active=f"{n_active}/{state.n}",
                refresh=False,
            )

    state.pos = pos
    state.vel = vel
    energies.append(total_energy(pos, vel, state.mass, state.eps))
    if ntropy_reporter is not None:
        ntropy_reporter.finish()
    return state, energies, diag_log
