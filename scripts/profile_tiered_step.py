#!/usr/bin/env python3
"""Profile one tiered leapfrog substep using existing campaign ICs."""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import time
from io import StringIO
from pathlib import Path

import numpy as np

from ntropy.analysis.tiered_diagnostics import collect_step_diagnostics, write_particle_bin_dump
from ntropy.benchmark.force_breakdown import time_bh_c_components
from ntropy.config import load_config
from ntropy.forces.context import ForceContext
from ntropy.integrators.leapfrog import leapfrog_step
from ntropy.integrators.timestep import active_mask_for_step, update_timestep_bins
from ntropy.particle_types import TypeRegistry
from ntropy.particles import ParticleState
from ntropy.softening import kinetic_energy, total_energy


def load_ic(path: Path) -> ParticleState:
    with np.load(path, allow_pickle=True) as data:
        return ParticleState.from_arrays(
            data["pos"],
            data["vel"],
            data["mass"],
            data["eps"],
            type_id=data["type_id"] if "type_id" in data else None,
            timestep_bin=data["timestep_bin"] if "timestep_bin" in data else None,
            tags=data["tags"] if "tags" in data else None,
        )


def profile_substeps(
    work_dir: Path,
    *,
    n_substeps: int,
    mpi: bool,
    out: Path | None,
) -> None:
    cfg = load_config(work_dir / "ntropy_config.json")
    state = load_ic(work_dir / "ic_state.npz")
    registry = TypeRegistry.default_galaxy()
    ts = cfg.integrator.timestep
    if cfg.integrator.dt_base is not None:
        from ntropy.integrators.timestep import TimestepConfig

        ts = TimestepConfig(
            eta=ts.eta,
            dt_base=cfg.integrator.dt_base,
            max_bin=ts.max_bin,
            update_every=ts.update_every,
            accel_floor=ts.accel_floor,
        )

    ctx = ForceContext(
        config=cfg.force,
        parallel_enabled=mpi,
        n_workers=cfg.parallel.n_workers,
    )
    pos = state.pos.copy()
    vel = state.vel.copy()
    last_acc: np.ndarray | None = None

    def accel_fn(p: np.ndarray) -> np.ndarray:
        nonlocal last_acc
        state.pos = p
        last_acc = ctx.accel_at_pos(state, p)
        ctx.after_force_eval()
        return last_acc

    acc = accel_fn(pos)
    bins = update_timestep_bins(
        acc, state.eps, state.type_id, registry, state.timestep_bin, ts
    )
    e0 = total_energy(pos, vel, state.mass, state.eps)

    def one_substep(step: int) -> None:
        nonlocal pos, vel, acc, bins, last_acc
        active_idx = np.nonzero(active_mask_for_step(step, bins))[0]
        acc = accel_fn(pos)
        pos_half, vel_half = leapfrog_step(
            pos[active_idx], vel[active_idx], acc[active_idx], ts.dt_base
        )
        pos[active_idx] = pos_half
        acc = accel_fn(pos)
        vel[active_idx] = vel_half + 0.5 * ts.dt_base * acc[active_idx]
        bins = update_timestep_bins(
            acc, state.eps, state.type_id, registry, bins, ts
        )
        energy = total_energy(pos, vel, state.mass, state.eps)
        record = collect_step_diagnostics(
            step=step,
            bins=bins,
            type_id=state.type_id,
            registry=registry,
            energy=energy,
            e0=e0,
            n_active=int(active_idx.size),
            dt_base=ts.dt_base,
            max_bin=ts.max_bin,
            acc=acc,
        )
        if step % cfg.output.particle_dump_every == 0:
            snap = ParticleState.from_arrays(
                pos,
                vel,
                state.mass,
                state.eps,
                type_id=state.type_id,
                timestep_bin=bins.copy(),
                tags=state.tags,
            )
            dump_path = work_dir / "evolution" / "profile_step.npz"
            write_particle_bin_dump(dump_path, snap, acc=acc)

    # Component micro-benchmarks (serial C BH on full state)
    bh = time_bh_c_components(pos, state.mass, state.eps, theta=cfg.force.theta, n_repeat=5)
    ke_t0 = time.perf_counter()
    for _ in range(200):
        kinetic_energy(vel, state.mass)
    ke_ms = (time.perf_counter() - ke_t0) / 200 * 1000
    te_t0 = time.perf_counter()
    for _ in range(200):
        total_energy(pos, vel, state.mass, state.eps)
    te_ms = (time.perf_counter() - te_t0) / 200 * 1000

    t0 = time.perf_counter()
    for step in range(1, n_substeps + 1):
        one_substep(step)
    wall_ms = (time.perf_counter() - t0) / n_substeps * 1000

    print(f"work_dir={work_dir}")
    print(f"N={state.n:,} mpi={mpi} substeps_profiled={n_substeps}")
    print(f"dt_base={ts.dt_base} rebuild_every={cfg.force.rebuild_every}")
    print(f"serial bh_c build={bh.ms_build:.1f}ms walk={bh.ms_walk:.1f}ms total={bh.ms_total:.1f}ms")
    print(f"KE-only {ke_ms:.3f}ms | total_energy (N>16k) {te_ms:.3f}ms per call")
    print(f"full substep wall {wall_ms:.1f}ms (diag+dump policy from config)")

    prof_path = out or (work_dir / "profile_substep.prof")
    cProfile.runctx(
        "one_substep(n_substeps + 1)",
        globals(),
        locals(),
        str(prof_path),
    )
    stats = pstats.Stats(str(prof_path))
    stats.strip_dirs().sort_stats("cumulative")
    buf = StringIO()
    stats.stream = buf
    stats.print_stats(25)
    print("\n--- cProfile (cumulative top 25) ---")
    print(buf.getvalue())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("work_dir", type=Path)
    p.add_argument("--substeps", type=int, default=3)
    p.add_argument("--mpi", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    profile_substeps(args.work_dir, n_substeps=args.substeps, mpi=args.mpi, out=args.out)


if __name__ == "__main__":
    main()
