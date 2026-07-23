#!/usr/bin/env python3
"""Compare legacy vs optimized bh_c force timing on campaign ICs."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ntropy.config import BhOptimizationsConfig, load_config
from ntropy.forces.bhtree_c import BarnesHutTreeC, extension_available
from ntropy.forces.context import ForceContext
from ntropy.particles import ParticleState


def load_ic(path: Path) -> ParticleState:
    with np.load(path, allow_pickle=True) as data:
        return ParticleState.from_arrays(
            data["pos"],
            data["vel"],
            data["mass"],
            data["eps"],
            type_id=data["type_id"] if "type_id" in data else None,
        )


def bench_force(
    state: ParticleState,
    *,
    preset: str,
    n_repeat: int,
    mpi: bool,
    n_workers: int,
) -> float:
    from ntropy.config import ForceConfig

    bh = BhOptimizationsConfig(preset=preset)  # type: ignore[arg-type]
    cfg = ForceConfig(method="bh_c", theta=0.5, bh_optimizations=bh)
    ctx = ForceContext(config=cfg, parallel_enabled=mpi, n_workers=n_workers)
    pos = state.pos
    for _ in range(1):
        ctx.accel_at_pos(state, pos)
        ctx.after_force_eval()
    ctx.reset()
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        ctx.accel_at_pos(state, pos)
        ctx.after_force_eval()
    return (time.perf_counter() - t0) / n_repeat * 1000


def bench_substep(
    state: ParticleState,
    *,
    preset: str,
    n_substeps: int,
    mpi: bool,
    n_workers: int,
    dt_base: float,
) -> float:
    from ntropy.config import ForceConfig
    from ntropy.integrators.leapfrog import leapfrog_step
    from ntropy.integrators.timestep import TimestepConfig, active_mask_for_step, update_timestep_bins
    from ntropy.particle_types import TypeRegistry

    bh = BhOptimizationsConfig(preset=preset)  # type: ignore[arg-type]
    cfg = ForceConfig(method="bh_c", theta=0.5, bh_optimizations=bh)
    ctx = ForceContext(config=cfg, parallel_enabled=mpi, n_workers=n_workers)
    registry = TypeRegistry.default_galaxy()
    ts = TimestepConfig(dt_base=dt_base)
    pos = state.pos.copy()
    vel = state.vel.copy()
    bins = np.zeros(state.n, dtype=np.int32)

    def accel_fn(p: np.ndarray) -> np.ndarray:
        state.pos = p
        acc = ctx.accel_at_pos(state, p)
        ctx.after_force_eval()
        return acc

    acc = accel_fn(pos)
    bins = update_timestep_bins(acc, state.eps, state.type_id, registry, bins, ts)

    t0 = time.perf_counter()
    for step in range(1, n_substeps + 1):
        active_idx = np.nonzero(active_mask_for_step(step, bins))[0]
        acc = accel_fn(pos)
        pos_half, vel_half = leapfrog_step(pos[active_idx], vel[active_idx], acc[active_idx], dt_base)
        pos[active_idx] = pos_half
        acc = accel_fn(pos)
        vel[active_idx] = vel_half + 0.5 * dt_base * acc[active_idx]
    return (time.perf_counter() - t0) / n_substeps * 1000


def main() -> None:
    if not extension_available():
        raise SystemExit("bh_c extension not built; pip install -e src/ntropy")

    p = argparse.ArgumentParser()
    p.add_argument("work_dir", type=Path)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--substeps", type=int, default=3)
    p.add_argument("--mpi", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.work_dir / "ntropy_config.json")
    state = load_ic(args.work_dir / "ic_state.npz")
    n_workers = cfg.parallel.n_workers if args.mpi else 1
    dt_base = cfg.integrator.dt_base or cfg.integrator.timestep.dt_base

    legacy_force = bench_force(
        state, preset="legacy", n_repeat=args.repeat, mpi=args.mpi, n_workers=n_workers
    )
    opt_force = bench_force(
        state, preset="optimized", n_repeat=args.repeat, mpi=args.mpi, n_workers=n_workers
    )
    legacy_step = bench_substep(
        state,
        preset="legacy",
        n_substeps=args.substeps,
        mpi=args.mpi,
        n_workers=n_workers,
        dt_base=dt_base,
    )
    opt_step = bench_substep(
        state,
        preset="optimized",
        n_substeps=args.substeps,
        mpi=args.mpi,
        n_workers=n_workers,
        dt_base=dt_base,
    )

    print(f"work_dir={args.work_dir}")
    print(f"N={state.n:,} mpi={args.mpi} ranks={n_workers}")
    print(f"force eval: legacy={legacy_force:.1f}ms optimized={opt_force:.1f}ms speedup={legacy_force/opt_force:.2f}x")
    print(f"substep:    legacy={legacy_step:.1f}ms optimized={opt_step:.1f}ms speedup={legacy_step/opt_step:.2f}x")


if __name__ == "__main__":
    main()
