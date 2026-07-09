"""MPI subprocess worker for multi-rank density parity tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ntropy.config import ForceConfig, IntegratorConfig, ParallelConfig, RunConfig
from ntropy.particles import ParticleState
from ntropy.simulation import Simulation


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit(f"usage: {argv[0]} <initial.npz> <final.npz>")

    state_path = Path(argv[1])
    out_path = Path(argv[2])

    with np.load(state_path, allow_pickle=True) as data:
        state = ParticleState.from_arrays(
            data["pos"],
            data["vel"],
            data["mass"],
            data["eps"],
        )
        state.tags = data["tags"]

    cfg = RunConfig()
    cfg.integrator = IntegratorConfig(dt=0.002, n_steps=14)
    cfg.force = ForceConfig(method="brute")
    cfg.parallel = ParallelConfig(enabled=True, n_workers=2)
    cfg.output.write_final = False
    cfg.output.every = 0

    result = Simulation(cfg, state=state.copy()).run()
    final = result.final_state

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        np.savez(
            out_path,
            pos=final.pos,
            vel=final.vel,
            mass=final.mass,
            eps=final.eps,
            tags=final.tags,
        )
    comm.Barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
