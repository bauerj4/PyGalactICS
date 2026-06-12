#!/usr/bin/env python3
"""MPI force-evaluation benchmark worker for notebook subprocess calls."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from ntropy.parallel.mpi import compute_forces_mpi


def main(argv: list[str]) -> int:
    if len(argv) != 6:
        raise SystemExit(
            f"usage: {argv[0]} <state.npz> <method> <theta> <n_repeat> <out.json>"
        )

    state_path = Path(argv[1])
    method = argv[2]
    theta = float(argv[3])
    n_repeat = int(argv[4])
    out_path = Path(argv[5])

    with np.load(state_path, allow_pickle=True) as data:
        pos = data["pos"]
        mass = data["mass"]
        eps = data["eps"]

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    for _ in range(2):
        compute_forces_mpi(pos, mass, eps, method=method, theta=theta, comm=comm)
    comm.Barrier()

    if rank == 0:
        start = time.perf_counter()
    for _ in range(n_repeat):
        compute_forces_mpi(pos, mass, eps, method=method, theta=theta, comm=comm)
    comm.Barrier()

    elapsed = None
    if rank == 0:
        elapsed = time.perf_counter() - start
        payload = {
            "method": method,
            "theta": theta,
            "n_repeat": n_repeat,
            "n_particles": int(len(mass)),
            "n_ranks": comm.Get_size(),
            "elapsed_s": elapsed,
            "time_per_force_s": elapsed / n_repeat,
        }
        out_path.write_text(json.dumps(payload, indent=2))
    comm.Barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
