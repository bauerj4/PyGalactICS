"""MPI worker: compare local-tree LET forces vs replicated full-tree BH."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

from ntropy.parallel.mpi import compute_forces_mpi


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        raise SystemExit(f"usage: {argv[0]} <state.npz> <out.npz> <theta>")
    state_path = Path(argv[1])
    out_path = Path(argv[2])
    theta = float(argv[3])
    comm = MPI.COMM_WORLD

    with np.load(state_path) as data:
        pos = np.asarray(data["pos"], dtype=float)
        mass = np.asarray(data["mass"], dtype=float)
        eps = np.asarray(data["eps"], dtype=float)

    acc_let = compute_forces_mpi(
        pos, mass, eps, method="bh_c", theta=theta, comm=comm, mpi_local_trees=True
    )
    acc_rep = compute_forces_mpi(
        pos, mass, eps, method="bh_c", theta=theta, comm=comm, mpi_local_trees=False
    )
    if comm.Get_rank() == 0:
        np.savez(out_path, acc_let=acc_let, acc_rep=acc_rep)
    comm.Barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
