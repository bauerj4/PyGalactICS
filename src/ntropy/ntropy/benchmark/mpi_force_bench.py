"""MPI force-evaluation benchmark worker (invoked via mpirun)."""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from ntropy.parallel.mpi import compute_forces_mpi


def main(argv: list[str]) -> int:
    """
    Benchmark one force-evaluation configuration under MPI.

    Usage: python -m ntropy.benchmark.mpi_force_bench \\
        <state.npz> <method> <theta> <n_repeat> <out.json> [mpi_local_trees]

    Parameters
    ----------
    argv : list of str
        ``argv[1]`` particle ``.npz`` with ``pos``/``mass``/``eps``;
        ``argv[2]`` force method (``brute``/``bh``/``bh_c``);
        ``argv[3]`` Barnes–Hut opening angle θ;
        ``argv[4]`` timed repeat count (2 warmup calls precede timing);
        ``argv[5]`` output JSON path (written by rank 0);
        ``argv[6]`` optional local-trees toggle (``1``/``true`` = LET
        path, ``0``/``false`` = replicated tree; default true).

    Returns
    -------
    exit_code : int
        ``0`` on success. On error, the traceback is printed on rank 0
        and ``comm.Abort(1)`` terminates all ranks.
    """
    if len(argv) not in (6, 7):
        raise SystemExit(
            f"usage: {argv[0]} <initial.npz> <method> <theta> <n_repeat> "
            f"<out.json> [mpi_local_trees]"
        )

    state_path = Path(argv[1])
    method = argv[2]
    theta = float(argv[3])
    n_repeat = int(argv[4])
    out_path = Path(argv[5])
    mpi_local_trees = True
    if len(argv) == 7:
        mpi_local_trees = argv[6].strip().lower() in ("1", "true", "yes", "on")

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    try:
        with np.load(state_path, allow_pickle=True) as data:
            pos = np.asarray(data["pos"], dtype=float)
            mass = np.asarray(data["mass"], dtype=float)
            eps = np.asarray(data["eps"], dtype=float)

        for _ in range(2):
            compute_forces_mpi(
                pos,
                mass,
                eps,
                method=method,
                theta=theta,
                comm=comm,
                mpi_local_trees=mpi_local_trees,
            )
        comm.Barrier()

        if rank == 0:
            start = time.perf_counter()
        for _ in range(n_repeat):
            compute_forces_mpi(
                pos,
                mass,
                eps,
                method=method,
                theta=theta,
                comm=comm,
                mpi_local_trees=mpi_local_trees,
            )
        comm.Barrier()

        if rank == 0:
            elapsed = time.perf_counter() - start
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "method": method,
                "theta": theta,
                "n_repeat": n_repeat,
                "n_particles": int(len(mass)),
                "n_ranks": comm.Get_size(),
                "mpi_local_trees": mpi_local_trees,
                "elapsed_s": elapsed,
                "time_per_force_s": elapsed / n_repeat,
            }
            out_path.write_text(json.dumps(payload, indent=2))
        comm.Barrier()
        return 0
    except Exception:
        msg = traceback.format_exc()
        if rank == 0:
            print(msg, file=sys.stderr)
        comm.Abort(1)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
