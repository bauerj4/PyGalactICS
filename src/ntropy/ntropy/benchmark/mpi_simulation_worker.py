"""MPI simulation worker for notebook energy-drift runs (invoked via mpirun)."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

from ntropy.config import ForceConfig, IntegratorConfig, ParallelConfig, RunConfig
from ntropy.particles import ParticleState
from ntropy.simulation import Simulation


def _load_config(raw: dict) -> RunConfig:
    cfg = RunConfig()
    integ = raw["integrator"]
    cfg.integrator = IntegratorConfig(
        type=integ.get("type", "leapfrog"),
        order=int(integ.get("order", 2)),
        dt=float(integ["dt"]),
        n_steps=int(integ["n_steps"]),
    )
    force = raw["force"]
    cfg.force = ForceConfig(
        method=force.get("method", "bh"),
        theta=float(force.get("theta", 0.5)),
    )
    par = raw.get("parallel", {})
    cfg.parallel = ParallelConfig(
        enabled=bool(par.get("enabled", True)),
        n_workers=int(par.get("n_workers", 1)),
    )
    cfg.output.write_final = False
    cfg.output.every = 0
    return cfg


def main(argv: list[str]) -> int:
    """
    Run a simulation under MPI and write energies (rank 0 only).

    Usage: python -m ntropy.benchmark.mpi_simulation_worker \\
        <state.npz> <config.json> <out.json> [final_state.npz]
    """
    if len(argv) not in (4, 5):
        raise SystemExit(
            f"usage: {argv[0]} <state.npz> <config.json> <out.json> [final_state.npz]"
        )

    state_path = Path(argv[1])
    config_path = Path(argv[2])
    out_path = Path(argv[3])
    final_path = Path(argv[4]) if len(argv) == 5 else None

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    try:
        with np.load(state_path, allow_pickle=True) as data:
            state = ParticleState.from_arrays(
                data["pos"],
                data["vel"],
                data["mass"],
                data["eps"],
            )
            if "tags" in data:
                state.tags = data["tags"]

        raw = json.loads(config_path.read_text())
        cfg = _load_config(raw)
        label = raw.get("label", "MPI simulation")
        result = Simulation(cfg, state=state.copy()).run(
            show_progress=rank == 0,
            progress_desc=label,
            print_config=rank == 0,
        )

        if rank == 0:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "energies": result.energies,
                "dt": cfg.integrator.dt,
                "n_steps": cfg.integrator.n_steps,
                "integrator_type": cfg.integrator.type,
                "integrator_order": cfg.integrator.order,
                "force_method": cfg.force.method,
                "n_ranks": comm.Get_size(),
            }
            out_path.write_text(json.dumps(payload, indent=2))
            if final_path is not None:
                np.savez(
                    final_path,
                    pos=result.final_state.pos,
                    vel=result.final_state.vel,
                    mass=result.final_state.mass,
                    eps=result.final_state.eps,
                )
        comm.Barrier()
        return 0
    except Exception:
        if rank == 0:
            print(traceback.format_exc(), file=sys.stderr)
        comm.Abort(1)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
