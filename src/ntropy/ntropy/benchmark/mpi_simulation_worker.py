"""MPI simulation worker for notebook energy-drift runs (invoked via mpirun)."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

from ntropy.config import ForceConfig, IntegratorConfig, ParallelConfig, RunConfig, load_config
from ntropy.integrators.timestep import TimestepConfig
from ntropy.particles import ParticleState
from ntropy.simulation import Simulation
from ntropy.units import DEFAULT_SIM_DT, DEFAULT_SIM_N_STEPS


def _load_config(raw: dict) -> RunConfig:
    """
    Build a :class:`RunConfig` from the notebook's energy-run JSON dict.

    Honours ``force.mpi_local_trees`` (default True) so notebook energy
    runs exercise the same Gadget-style LET path as campaign evolves.
    Supports fixed-step integrators and ``tiered_leapfrog`` (``eta``,
    ``dt_base``, ``end_time_gyr``, ``max_bin``).

    Parameters
    ----------
    raw : dict
        Parsed JSON with ``integrator``, ``force``, and ``parallel`` blocks
        (see ``_energy_config_dict`` in the NFW walkthrough notebook).

    Returns
    -------
    cfg : RunConfig
        Config with output writing disabled (energies only).
    """
    cfg = RunConfig()
    integ = raw["integrator"]
    integ_type = integ.get("type", "leapfrog")
    dt = float(integ.get("dt", DEFAULT_SIM_DT))
    n_steps = int(integ.get("n_steps", DEFAULT_SIM_N_STEPS))
    dt_base_raw = integ.get("dt_base")
    end_gyr_raw = integ.get("end_time_gyr")
    ts_raw = integ.get("timestep") or {}
    dt_base_val = float(dt_base_raw) if dt_base_raw is not None else dt
    ts_config = TimestepConfig(
        eta=float(ts_raw.get("eta", 0.025)),
        dt_base=float(ts_raw.get("dt_base", dt_base_val)),
        max_bin=int(ts_raw.get("max_bin", 6)),
        update_every=max(1, int(ts_raw.get("update_every", 1))),
        accel_floor=float(ts_raw.get("accel_floor", 1e-6)),
    )
    cfg.integrator = IntegratorConfig(
        type=integ_type,
        order=int(integ.get("order", 2)),
        dt=dt,
        n_steps=n_steps,
        dt_base=float(dt_base_raw) if dt_base_raw is not None else None,
        end_time_gyr=float(end_gyr_raw) if end_gyr_raw is not None else None,
        timestep=ts_config,
    )
    force = raw["force"]
    cfg.force = ForceConfig(
        method=force.get("method", "bh"),
        theta=float(force.get("theta", 0.5)),
        mpi_local_trees=bool(force.get("mpi_local_trees", True)),
    )
    par = raw.get("parallel", {})
    cfg.parallel = ParallelConfig(
        enabled=bool(par.get("enabled", True)),
        n_workers=int(par.get("n_workers", 1)),
    )
    out = raw.get("output", {})
    cfg.output.write_final = False
    cfg.output.every = 0
    cfg.output.diagnostics_every = max(0, int(out.get("diagnostics_every", 0)))
    cfg.output.energy_every = max(1, int(out.get("energy_every", 1)))
    return cfg


def _load_state_npz(path: Path) -> ParticleState:
    """
    Load a :class:`ParticleState` from a campaign ``ic_state.npz`` dump.

    Parameters
    ----------
    path : Path
        Archive with ``pos``/``vel``/``mass``/``eps`` and optional
        ``type_id``/``timestep_bin``/``tags`` arrays.

    Returns
    -------
    state : ParticleState
    """
    with np.load(path, allow_pickle=True) as data:
        state = ParticleState.from_arrays(
            data["pos"],
            data["vel"],
            data["mass"],
            data["eps"],
            type_id=data["type_id"] if "type_id" in data else None,
            timestep_bin=data["timestep_bin"] if "timestep_bin" in data else None,
        )
        if "tags" in data:
            state.tags = data["tags"]
    return state


def _run_campaign_dir(work_dir: Path, *, label: str) -> int:
    """
    Run a campaign tiered evolve from a prepared work directory.

    Reads ``ntropy_config.json`` and ``ic_state.npz`` from ``work_dir``,
    runs the simulation under MPI, and writes ``evolve_result.json``
    (rank 0). Errors are dumped to ``evolve_error.log`` before aborting
    all ranks.

    Parameters
    ----------
    work_dir : Path
        Campaign model directory prepared by the galacticsics runner.
    label : str
        Progress-bar label shown on rank 0.

    Returns
    -------
    exit_code : int
        ``0`` on success; ``comm.Abort(1)`` on failure.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    try:
        cfg_path = work_dir / "ntropy_config.json"
        cfg = load_config(cfg_path)
        cfg.parallel.enabled = True

        state_path = work_dir / "ic_state.npz"
        if not state_path.is_file():
            raise FileNotFoundError(f"missing {state_path}")
        state = _load_state_npz(state_path)

        result = Simulation(cfg, state=state.copy()).run(
            show_progress=rank == 0,
            progress_desc=label,
            print_config=rank == 0,
        )

        if rank == 0:
            e0 = result.energies[0] if result.energies else 0.0
            ef = result.energies[-1] if result.energies else 0.0
            dE = abs(ef - e0) / max(abs(e0), 1e-30)
            out_path = work_dir / "evolve_result.json"
            payload = {
                "energies": result.energies,
                "dE_over_E0": dE,
                "n_energies": len(result.energies),
                "n_ranks": comm.Get_size(),
                "integrator_type": cfg.integrator.type,
                "force_method": cfg.force.method,
                "label": label,
            }
            out_path.write_text(json.dumps(payload, indent=2))
        comm.Barrier()
        return 0
    except Exception:
        err_path = work_dir / "evolve_error.log"
        if rank == 0:
            tb = traceback.format_exc()
            err_path.write_text(tb)
            print(tb, file=sys.stderr)
        comm.Abort(1)
        return 1


def main(argv: list[str]) -> int:
    """
    Run a simulation under MPI and write energies (rank 0 only).

    Usage
    -----
    Legacy::

        python -m ntropy.benchmark.mpi_simulation_worker \\
            <state.npz> <config.json> <out.json> [final_state.npz]

    Campaign (tiered evolve with diagnostics)::

        python -m ntropy.benchmark.mpi_simulation_worker \\
            --campaign-dir <work_dir> [--label NAME]
    """
    if len(argv) >= 3 and argv[1] == "--campaign-dir":
        work_dir = Path(argv[2])
        label = "campaign evolve"
        if "--label" in argv:
            label = argv[argv.index("--label") + 1]
        return _run_campaign_dir(work_dir, label=label)

    if len(argv) not in (4, 5):
        raise SystemExit(
            f"usage: {argv[0]} --campaign-dir <work_dir> [--label NAME]\n"
            f"   or: {argv[0]} <state.npz> <config.json> <out.json> [final_state.npz]"
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
                type_id=data["type_id"] if "type_id" in data else None,
                timestep_bin=data["timestep_bin"] if "timestep_bin" in data else None,
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
            integ = cfg.integrator
            payload = {
                "energies": result.energies,
                "kinetic_energies": result.kinetic_energies,
                "dt": integ.dt,
                "n_steps": integ.n_steps,
                "dt_base": integ.dt_base if integ.dt_base is not None else integ.timestep.dt_base,
                "end_time_gyr": integ.end_time_gyr,
                "eta": integ.timestep.eta,
                "integrator_type": integ.type,
                "integrator_order": integ.order,
                "force_method": cfg.force.method,
                "n_ranks": comm.Get_size(),
            }
            out_path.write_text(json.dumps(payload, indent=2))
            if final_path is not None:
                save_kw = dict(
                    pos=result.final_state.pos,
                    vel=result.final_state.vel,
                    mass=result.final_state.mass,
                    eps=result.final_state.eps,
                )
                if result.final_state.type_id is not None:
                    save_kw["type_id"] = result.final_state.type_id
                if result.final_state.timestep_bin is not None:
                    save_kw["timestep_bin"] = result.final_state.timestep_bin
                np.savez(final_path, **save_kw)
        comm.Barrier()
        return 0
    except Exception:
        err_path = out_path.parent / "evolve_error.log"
        if rank == 0:
            tb = traceback.format_exc()
            err_path.write_text(tb)
            print(tb, file=sys.stderr)
        comm.Abort(1)
        return 1
