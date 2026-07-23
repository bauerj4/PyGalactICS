"""DBH campaign runner: solve → sample → evolve."""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal

import numpy as np

from galacticsics.campaign.manifest import CampaignManifest, append_manifest_row
from galacticsics.campaign.parallel_budget import (
    ParallelBudget,
    apply_parallel_env,
    format_parallel_budget,
    parallel_env_dict,
    resolve_parallel_budget,
)
from galacticsics.campaign.progress import CampaignProgress, stream_ntropy_jsonl
from galacticsics.campaign.serialize import model_to_dict
from galacticsics.campaign.spec import GridSpec, expand_grid, model_hash
from galacticsics.models import GalaxyModel

Stage = Literal["solve", "sample", "evolve"]

# Filenames in a model work dir that are not legacy ascii IC components.
_IC_SKIP_NAMES = frozenset(
    {
        "in.dbh",
        "in.gendenspsi",
        "model.json",
        "ntropy_config.json",
        "ic_state.npz",
        "merged.dat",
        "toomre2.5",
        "evolve_result.json",
        "dbh.dat",
        "mr.dat",
        "h.dat",
        "cordbh.dat",
        "freqdbh.dat",
        "denspsihalo.dat",
        "denspsibulge.dat",
        "dfnfw.dat",
        "dfsersic.dat",
        "dfhalo.table",
    }
)


def _ic_component_names(work_dir: Path, *, preferred: tuple[str, ...] = ()) -> list[str]:
    """Discover legacy ascii IC files (no extension) in ``work_dir``."""
    names: list[str] = []
    seen: set[str] = set()
    for name in preferred:
        if name in seen:
            continue
        if (work_dir / name).is_file():
            names.append(name)
            seen.add(name)
    for path in sorted(work_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        name = path.name
        if name in seen or name in _IC_SKIP_NAMES or "." in name:
            continue
        names.append(name)
        seen.add(name)
    return names


def _load_ic_particles(work_dir: Path, *, preferred: tuple[str, ...] = ()) -> dict:
    from galacticsics.sampling.particles import ParticleSet

    particles: dict = {}
    for name in _ic_component_names(work_dir, preferred=preferred):
        particles[name] = ParticleSet.from_ascii(work_dir / name, component=name)
    return particles


def _model_done_marker(work_dir: Path, stage: Stage) -> Path:
    return work_dir / f".done_{stage}"


def _mpi_evolve_available(n_ranks: int) -> bool:
    if n_ranks <= 1:
        return False
    if shutil.which("mpirun") is None:
        return False
    try:
        from ntropy.parallel.mpi import mpi_available

        return mpi_available()
    except ImportError:
        return False


def _venv_bin() -> Path | None:
    exe = Path(sys.executable).resolve()
    if exe.parent.name == "bin":
        return exe.parent
    return None


@contextmanager
def _scoped_parallel_env(budget: ParallelBudget) -> Iterator[None]:
    """Temporarily set OMP/BLAS thread limits for in-process evolve."""
    keys = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    saved = {k: os.environ.get(k) for k in keys}
    apply_parallel_env(budget)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _invalidate_diskdf_artifacts(work_dir: Path) -> None:
    """Drop disk DF tables so sample re-runs ``diskdf`` against fresh ``dbh.dat``."""
    for name in ("cordbh.dat", "toomre2.5"):
        path = work_dir / name
        if path.is_file():
            path.unlink()


def _cached_ics_unstable(
    work_dir: Path,
    *,
    counts: dict[str, int],
    eps_by_component: dict[str, float] | None,
    type_registry,
) -> bool:
    """True when on-disk ICs fail :func:`ic_looks_stable` heuristics."""
    if counts.get("disk", 0) <= 0:
        return False
    try:
        from galacticsics.campaign.benchmarks import ic_looks_stable
        from ntropy.integrations.galacticsics import merge_galacticsics_components

        particles = _load_ic_particles(work_dir, preferred=tuple(counts))
        if not particles:
            return False
        state = merge_galacticsics_components(
            particles,
            type_registry=type_registry,
            eps_by_component=eps_by_component or {},
        )
        return not ic_looks_stable(state)
    except Exception:
        return False


def _run_solve(
    model: GalaxyModel,
    work_dir: Path,
    *,
    progress: CampaignProgress | None = None,
    backend: str | None = None,
    solve_kwargs: dict | None = None,
) -> dict:
    from galacticsics.potential.solver import solve_potential

    solve_kwargs = solve_kwargs or {}
    stream = progress.stream_output if progress else False
    if progress:
        progress.legacy_command(f"dbh ({backend or 'python'})", str(work_dir))
    t0 = time.perf_counter()
    result = solve_potential(
        model,
        work_dir=work_dir,
        cleanup=False,
        stream_output=stream,
        backend=backend,
        **solve_kwargs,
    )
    elapsed = time.perf_counter() - t0
    diag = result.diagnostics
    _invalidate_diskdf_artifacts(work_dir)
    return {
        "solve_seconds": elapsed,
        "rtidal": getattr(diag, "tidal_radius", None),
        "work_dir": str(work_dir),
    }


def _run_sample(
    model: GalaxyModel,
    work_dir: Path,
    *,
    particles_by_component: dict[str, int] | None = None,
    n_disk: int = 0,
    n_halo: int = 0,
    n_bulge: int = 0,
    progress: CampaignProgress | None = None,
    backend: str | None = None,
    eps_by_component: dict[str, float] | None = None,
    type_registry=None,
    raw_config: dict | None = None,
) -> dict:
    from galacticsics.builder import GalaxyBuilder
    from galacticsics.campaign.run_config import (
        build_type_registry,
        default_walkthrough_config,
        particles_by_component as parse_particles,
        sample_config_kwargs,
        softening_eps_by_component,
    )

    if particles_by_component is None:
        particles_by_component = {
            "disk": n_disk,
            "halo": n_halo,
            "bulge": n_bulge,
        }
    counts = dict(particles_by_component)
    if not (model.disk and model.disk.enabled):
        counts["disk"] = 0
    if not (model.halo and model.halo.enabled):
        counts["halo"] = 0
    if not (model.bulge and model.bulge.enabled):
        counts["bulge"] = 0

    stream = progress.stream_output if progress else False
    builder = GalaxyBuilder(model=model, model_dir=work_dir)
    t0 = time.perf_counter()
    tag = backend or "python"
    sample_kw = sample_config_kwargs(raw_config or default_walkthrough_config())
    from galacticsics.sampling.openmp import openmp_sampler_status
    from galacticsics.sampling.sampler import SampleConfig as _SampleConfig

    sampler_status = openmp_sampler_status(_SampleConfig(n_disk=0, n_halo=0, **sample_kw))
    if progress:
        progress.log(f"  IC sampler: {sampler_status.log_label()}")

    def _on_stage(name: str) -> None:
        if progress:
            if name in ("gendisk", "genhalo"):
                progress.legacy_command(f"{name} [{sampler_status.log_label()}]", str(work_dir))
            else:
                progress.legacy_command(f"{name} [{tag}]", str(work_dir))

    builder.sample(
        n_disk=counts.get("disk", 0),
        n_halo=counts.get("halo", 0),
        n_bulge=counts.get("bulge", 0),
        work_dir=str(work_dir),
        cleanup=False,
        stream_output=stream,
        backend=backend,
        progress_log=progress.log if progress and progress.enabled else None,
        on_stage=_on_stage,
        **sample_kw,
    )
    file_counts = _validate_particle_files(work_dir)
    if progress:
        parts = [f"{name}={n:,}" for name, n in sorted(file_counts.items())]
        progress.log(f"  sampled {sum(file_counts.values()):,} particles ({', '.join(parts)})")
    summary = {
        "sample_seconds": time.perf_counter() - t0,
        "particles_dir": str(work_dir),
        "n_particles": sum(file_counts.values()),
        **{f"n_{k}": v for k, v in file_counts.items()},
    }
    from galacticsics.io.formats import read_toomre_q

    toomre_path = work_dir / "toomre2.5"
    if toomre_path.is_file():
        summary["toomre_q"] = read_toomre_q(toomre_path)
    try:
        from ntropy.integrations.galacticsics import merge_galacticsics_components

        cfg = raw_config or default_walkthrough_config()
        if eps_by_component is None:
            eps_by_component = softening_eps_by_component(cfg)
        registry = type_registry or build_type_registry(cfg)
        particles = _load_ic_particles(work_dir, preferred=tuple(counts))
        if particles:
            state = merge_galacticsics_components(
                particles,
                type_registry=registry,
                eps_by_component=eps_by_component,
            )
            from galacticsics.campaign.benchmarks import (
                diagnose_ic_stability,
                explain_ic_instability,
                ic_looks_stable,
            )

            ic_diag = diagnose_ic_stability(state)
            summary["ic_velocity_diag"] = ic_diag
            if counts.get("disk", 0) > 0 and not ic_looks_stable(state):
                raise RuntimeError(
                    f"disk IC velocity structure looks wrong: {explain_ic_instability(diag=ic_diag)} "
                    f"(diag={ic_diag}). Re-run solve+sample with skip_done=False."
                )
            summary["rotation_curve_ic"] = _write_rotation_diagnostics(
                work_dir, state, model=model, label="ic"
            )
    except RuntimeError:
        raise
    except Exception:
        pass
    return summary


def _default_force_method() -> str:
    """Prefer compiled Barnes–Hut when the C extension is built."""
    try:
        from ntropy.forces.bhtree_c import extension_available

        if extension_available():
            return "bh_c"
    except ImportError:
        pass
    return "bh"


def _default_bh_optimizations() -> "BhOptimizationsConfig":
    """Use optimized C kernels when bh_c is built; else legacy."""
    from ntropy.config import BhOptimizationsConfig

    try:
        from ntropy.forces.bhtree_c import extension_available

        preset = "optimized" if extension_available() else "legacy"
    except ImportError:
        preset = "legacy"
    return BhOptimizationsConfig.from_preset(preset)  # type: ignore[arg-type]


def _write_evolve_config(
    work_dir: Path,
    registry,
    *,
    end_time_gyr: float,
    diagnostics_every: int,
    particle_dump_every: int,
    mpi_ranks: int,
    force_method: str | None = None,
    bh_optimizations: "BhOptimizationsConfig | None" = None,
    force_rebuild_every: int = 5,
    force_theta: float = 0.6,
    force_active_subset: bool = True,
    mpi_local_trees: bool = True,
    dt_base: float = 0.025,
    timestep_eta: float = 0.025,
    max_timestep_bin: int = 6,
    timestep_update_every: int = 1,
    integrator_order: int = 2,
) -> Path:
    """
    Write ``ntropy_config.json`` for one campaign model's evolve stage.

    Parameters
    ----------
    work_dir : Path
        Model directory; the config references ``merged.dat`` therein.
    registry : TypeRegistry
        Particle type registry serialized into ``particle_types``.
    end_time_gyr : float
        Evolve duration [Gyr].
    diagnostics_every, particle_dump_every : int
        Tiered diagnostics / particle dump cadence (fine substeps).
    mpi_ranks : int
        Enables the ``parallel`` block when > 1.
    force_method : str, optional
        ``bh_c`` when the C extension is built, else ``bh`` (auto).
    bh_optimizations : BhOptimizationsConfig, optional
        C kernel preset/flags written under ``force.bh_optimizations``.
    force_rebuild_every, force_theta, force_active_subset : int, float, bool
        Barnes–Hut tree rebuild cadence, opening angle, and active-subset
        force evaluation toggle.
    mpi_local_trees : bool
        Gadget-style local octrees + LET under MPI (default True);
        False writes the replicated-tree fallback.
    dt_base, timestep_eta, max_timestep_bin, timestep_update_every : float, float, int, int
        Tiered leapfrog timestep hierarchy parameters.
    integrator_order : int
        Leapfrog order (1 or 2).

    Returns
    -------
    cfg_path : Path
        Path of the written JSON config.
    """
    cfg_path = work_dir / "ntropy_config.json"
    method = force_method or _default_force_method()
    bh_opts = bh_optimizations or _default_bh_optimizations()
    parallel = {
        "enabled": mpi_ranks > 1,
        "n_workers": max(1, mpi_ranks),
        "mode": "mpi",
    }
    cfg_data = {
        "particles": {"file": "merged.dat"},
        "particle_types": registry.to_config_dict(),
        "force": {
            "method": method,
            "theta": force_theta,
            "rebuild_every": force_rebuild_every,
            "active_subset": force_active_subset,
            "bh_optimizations": bh_opts.to_config_dict(),
            "mpi_local_trees": mpi_local_trees,
        },
        "parallel": parallel,
        "integrator": {
            "type": "tiered_leapfrog",
            "order": integrator_order,
            "dt_base": dt_base,
            "end_time_gyr": end_time_gyr,
            "timestep": {
                "eta": timestep_eta,
                "dt_base": dt_base,
                "max_bin": max_timestep_bin,
                "update_every": timestep_update_every,
            },
        },
        "output": {
            "dir": "evolution",
            "write_final": True,
            "diagnostics_every": diagnostics_every,
            "particle_dump_every": particle_dump_every,
            "write_particle_bins": True,
        },
    }
    if cfg_path.is_file():
        existing = json.loads(cfg_path.read_text())
        existing.setdefault("output", {})
        existing["output"].update(cfg_data["output"])
        existing["parallel"] = parallel
        existing["force"] = cfg_data["force"]
        existing.setdefault("integrator", {}).update(cfg_data["integrator"])
        cfg_path.write_text(json.dumps(existing, indent=2))
    else:
        cfg_path.write_text(json.dumps(cfg_data, indent=2))
    return cfg_path


def _particle_file_counts(work_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in _ic_component_names(work_dir):
        path = work_dir / name
        counts[name] = sum(1 for line in path.read_text().splitlines() if line.strip())
    return counts


def _validate_particle_files(work_dir: Path, *, min_total: int = 1) -> dict[str, int]:
    counts = _particle_file_counts(work_dir)
    total = sum(counts.values())
    if total < min_total:
        raise RuntimeError(
            f"particle files in {work_dir} are empty or missing "
            f"(counts={counts or 'none'}). "
            "Re-run sample with skip_done=False or delete .done_sample and empty disk/halo files."
        )
    return counts


def _save_ic_state(state, work_dir: Path) -> Path:
    if state.n < 1:
        raise RuntimeError(f"refusing to save empty ic_state.npz in {work_dir}")
    path = work_dir / "ic_state.npz"
    kwargs: dict = {
        "pos": state.pos,
        "vel": state.vel,
        "mass": state.mass,
        "eps": state.eps,
    }
    if state.type_id is not None:
        kwargs["type_id"] = state.type_id
    if state.timestep_bin is not None:
        kwargs["timestep_bin"] = state.timestep_bin
    if state.tags is not None:
        kwargs["tags"] = state.tags
    np.savez(path, **kwargs)
    return path


def _write_rotation_diagnostics(
    work_dir: Path,
    state,
    *,
    model: GalaxyModel | None = None,
    label: str,
) -> str:
    from galacticsics.diagnostics.rotation_curve import write_rotation_curve_diagnostic

    path = write_rotation_curve_diagnostic(
        work_dir, state=state, model=model, label=label
    )
    return str(path)


def _load_ic_state(work_dir: Path):
    from ntropy.particles import ParticleState

    ic_path = work_dir / "ic_state.npz"
    if not ic_path.is_file():
        return None
    with np.load(ic_path, allow_pickle=True) as data:
        kwargs = {
            "pos": data["pos"],
            "vel": data["vel"],
            "mass": data["mass"],
            "eps": data["eps"],
        }
        if "type_id" in data:
            kwargs["type_id"] = data["type_id"]
        return ParticleState.from_arrays(**kwargs)


def _load_final_state(work_dir: Path):
    from ntropy.particles import ParticleState

    final_path = work_dir / "evolution" / "final.dat"
    if not final_path.is_file():
        return None
    from ntropy.config import load_config

    cfg = load_config(work_dir / "ntropy_config.json")
    return ParticleState.from_file(final_path, config=cfg)


def _run_evolve(
    work_dir: Path,
    *,
    end_time_gyr: float = 1.0,
    diagnostics_every: int = 1,
    particle_dump_every: int = 50,
    mpi_ranks: int = 1,
    core_fraction: float = 0.75,
    force_rebuild_every: int = 5,
    force_theta: float = 0.6,
    force_active_subset: bool = True,
    mpi_local_trees: bool = True,
    dt_base: float = 0.025,
    timestep_eta: float = 0.025,
    max_timestep_bin: int = 6,
    timestep_update_every: int = 1,
    integrator_order: int = 2,
    bh_optimizations: "BhOptimizationsConfig | None" = None,
    eps_by_component: dict[str, float] | None = None,
    type_registry=None,
    raw_config: dict | None = None,
    progress: CampaignProgress | None = None,
    evolve_label: str = "evolve",
) -> dict:
    """
    Merge sampled ICs and run the N-body evolve stage for one model.

    Validates IC stability (virial diagnostic) before evolving. When MPI
    is available and ``mpi_ranks > 1``, launches
    ``ntropy.benchmark.mpi_simulation_worker`` via ``mpirun`` with the
    resolved core budget, tees worker output to
    ``work_dir/evolve_mpirun.log``, and tails the JSONL progress stream;
    otherwise runs the simulation in-process.

    Parameters
    ----------
    work_dir : Path
        Model directory containing sampled component particle files.
    end_time_gyr : float
        Evolve duration [Gyr].
    mpi_ranks, core_fraction : int, float
        Requested ranks and the share of logical CPUs to budget across
        ranks × OpenMP threads (ranks are clamped to the budget).
    mpi_local_trees : bool
        Local octrees + LET under MPI (default True).
    progress : CampaignProgress, optional
        Stage logger; silent when disabled.
    evolve_label : str
        Progress label for logs and the MPI worker.

    Other parameters mirror :func:`_write_evolve_config` and are passed
    through to the generated ``ntropy_config.json``.

    Returns
    -------
    summary : dict
        ``evolve_seconds``, ``dE_over_E0``, ``mpi`` flag, rank count, and
        rotation-curve diagnostic paths.

    Raises
    ------
    RuntimeError
        When ICs fail the stability gate or the MPI worker exits without
        writing ``evolve_result.json``.
    """
    from galacticsics.campaign.run_config import (
        build_type_registry,
        default_walkthrough_config,
        softening_eps_by_component,
    )
    from ntropy.config import load_config
    from ntropy.integrations.galacticsics import merge_galacticsics_components
    from ntropy.simulation import Simulation

    walk_cfg = raw_config or default_walkthrough_config()
    if eps_by_component is None:
        eps_by_component = softening_eps_by_component(walk_cfg)
    registry = type_registry or build_type_registry(walk_cfg)
    particles = _load_ic_particles(work_dir)

    if not particles:
        raise FileNotFoundError(f"no particle files in {work_dir}")

    _validate_particle_files(work_dir)
    state = merge_galacticsics_components(
        particles,
        type_registry=registry,
        eps_by_component=eps_by_component,
    )
    if state.n < 1:
        raise RuntimeError(f"merged particle state is empty in {work_dir}")

    use_mpi = _mpi_evolve_available(mpi_ranks)
    parallel_budget = resolve_parallel_budget(
        mpi_ranks if use_mpi else 1,
        core_fraction=core_fraction,
    )
    effective_mpi_ranks = parallel_budget.mpi_ranks if use_mpi else 1
    bh_opts = bh_optimizations or _default_bh_optimizations()
    if progress and progress.enabled:
        if use_mpi and effective_mpi_ranks != mpi_ranks:
            progress.log(
                f"  MPI ranks clamped {mpi_ranks} → {effective_mpi_ranks} "
                f"(core budget {parallel_budget.budget_cores})"
            )
        progress.log(f"  parallel: {format_parallel_budget(parallel_budget)}")
        progress.log(
            f"  force: {_default_force_method()} | bh_optimizations preset={bh_opts.preset} "
            f"| mpi_local_trees={mpi_local_trees}"
        )

    cfg_path = _write_evolve_config(
        work_dir,
        registry,
        end_time_gyr=end_time_gyr,
        diagnostics_every=diagnostics_every,
        particle_dump_every=particle_dump_every,
        mpi_ranks=effective_mpi_ranks,
        bh_optimizations=bh_opts,
        force_rebuild_every=force_rebuild_every,
        force_theta=force_theta,
        force_active_subset=force_active_subset,
        mpi_local_trees=mpi_local_trees,
        dt_base=dt_base,
        timestep_eta=timestep_eta,
        max_timestep_bin=max_timestep_bin,
        timestep_update_every=timestep_update_every,
        integrator_order=integrator_order,
    )
    state.write_ascii(work_dir / "merged.dat")
    _save_ic_state(state, work_dir)
    ic_rot_path = _write_rotation_diagnostics(work_dir, state, label="ic")

    from galacticsics.campaign.benchmarks import (
        diagnose_ic_stability,
        explain_ic_instability,
        ic_looks_stable,
    )

    ic_diag = diagnose_ic_stability(state)
    if not ic_looks_stable(state):
        msg = (
            f"refusing to evolve with unstable ICs: {explain_ic_instability(diag=ic_diag)} "
            f"(diag={ic_diag}). Re-run solve+sample with skip_done=False."
        )
        if progress and progress.enabled:
            progress.log(f"  ERROR: {msg}")
        raise RuntimeError(msg)

    if use_mpi:
        from ntropy.benchmark.mpi_subprocess import run_mpirun_simulation
        from ntropy.units import code_time_to_gyr

        cfg = load_config(cfg_path)
        cfg_dt_base = cfg.integrator.dt_base or cfg.integrator.timestep.dt_base
        n_substeps = max(
            1, int(round(end_time_gyr / code_time_to_gyr(1.0) / cfg_dt_base))
        )
        progress_jsonl = work_dir / "evolution" / "diagnostics.progress.jsonl"
        progress_jsonl.parent.mkdir(parents=True, exist_ok=True)
        progress_jsonl.write_text("")
        tail_stop = threading.Event()
        tail_thread: threading.Thread | None = None
        if progress and progress.enabled:
            progress.mpi_launch(effective_mpi_ranks, evolve_label)
            progress.log(
                f"  ntropy progress: {n_substeps} fine substeps "
                f"(step/t, active %, bin histogram, dE/E0)"
            )
            tail_thread = threading.Thread(
                target=stream_ntropy_jsonl,
                kwargs={
                    "path": progress_jsonl,
                    "n_steps_total": n_substeps,
                    "stop_event": tail_stop,
                },
                daemon=True,
            )
            tail_thread.start()
        mpi_env = {
            "NTROPY_PROGRESS_JSONL_ONLY": "1",
            **parallel_env_dict(parallel_budget),
        }
        t0 = time.perf_counter()
        try:
            run_mpirun_simulation(
                effective_mpi_ranks,
                ["--campaign-dir", str(work_dir), "--label", evolve_label],
                cwd=work_dir,
                venv_bin=_venv_bin(),
                extra_env=mpi_env,
                log_path=work_dir / "evolve_mpirun.log",
            )
        finally:
            tail_stop.set()
            if tail_thread is not None:
                tail_thread.join(timeout=2.0)
        elapsed = time.perf_counter() - t0
        result_path = work_dir / "evolve_result.json"
        if not result_path.is_file():
            raise RuntimeError(f"MPI evolve did not write {result_path}")
        payload = json.loads(result_path.read_text())
        return {
            "evolve_seconds": elapsed,
            "dE_over_E0": payload.get("dE_over_E0"),
            "n_energies": payload.get("n_energies"),
            "n_ranks": payload.get("n_ranks", effective_mpi_ranks),
            "mpi": True,
            "rotation_curve_ic": ic_rot_path,
            "rotation_curve_final": _write_rotation_diagnostics(
                work_dir,
                _load_final_state(work_dir),
                label="final",
            ),
        }

    if progress and mpi_ranks > 1:
        progress.log(
            f"  MPI unavailable (need mpi4py + mpirun); running serial evolve"
        )

    cfg = load_config(cfg_path)
    t0 = time.perf_counter()
    with _scoped_parallel_env(parallel_budget):
        result = Simulation(cfg, state=state).run(
            show_progress=bool(progress and progress.enabled),
            progress_desc=evolve_label,
            print_config=bool(progress and progress.enabled),
        )
    elapsed = time.perf_counter() - t0
    e0 = result.energies[0] if result.energies else 0.0
    ef = result.energies[-1] if result.energies else 0.0
    dE = abs(ef - e0) / max(abs(e0), 1e-30)
    final_rot_path = _write_rotation_diagnostics(
        work_dir, result.final_state, label="final"
    )
    return {
        "evolve_seconds": elapsed,
        "dE_over_E0": dE,
        "n_energies": len(result.energies),
        "n_ranks": 1,
        "mpi": False,
        "rotation_curve_ic": ic_rot_path,
        "rotation_curve_final": final_rot_path,
    }


def run_campaign(
    spec: GridSpec,
    work_root: Path | str,
    *,
    stages: list[Stage] | None = None,
    particles_by_component: dict[str, int] | None = None,
    n_disk: int = 5000,
    n_halo: int = 10000,
    n_bulge: int = 2000,
    end_time_gyr: float = 1.0,
    dt_base: float = 0.025,
    timestep_eta: float = 0.025,
    max_timestep_bin: int = 6,
    timestep_update_every: int = 1,
    integrator_order: int = 2,
    diagnostics_every: int = 1,
    particle_dump_every: int = 50,
    skip_done: bool = True,
    verbose: bool = True,
    mpi_ranks: int = 1,
    core_fraction: float = 0.75,
    force_rebuild_every: int = 5,
    force_theta: float = 0.6,
    force_active_subset: bool = True,
    mpi_local_trees: bool = True,
    bh_optimizations_preset: str = "optimized",
    bh_optimizations_extra: dict | None = None,
    eps_by_component: dict[str, float] | None = None,
    raw_config: dict | None = None,
    solve_kwargs: dict | None = None,
    progress: CampaignProgress | None = None,
) -> CampaignManifest:
    """
    Execute a parameter grid campaign.

    Parameters
    ----------
    spec : GridSpec
        Grid definition.
    work_root : path
        Root directory for all model work dirs.
    stages : list of str, optional
        Subset of ``solve``, ``sample``, ``evolve``.
    n_disk, n_halo, n_bulge : int
        Particle counts for sampling (legacy; prefer ``particles_by_component``).
    particles_by_component : dict, optional
        Per-component IC counts keyed by label (e.g. ``disk``, ``halo``, ``gas``).
    end_time_gyr : float
        Evolution duration for ntropy stage.
    dt_base : float
        Finest tiered substep size [code units] (bin 0).
    timestep_eta : float
        Accuracy parameter η for per-particle bin assignment.
    max_timestep_bin : int
        Coarsest allowed bin (max Δt = ``dt_base * 2**max_bin``).
    timestep_update_every : int
        Recompute timestep bins every N fine substeps.
    integrator_order : int
        Leapfrog order (``1`` or ``2``).
    diagnostics_every : int
        Record tiered bin diagnostics every N fine substeps.
    particle_dump_every : int
        Write per-particle ``.npz`` bin dumps every N recorded steps.
    skip_done : bool
        Skip stages with ``.done_{stage}`` marker files.
    verbose : bool
        Print timestamped stage logs, ETAs, and stream legacy/MPI output.
    mpi_ranks : int
        MPI ranks for ntropy evolve (``1`` by default for single-node CPU).
        Use ``2``+ only when ``mpi4py`` and ``mpirun`` are available and
        particle count is large enough to amortize communication.
    core_fraction : float
        Share of logical CPUs for MPI×OpenMP (default ``0.75``).
    force_rebuild_every : int
        Rebuild Barnes–Hut tree every N fine substeps (default ``5``).
        Larger values reduce tree-build overhead on CPU at the cost of
        slightly lower force accuracy between rebuilds.
    mpi_local_trees : bool
        When True (default), MPI Barnes–Hut uses per-rank local octrees plus
        a Local Essential Tree (LET) exchange. Set False for a full replicated
        tree on every rank.
    bh_optimizations_preset : {'legacy', 'optimized'}
        C Barnes–Hut kernel preset written to ``ntropy_config.json``
        (only applies when ``force.method`` is ``bh_c``).
    progress : CampaignProgress, optional
        Custom progress reporter (defaults to :class:`CampaignProgress`).

    Returns
    -------
    CampaignManifest
    """
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    stages = stages or ["solve", "sample", "evolve"]
    manifest = CampaignManifest(name=spec.name, work_root=work_root)

    reporter = progress if progress is not None else CampaignProgress(enabled=verbose)
    models = list(expand_grid(spec))
    from galacticsics.physics.backend import resolve_physics_backend
    from ntropy.config import BhOptimizationsConfig

    physics_backend = resolve_physics_backend(spec.physics_backend).value
    if bh_optimizations_preset not in ("legacy", "optimized"):
        raise ValueError(
            f"bh_optimizations_preset must be 'legacy' or 'optimized', "
            f"got {bh_optimizations_preset!r}"
        )
    bh_optimizations = BhOptimizationsConfig.from_dict(
        {"preset": bh_optimizations_preset, **(bh_optimizations_extra or {})}
    )
    from galacticsics.campaign.run_config import (
        build_type_registry,
        default_walkthrough_config,
        particles_by_component as parse_particles,
        softening_eps_by_component,
    )

    cfg = raw_config or default_walkthrough_config()
    if particles_by_component is None:
        particles_by_component = parse_particles(cfg)
        if not any(particles_by_component.values()):
            particles_by_component = {"disk": n_disk, "halo": n_halo, "bulge": n_bulge}
    if eps_by_component is None:
        eps_by_component = softening_eps_by_component(cfg)
    type_registry = build_type_registry(cfg)
    parts_summary = ", ".join(
        f"{k}={v:,}" for k, v in sorted(particles_by_component.items()) if v
    )
    reporter.banner(
        name=spec.name,
        work_root=str(work_root),
        stages=stages,
        n_models=len(models),
        extra={
            "particles": parts_summary or "none",
            "end_time_gyr": end_time_gyr,
            "dt_base": dt_base,
            "mpi_ranks": mpi_ranks,
            "core_fraction": core_fraction,
            "bh_optimizations": bh_optimizations.preset,
            "force_theta": force_theta,
            "force_active_subset": force_active_subset,
            "force_rebuild_every": force_rebuild_every,
            "softening_kpc": eps_by_component,
            "diagnostics_every": diagnostics_every,
            "skip_done": skip_done,
            "physics_backend": physics_backend,
        },
    )

    for model_idx, (label, model) in enumerate(models, start=1):
        mhash = model_hash(model)
        model_dir = work_root / mhash
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model.json").write_text(json.dumps(model_to_dict(model), indent=2))

        row: dict = {"label": label, "hash": mhash, "path": str(model_dir)}
        reporter.start_model(model_idx, len(models), label, mhash, str(model_dir))
        model_t0 = time.perf_counter()

        if "solve" in stages:
            marker = _model_done_marker(model_dir, "solve")
            if skip_done and marker.exists():
                reporter.skip_stage("solve", reason=".done_solve")
            else:
                reporter.start_stage(
                    "solve",
                    detail=f"dbh (nr={model.grid.nr}, lmax={model.grid.lmax})",
                )
                t0 = time.perf_counter()
                summary = _run_solve(
                    model,
                    model_dir,
                    progress=reporter,
                    backend=physics_backend,
                    solve_kwargs=solve_kwargs,
                )
                row.update(summary)
                reporter.end_stage("solve", time.perf_counter() - t0, summary)
                marker.touch()

        if "sample" in stages:
            marker = _model_done_marker(model_dir, "sample")
            sample_counts = _particle_file_counts(model_dir)
            sample_done = (
                skip_done
                and marker.exists()
                and sum(sample_counts.values()) > 0
            )
            if sample_done:
                if _cached_ics_unstable(
                    model_dir,
                    counts=particles_by_component,
                    eps_by_component=eps_by_component,
                    type_registry=type_registry,
                ):
                    from galacticsics.campaign.benchmarks import invalidate_ic_artifacts

                    removed = invalidate_ic_artifacts(
                        model_dir,
                        include_evolve=False,
                        include_solve_marker=False,
                    )
                    reporter.log(
                        "  cached ICs look unstable — re-running sample"
                        + (f" (removed {', '.join(removed)})" if removed else "")
                    )
                    sample_done = False
                else:
                    reporter.skip_stage("sample", reason=".done_sample")
            if not sample_done:
                if skip_done and marker.exists():
                    reporter.log(
                        "  sample marker present but particle files empty — re-running sample"
                    )
                reporter.start_stage(
                    "sample",
                    detail=parts_summary or "no particles requested",
                )
                t0 = time.perf_counter()
                summary = _run_sample(
                    model,
                    model_dir,
                    particles_by_component=particles_by_component,
                    progress=reporter,
                    backend=physics_backend,
                    eps_by_component=eps_by_component,
                    type_registry=type_registry,
                    raw_config=cfg,
                )
                row.update(summary)
                reporter.end_stage("sample", time.perf_counter() - t0, summary)
                marker.touch()

        if "evolve" in stages:
            marker = _model_done_marker(model_dir, "evolve")
            evolve_ok = (
                skip_done
                and marker.exists()
                and (
                    (model_dir / "evolve_result.json").is_file()
                    or (model_dir / "evolution" / "final.dat").is_file()
                )
            )
            if evolve_ok and skip_done and marker.exists():
                reporter.skip_stage("evolve", reason=".done_evolve")
            else:
                if skip_done and marker.exists():
                    reporter.log(
                        "  evolve marker present but outputs missing — re-running evolve"
                    )
                mpi_note = (
                    f"mpirun -n {mpi_ranks}"
                    if _mpi_evolve_available(mpi_ranks)
                    else "serial (MPI unavailable)"
                )
                reporter.start_stage(
                    "evolve",
                    detail=f"{end_time_gyr} Gyr tiered | {mpi_note}",
                )
                t0 = time.perf_counter()
                summary = _run_evolve(
                    model_dir,
                    end_time_gyr=end_time_gyr,
                    diagnostics_every=diagnostics_every,
                    particle_dump_every=particle_dump_every,
                    mpi_ranks=mpi_ranks,
                    core_fraction=core_fraction,
                    force_rebuild_every=force_rebuild_every,
                    force_theta=force_theta,
                    force_active_subset=force_active_subset,
                    mpi_local_trees=mpi_local_trees,
                    dt_base=dt_base,
                    timestep_eta=timestep_eta,
                    max_timestep_bin=max_timestep_bin,
                    timestep_update_every=timestep_update_every,
                    integrator_order=integrator_order,
                    bh_optimizations=bh_optimizations,
                    eps_by_component=eps_by_component,
                    type_registry=type_registry,
                    raw_config=cfg,
                    progress=reporter,
                    evolve_label=f"{label} evolve",
                )
                row.update(summary)
                reporter.end_stage("evolve", time.perf_counter() - t0, summary)
                marker.touch()

        append_manifest_row(manifest, row)
        reporter.end_model(time.perf_counter() - model_t0)

    reporter.finish()
    return manifest


def run_campaign_from_config(
    config_path: Path | str,
    *,
    work_root: Path | str | None = None,
    grid_spec: GridSpec | None = None,
    repo: Path | str | None = None,
    raw: dict | None = None,
) -> CampaignManifest:
    """
    Run a campaign from a walkthrough JSON config (see ``notebooks/campaigns/``).

    Parameters
    ----------
    config_path : path
        JSON file with ``particles``, ``evolve``, ``force``, and ``base_model`` blocks.
    work_root : path, optional
        Override output root (default: ``artifacts.base_mw`` or ``artifacts.runs``).
    grid_spec : GridSpec, optional
        Override grid definition (default: from config ``base_model.grid`` or sweep).
    repo : path, optional
        Repository root for resolving relative paths.
    raw : dict, optional
        In-memory config (e.g. notebook ``CONFIG``); overrides on-disk JSON.
    """
    from galacticsics.campaign.run_config import WalkthroughConfig, load_walkthrough_config

    if raw is not None:
        cfg = WalkthroughConfig.from_raw(
            raw, config_path=Path(config_path), repo=Path(repo) if repo else None
        )
    else:
        cfg = load_walkthrough_config(config_path, repo=Path(repo) if repo else None)
    cfg.ensure_artifact_dirs()
    spec = grid_spec or cfg.base_grid
    root = Path(work_root) if work_root is not None else cfg.paths.base_root
    kwargs = dict(cfg.run_campaign_kwargs())
    extra = kwargs.pop("bh_optimizations_extra", None)
    kwargs["raw_config"] = cfg.raw
    return run_campaign(
        spec,
        root,
        bh_optimizations_extra=extra,
        **kwargs,
    )
