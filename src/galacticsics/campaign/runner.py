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


def _run_solve(
    model: GalaxyModel,
    work_dir: Path,
    *,
    progress: CampaignProgress | None = None,
) -> dict:
    from galacticsics.potential.solver import solve_potential

    stream = progress.stream_output if progress else False
    if progress:
        progress.legacy_command("dbh", str(work_dir))
    t0 = time.perf_counter()
    result = solve_potential(
        model,
        work_dir=work_dir,
        cleanup=False,
        stream_output=stream,
    )
    elapsed = time.perf_counter() - t0
    diag = result.diagnostics
    return {
        "solve_seconds": elapsed,
        "rtidal": getattr(diag, "tidal_radius", None),
        "work_dir": str(work_dir),
    }


def _run_sample(
    model: GalaxyModel,
    work_dir: Path,
    *,
    n_disk: int,
    n_halo: int,
    n_bulge: int,
    progress: CampaignProgress | None = None,
) -> dict:
    from galacticsics.builder import GalaxyBuilder

    stream = progress.stream_output if progress else False
    builder = GalaxyBuilder(model=model, model_dir=work_dir)
    t0 = time.perf_counter()
    if progress:
        if n_disk > 0 and model.disk and model.disk.enabled:
            progress.legacy_command("gendisk (+ diskdf if needed)", str(work_dir))
        if n_halo > 0 and model.halo and model.halo.enabled:
            progress.legacy_command("genhalo", str(work_dir))
        if n_bulge > 0 and model.bulge and model.bulge.enabled:
            progress.legacy_command("genbulge", str(work_dir))
    builder.sample(
        n_disk=n_disk if model.disk and model.disk.enabled else 0,
        n_halo=n_halo if model.halo and model.halo.enabled else 0,
        n_bulge=n_bulge if model.bulge and model.bulge.enabled else 0,
        work_dir=str(work_dir),
        cleanup=False,
        stream_output=stream,
    )
    counts = _validate_particle_files(work_dir)
    if progress:
        parts = [f"{name}={n:,}" for name, n in sorted(counts.items())]
        progress.log(f"  sampled {sum(counts.values()):,} particles ({', '.join(parts)})")
    return {
        "sample_seconds": time.perf_counter() - t0,
        "particles_dir": str(work_dir),
        "n_particles": sum(counts.values()),
        **{f"n_{k}": v for k, v in counts.items()},
    }


def _default_force_method() -> str:
    """Prefer compiled Barnes–Hut when the C extension is built."""
    try:
        from ntropy.forces.bhtree_c import extension_available

        if extension_available():
            return "bh_c"
    except ImportError:
        pass
    return "bh"


def _write_evolve_config(
    work_dir: Path,
    registry,
    *,
    end_time_gyr: float,
    diagnostics_every: int,
    particle_dump_every: int,
    mpi_ranks: int,
    force_method: str | None = None,
    dt_base: float = 0.025,
    timestep_eta: float = 0.025,
    max_timestep_bin: int = 6,
    timestep_update_every: int = 1,
    integrator_order: int = 2,
) -> Path:
    cfg_path = work_dir / "ntropy_config.json"
    method = force_method or _default_force_method()
    parallel = {
        "enabled": mpi_ranks > 1,
        "n_workers": max(1, mpi_ranks),
        "mode": "mpi",
    }
    cfg_data = {
        "particles": {"file": "merged.dat"},
        "particle_types": registry.to_config_dict(),
        "force": {"method": method, "theta": 0.5, "rebuild_every": 1},
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
    for name in ("disk", "halo", "bulge"):
        path = work_dir / name
        if path.is_file():
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


def _run_evolve(
    work_dir: Path,
    *,
    end_time_gyr: float = 1.0,
    diagnostics_every: int = 1,
    particle_dump_every: int = 50,
    mpi_ranks: int = 1,
    core_fraction: float = 0.75,
    dt_base: float = 0.025,
    timestep_eta: float = 0.025,
    max_timestep_bin: int = 6,
    timestep_update_every: int = 1,
    integrator_order: int = 2,
    progress: CampaignProgress | None = None,
    evolve_label: str = "evolve",
) -> dict:
    from galacticsics.sampling.particles import ParticleSet
    from ntropy.config import load_config
    from ntropy.integrations.galacticsics import merge_galacticsics_components
    from ntropy.particle_types import TypeRegistry
    from ntropy.simulation import Simulation

    registry = TypeRegistry.default_galaxy()
    particles: dict = {}
    for name in ("halo", "bulge", "disk"):
        path = work_dir / name
        if path.exists():
            particles[name] = ParticleSet.from_ascii(path, component=name)

    if not particles:
        raise FileNotFoundError(f"no particle files in {work_dir}")

    _validate_particle_files(work_dir)
    state = merge_galacticsics_components(particles, type_registry=registry)
    if state.n < 1:
        raise RuntimeError(f"merged particle state is empty in {work_dir}")

    use_mpi = _mpi_evolve_available(mpi_ranks)
    parallel_budget = resolve_parallel_budget(
        mpi_ranks if use_mpi else 1,
        core_fraction=core_fraction,
    )
    effective_mpi_ranks = parallel_budget.mpi_ranks if use_mpi else 1
    if progress and progress.enabled:
        if use_mpi and effective_mpi_ranks != mpi_ranks:
            progress.log(
                f"  MPI ranks clamped {mpi_ranks} → {effective_mpi_ranks} "
                f"(core budget {parallel_budget.budget_cores})"
            )
        progress.log(f"  parallel: {format_parallel_budget(parallel_budget)}")

    cfg_path = _write_evolve_config(
        work_dir,
        registry,
        end_time_gyr=end_time_gyr,
        diagnostics_every=diagnostics_every,
        particle_dump_every=particle_dump_every,
        mpi_ranks=effective_mpi_ranks,
        dt_base=dt_base,
        timestep_eta=timestep_eta,
        max_timestep_bin=max_timestep_bin,
        timestep_update_every=timestep_update_every,
        integrator_order=integrator_order,
    )
    state.write_ascii(work_dir / "merged.dat")
    _save_ic_state(state, work_dir)

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
    return {
        "evolve_seconds": elapsed,
        "dE_over_E0": dE,
        "n_energies": len(result.energies),
        "n_ranks": 1,
        "mpi": False,
    }


def run_campaign(
    spec: GridSpec,
    work_root: Path | str,
    *,
    stages: list[Stage] | None = None,
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
    mpi_ranks: int = 2,
    core_fraction: float = 0.75,
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
        Particle counts for sampling stage.
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
        MPI ranks for ntropy evolve (``2`` by default).  Falls back to
        serial when ``mpi4py`` or ``mpirun`` is unavailable.
    core_fraction : float
        Share of logical CPUs for MPI×OpenMP (default ``0.75``).
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
    reporter.banner(
        name=spec.name,
        work_root=str(work_root),
        stages=stages,
        n_models=len(models),
        extra={
            "particles": f"{n_disk:,} disk + {n_halo:,} halo",
            "end_time_gyr": end_time_gyr,
            "dt_base": dt_base,
            "mpi_ranks": mpi_ranks,
            "core_fraction": core_fraction,
            "diagnostics_every": diagnostics_every,
            "skip_done": skip_done,
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
                summary = _run_solve(model, model_dir, progress=reporter)
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
                reporter.skip_stage("sample", reason=".done_sample")
            else:
                if skip_done and marker.exists():
                    reporter.log(
                        "  sample marker present but particle files empty — re-running sample"
                    )
                reporter.start_stage(
                    "sample",
                    detail=f"n_disk={n_disk:,}, n_halo={n_halo:,}, n_bulge={n_bulge:,}",
                )
                t0 = time.perf_counter()
                summary = _run_sample(
                    model,
                    model_dir,
                    n_disk=n_disk,
                    n_halo=n_halo,
                    n_bulge=n_bulge,
                    progress=reporter,
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
                    dt_base=dt_base,
                    timestep_eta=timestep_eta,
                    max_timestep_bin=max_timestep_bin,
                    timestep_update_every=timestep_update_every,
                    integrator_order=integrator_order,
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
