"""Load notebook / campaign walkthrough settings from a single JSON file."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from galacticsics.campaign.parallel_budget import (
    apply_parallel_env,
    format_parallel_budget,
    resolve_parallel_budget,
)
from galacticsics.campaign.spec import (
    GridSpec,
    expand_grid,
    format_dbh_grid_summary,
    model_patch_defaults,
    preview_dbh_model,
)
from galacticsics.campaign.timestep_summary import format_tiered_timestep_summary, summarize_tiered_timestep
from galacticsics.models import GalaxyModel

# Standard Milky-Way-like components (10× legacy ntropy ε defaults).
DEFAULT_COMPONENT_PARTICLES: dict[str, int] = {
    "disk": 100_000,
    "halo": 100_000,
    "bulge": 0,
}
DEFAULT_COMPONENT_SOFTENING: dict[str, float] = {
    "disk": 0.1,
    "halo": 0.5,
    "bulge": 0.2,
}
_LEGACY_PARTICLE_KEYS: dict[str, str] = {
    "n_disk": "disk",
    "n_halo": "halo",
    "n_bulge": "bulge",
}


def default_walkthrough_config() -> dict[str, Any]:
    """
    Full walkthrough defaults (notebook source of truth).

    Edit the returned dict in the notebook; JSON on disk is optional export only.
    """
    return {
        "name": "mw_walkthrough",
        "artifacts": {
            "root": "notebooks/artifacts/campaign_walkthrough",
        },
        "run": {
            "skip_done": False,
            "verbose": True,
            "stages": ["solve", "sample", "evolve"],
            "reevolve_checkpoints": False,
        },
        "particles": dict(DEFAULT_COMPONENT_PARTICLES),
        "softening": dict(DEFAULT_COMPONENT_SOFTENING),
        "parallel": {
            "mpi_ranks": 1,
            "core_fraction": 0.75,
        },
        "solve": {
            "npsi": 1000,
            "nint": 20,
            "max_iter": 100,
            "n_workers": 0,
            "diskdf_backend": "python",
        },
        "sample": {
            "use_openmp": True,
            "n_openmp_threads": 0,
            "run_diskdf": True,
        },
        "base_model": {
            "grid": {
                "name": "base_mw",
                "base": "milky_way_disk_halo",
                "axes": {},
                "omit_components": [[]],
                "coarse_grid": True,
                "physics_backend": "python",
            },
            "patch": _default_base_model_patch(),
        },
        "sweep": {
            "enabled": False,
            "grid_spec": "campaigns/mw_grid.json",
        },
        "evolve": {
            "end_time_gyr": 0.5,
            "dt_base": 0.05,
            "timestep_eta": 0.025,
            "max_timestep_bin": 5,
            "timestep_update_every": 2,
            "integrator_order": 2,
            "diagnostics_every": 20,
            "particle_dump_every": 100,
        },
        "force": {
            "theta": 0.6,
            "rebuild_every": 10,
            "active_subset": True,
            "bh_optimizations_preset": "optimized",
            "bh_optimizations": {
                "preset": "optimized",
                "simd_leaves": True,
                "omp_schedule": "guided",
            },
            # Gadget-style local octrees + LET under MPI (default on).
            "mpi_local_trees": True,
        },
        "checkpoints_gyr": [0.0, 0.05, 0.1],
    }


def merge_walkthrough_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Deep-copy defaults and apply nested ``overrides``."""
    cfg = copy.deepcopy(default_walkthrough_config())
    if not overrides:
        return cfg

    def _merge(base: dict, patch: dict) -> dict:
        for key, val in patch.items():
            if isinstance(val, dict) and isinstance(base.get(key), dict):
                base[key] = _merge(dict(base[key]), val)
            else:
                base[key] = val
        return base

    return _merge(cfg, overrides)


def _default_base_model_patch() -> dict[str, Any]:
    """Physical component defaults for the walkthrough base model + Toomre target."""
    patch = model_patch_defaults("milky_way_disk_halo")
    patch["disk_kinematics.toomre_q_target"] = 1.5
    return patch


def _normalize_component_block(
    block: dict[str, Any],
    *,
    defaults: dict[str, Any],
    legacy_map: dict[str, str],
    coerce,
) -> dict[str, Any]:
    """Parse a config block keyed by component label (with legacy aliases)."""
    out = dict(defaults)
    for key, val in block.items():
        label = legacy_map.get(key, key)
        if key.endswith("_kpc") and key[:-4] not in legacy_map:
            label = key[:-4]
        out[label] = coerce(val)
    return out


def particles_by_component(raw: dict[str, Any]) -> dict[str, int]:
    """Map config ``particles`` block → component label → count."""
    return {
        k: int(v)
        for k, v in _normalize_component_block(
            raw.get("particles", {}),
            defaults=DEFAULT_COMPONENT_PARTICLES,
            legacy_map=_LEGACY_PARTICLE_KEYS,
            coerce=int,
        ).items()
    }


def sample_config_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    """Keyword args for :class:`~galacticsics.sampling.sampler.SampleConfig`."""
    block = raw.get("sample", {})
    defaults = default_walkthrough_config()["sample"]
    return {
        "use_openmp": bool(block.get("use_openmp", defaults["use_openmp"])),
        "n_openmp_threads": int(block.get("n_openmp_threads", defaults["n_openmp_threads"])),
        "run_diskdf": bool(block.get("run_diskdf", defaults["run_diskdf"])),
    }


def sample_openmp_summary(raw: dict[str, Any]) -> str:
    """One-line status for notebook logs and :meth:`WalkthroughConfig.summary_lines`."""
    from galacticsics.sampling.openmp import openmp_sampler_status
    from galacticsics.sampling.sampler import SampleConfig

    sk = sample_config_kwargs(raw)
    status = openmp_sampler_status(
        SampleConfig(n_disk=0, n_halo=0, use_openmp=sk["use_openmp"], n_openmp_threads=sk["n_openmp_threads"])
    )
    if status.is_openmp:
        return f"Sample: OpenMP C samplers ({status.threads_label})"
    return f"Sample: Python fallback ({status.reason})"


def softening_eps_by_component(raw: dict[str, Any]) -> dict[str, float]:
    """Map config ``softening`` block → component label → ε [kpc]."""
    return {
        k: float(v)
        for k, v in _normalize_component_block(
            raw.get("softening", {}),
            defaults=DEFAULT_COMPONENT_SOFTENING,
            legacy_map=_LEGACY_PARTICLE_KEYS,
            coerce=float,
        ).items()
    }


def build_type_registry(raw: dict[str, Any]):
    """
    ntropy :class:`TypeRegistry` from config.

    Starts from :meth:`TypeRegistry.default_galaxy` (disk / bulge / halo) and
    applies per-component softening.  Extra labels in ``particles``,
    ``softening``, or an explicit ``particle_types`` block are registered
    automatically.
    """
    from ntropy.particle_types import ParticleTypeSpec, TypeRegistry

    eps = softening_eps_by_component(raw)
    particles = particles_by_component(raw)

    if raw.get("particle_types"):
        reg = TypeRegistry.from_config_dict(raw["particle_types"])
        types = dict(reg.types)
        for label, spec in list(types.items()):
            if label in eps:
                types[label] = replace(spec, eps=eps[label])
    else:
        reg = TypeRegistry.default_galaxy()
        types = {
            label: replace(spec, eps=eps.get(label, spec.eps))
            for label, spec in reg.types.items()
        }

    extra = (set(particles) | set(eps)) - set(types)
    next_id = max((s.id for s in types.values()), default=0) + 1
    for label in sorted(extra):
        types[label] = ParticleTypeSpec(
            id=next_id,
            label=label,
            eps=eps.get(label, 0.05),
            min_timestep_bin=0,
            max_timestep_bin=5,
        )
        next_id += 1

    for label, eps_val in eps.items():
        if label in types:
            types[label] = replace(types[label], eps=eps_val)

    return TypeRegistry(types=types)


def config_parameter_table(raw: dict[str, Any]) -> "pd.DataFrame":
    """Flat parameter listing for notebook display."""
    import pandas as pd

    rows: list[dict[str, Any]] = []

    def _walk(prefix: str, obj: Any) -> None:
        if isinstance(obj, dict):
            if not obj:
                rows.append({"parameter": prefix, "value": {}})
                return
            for key, val in obj.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                _walk(path, val)
        elif isinstance(obj, list):
            rows.append({"parameter": prefix, "value": obj})
        else:
            rows.append({"parameter": prefix, "value": obj})

    _walk("", raw)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("parameter").reset_index(drop=True)


def _repo_root(start: Path | None = None) -> Path:
    root = (start or Path.cwd()).resolve()
    for candidate in (root, *root.parents):
        if (candidate / "src" / "galacticsics").is_dir():
            return candidate
    return root


def _resolve_path(path: str | Path, *, repo: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (repo / p).resolve()


def _patch_model(model: GalaxyModel, patch: dict[str, Any]) -> GalaxyModel:
    from galacticsics.campaign.spec import _set_nested

    for key, value in patch.items():
        model = _set_nested(model, key, value)
    return model


@dataclass
class WalkthroughPaths:
    """Resolved artifact directories."""

    repo: Path
    artifacts: Path
    base_root: Path
    sweep_root: Path
    config_path: Path


@dataclass
class WalkthroughConfig:
    """End-to-end campaign settings loaded from JSON."""

    name: str
    paths: WalkthroughPaths
    run: dict[str, Any]
    particles: dict[str, int]
    parallel: dict[str, Any]
    evolve: dict[str, Any]
    force: dict[str, Any]
    base_grid: GridSpec
    base_patch: dict[str, Any]
    sweep_enabled: bool
    sweep_grid_path: Path | None
    sweep_grid_paths: list[Path] = field(default_factory=list)
    checkpoints_gyr: list[float] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path, *, repo: Path | None = None) -> WalkthroughConfig:
        config_path = Path(path).resolve()
        raw = json.loads(config_path.read_text())
        return cls.from_raw(raw, config_path=config_path, repo=repo)

    @classmethod
    def from_raw(
        cls,
        raw: dict[str, Any],
        *,
        config_path: Path,
        repo: Path | None = None,
    ) -> WalkthroughConfig:
        config_path = Path(config_path).resolve()
        repo = repo or _repo_root(config_path.parent)

        art = raw.get("artifacts", {})
        artifacts = _resolve_path(art.get("root", "notebooks/artifacts/campaign_walkthrough"), repo=repo)

        base_raw = raw.get("base_model", {})
        grid_raw = dict(base_raw.get("grid", {}))
        grid_raw.setdefault("name", "base_mw")
        grid_raw["patch"] = dict(base_raw.get("patch", {}))
        base_grid = GridSpec(**grid_raw)

        sweep = raw.get("sweep", {})
        sweep_path = sweep.get("grid_spec")
        sweep_paths_raw = sweep.get("grid_specs") or []
        if isinstance(sweep_paths_raw, str):
            sweep_paths_raw = [sweep_paths_raw]
        sweep_paths = [
            _resolve_path(p, repo=repo) for p in sweep_paths_raw if p
        ]
        if sweep_path and not sweep_paths:
            sweep_paths = [_resolve_path(sweep_path, repo=repo)]
        return cls(
            name=raw.get("name", "walkthrough"),
            paths=WalkthroughPaths(
                repo=repo,
                artifacts=artifacts,
                base_root=artifacts / "base_mw",
                sweep_root=artifacts / "runs",
                config_path=config_path,
            ),
            run=raw.get("run", {}),
            particles=raw.get("particles", {}),
            parallel=raw.get("parallel", {}),
            evolve=raw.get("evolve", {}),
            force=raw.get("force", {}),
            base_grid=base_grid,
            base_patch=dict(base_raw.get("patch", {})),
            sweep_enabled=bool(sweep.get("enabled", False)),
            sweep_grid_path=sweep_paths[0] if sweep_paths else None,
            sweep_grid_paths=sweep_paths,
            checkpoints_gyr=list(raw.get("checkpoints_gyr", [])),
            raw=raw,
        )

    def ensure_artifact_dirs(self) -> None:
        self.paths.artifacts.mkdir(parents=True, exist_ok=True)

    def apply_parallel_env(self) -> Any:
        budget = resolve_parallel_budget(
            int(self.parallel.get("mpi_ranks", 1)),
            core_fraction=float(self.parallel.get("core_fraction", 0.75)),
        )
        apply_parallel_env(budget)
        return budget

    def base_model(self) -> tuple[str, GalaxyModel]:
        return expand_grid(self.base_grid)[0]

    def sweep_grid(self) -> GridSpec | None:
        """
        Load the sweep grid.

        When ``sweep.grid_specs`` lists multiple suite files, expand and merge
        them into one ``grid_mode='list'`` spec (de-duplicated by model hash).
        """
        if not self.sweep_enabled:
            return None
        paths = list(self.sweep_grid_paths)
        if not paths and self.sweep_grid_path is not None:
            paths = [self.sweep_grid_path]
        if not paths:
            return None
        from galacticsics.campaign.spec import (
            expand_grids,
            grids_as_list_spec,
            load_grid_spec,
        )

        specs = [load_grid_spec(p) for p in paths]
        if len(specs) == 1:
            return specs[0]
        pairs = expand_grids(specs)
        backend = specs[0].physics_backend
        coarse = any(s.coarse_grid for s in specs)
        return grids_as_list_spec(
            self.name,
            pairs,
            base=specs[0].base,
            coarse_grid=coarse,
            physics_backend=backend,
        )

    def run_campaign_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for :func:`~galacticsics.campaign.runner.run_campaign`."""
        ev = self.evolve
        fc = self.force
        bh_extra = dict(fc.get("bh_optimizations", {}))
        preset = fc.get("bh_optimizations_preset", bh_extra.get("preset", "optimized"))
        bh_extra.setdefault("preset", preset)
        pbc = particles_by_component(self.raw)
        return {
            "stages": list(self.run.get("stages", ["solve", "sample", "evolve"])),
            "particles_by_component": pbc,
            "n_disk": pbc.get("disk", 0),
            "n_halo": pbc.get("halo", 0),
            "n_bulge": pbc.get("bulge", 0),
            "end_time_gyr": float(ev.get("end_time_gyr", 1.0)),
            "dt_base": float(ev.get("dt_base", 0.025)),
            "timestep_eta": float(ev.get("timestep_eta", 0.025)),
            "max_timestep_bin": int(ev.get("max_timestep_bin", 6)),
            "timestep_update_every": int(ev.get("timestep_update_every", 1)),
            "integrator_order": int(ev.get("integrator_order", 2)),
            "diagnostics_every": int(ev.get("diagnostics_every", 1)),
            "particle_dump_every": int(ev.get("particle_dump_every", 50)),
            "skip_done": bool(self.run.get("skip_done", True)),
            "verbose": bool(self.run.get("verbose", True)),
            "mpi_ranks": int(self.parallel.get("mpi_ranks", 1)),
            "core_fraction": float(self.parallel.get("core_fraction", 0.75)),
            "force_rebuild_every": int(fc.get("rebuild_every", 10)),
            "force_theta": float(fc.get("theta", 0.6)),
            "force_active_subset": bool(fc.get("active_subset", True)),
            "mpi_local_trees": bool(fc.get("mpi_local_trees", True)),
            "bh_optimizations_preset": preset,
            "bh_optimizations_extra": bh_extra,
            "eps_by_component": softening_eps_by_component(self.raw),
            "raw_config": self.raw,
            "force_method": fc.get("method"),
            "gpu_batch_size": int(self.run.get("gpu_batch_size", 1)),
            "dump_float32": bool(ev.get("dump_float32", False)),
            "solve_kwargs": {
                k: v
                for k, v in {
                    "npsi": int(self.raw.get("solve", {}).get("npsi", 1000)),
                    "nint": int(self.raw.get("solve", {}).get("nint", 20)),
                    "max_iter": int(self.raw.get("solve", {}).get("max_iter", 100)),
                    "n_workers": int(self.raw.get("solve", {}).get("n_workers", 0)),
                }.items()
            },
        }

    def summary_lines(self) -> list[str]:
        kw = self.run_campaign_kwargs()
        ts = summarize_tiered_timestep(
            dt_base=kw["dt_base"],
            end_time_gyr=kw["end_time_gyr"],
            max_bin=kw["max_timestep_bin"],
            eta=kw["timestep_eta"],
            update_every=kw["timestep_update_every"],
            integrator_order=kw["integrator_order"],
        )
        preview = preview_dbh_model(
            base=self.base_grid.base,
            coarse=self.base_grid.coarse_grid,
            dr=self.base_grid.grid_dr,
            nr=self.base_grid.grid_nr,
            lmax=self.base_grid.grid_lmax,
        )
        budget = resolve_parallel_budget(kw["mpi_ranks"], core_fraction=kw["core_fraction"])
        pbc = kw["particles_by_component"]
        eps = kw["eps_by_component"]
        parts = ", ".join(f"{k}={v:,}" for k, v in sorted(pbc.items()) if v)
        eps_line = ", ".join(f"{k}={v}" for k, v in sorted(eps.items()))
        lines = [
            f"Config: {self.paths.config_path.name}",
            f"Particles: {parts or 'none'} (total {sum(pbc.values()):,})",
            f"Softening ε [kpc]: {eps_line}",
            f"Evolve: {kw['end_time_gyr']} Gyr | verbose: {kw['verbose']}",
            format_tiered_timestep_summary(ts),
            format_dbh_grid_summary(preview),
            f"Force: theta={kw['force_theta']}, rebuild_every={kw['force_rebuild_every']}, "
            f"active_subset={kw['force_active_subset']}, "
            f"mpi_local_trees={kw['mpi_local_trees']}, bh={kw['bh_optimizations_preset']}",
            f"Parallel: {format_parallel_budget(budget)}",
            sample_openmp_summary(self.raw),
            f"Artifacts → {self.paths.artifacts}",
        ]
        if self.base_patch:
            lines.append(f"Model patch: {self.base_patch}")
        return lines


def load_walkthrough_config(path: str | Path, *, repo: Path | None = None) -> WalkthroughConfig:
    """Load :class:`WalkthroughConfig` from ``path``."""
    return WalkthroughConfig.load(path, repo=repo)


CONFIG_FIELD_DOCS: dict[str, str] = {
    "artifacts.root": "Output directory for all notebook artifacts (gitignored).",
    "run.skip_done": "Skip stages with .done_{stage} markers when true.",
    "run.verbose": "Print timestamped campaign logs and ntropy progress.",
    "run.stages": "Pipeline stages: solve (dbh), sample (ICs), evolve (ntropy).",
    "run.reevolve_checkpoints": "Re-run ntropy for intermediate times (expensive).",
    "particles": "Per-component IC counts keyed by label (default disk/halo/bulge).",
    "particles.disk": "Stellar disk particle count for sample stage.",
    "particles.halo": "Halo particle count for sample stage.",
    "particles.bulge": "Bulge particle count (0 to omit).",
    "softening": "Per-component Plummer ε [kpc] keyed by label (default disk/halo/bulge).",
    "softening.disk": "Disk ε [kpc] (default 0.1 = 10× legacy 0.01).",
    "softening.halo": "Halo ε [kpc] (default 0.5 = 10× legacy 0.05).",
    "softening.bulge": "Bulge ε [kpc] (default 0.2 = 10× legacy 0.02).",
    "particle_types": "Optional explicit ntropy type registry (id, eps, bin limits per label).",
    "parallel.mpi_ranks": "MPI ranks for evolve (1 = serial OpenMP on one node).",
    "parallel.core_fraction": "Share of logical CPUs for MPI×OpenMP (0.75 = leave headroom).",
    "sample.use_openmp": "Use OpenMP C samplers for disk/halo when extension is built (default true).",
    "sample.n_openmp_threads": "OpenMP threads for IC sampling (0 = runtime default / all cores).",
    "sample.run_diskdf": "Run diskdf before gendisk when cordbh.dat is missing or stale.",
    "base_model.grid.base": "Factory name: milky_way_disk_halo or reference_disk_halo.",
    "base_model.grid.coarse_grid": "Cap nr≤4000, lmax≤4 for fast laptop screening.",
    "base_model.grid.grid_dr": "DBH Poisson radial step [kpc].",
    "base_model.grid.grid_nr": "DBH radial bins (cost ~ nr × lmax²).",
    "base_model.grid.grid_lmax": "Multipole order for self-consistent potential.",
    "base_model.grid.physics_backend": "python (default) or legacy Fortran subprocesses.",
    "base_model.patch": (
        "Dot-path overrides to :class:`~galacticsics.models.GalaxyModel` components "
        "(masses, scale lengths, halo v0/a, disk kinematics, etc.). "
        "Lengths [kpc], velocities [100 km/s], masses [GalactICS units]."
    ),
    "base_model.patch.disk.mass": "Stellar disk mass [GalactICS units; MW default 17].",
    "base_model.patch.disk.scale_length": "Disk scale length R_d [kpc].",
    "base_model.patch.disk.scale_height": "Disk scale height z_d [kpc].",
    "base_model.patch.halo.v0": "Halo characteristic velocity [100 km/s].",
    "base_model.patch.halo.a": "NFW halo scale radius [kpc].",
    "base_model.patch.disk_kinematics.toomre_q_target": "Target Toomre Q at 2.5 R_d (scales sigma_r0).",
    "sweep.enabled": "Run factorial grid from sweep.grid_spec after base case.",
    "sweep.grid_spec": "Path to GridSpec JSON (see campaigns/mw_grid.json).",
    "sweep.grid_specs": "Optional list of GridSpec JSON paths merged into one corpus.",
    "evolve.end_time_gyr": "N-body duration [Gyr].",
    "evolve.dt_base": "Finest tiered substep [code units]; 1 unit ≈ 9.78 Myr.",
    "evolve.timestep_eta": "η in Δt ∝ √(ε/|a|) for per-particle bin assignment.",
    "evolve.max_timestep_bin": "Coarsest bin b (Δt = dt_base × 2^b).",
    "evolve.timestep_update_every": "Recompute timestep bins every N fine substeps.",
    "evolve.integrator_order": "1 = symplectic Euler, 2 = velocity Verlet.",
    "evolve.diagnostics_every": "Write tiered diagnostics every N fine substeps.",
    "evolve.particle_dump_every": "Write per-particle .npz dumps every N recorded steps.",
    "force.theta": "Barnes–Hut opening angle (0.5 accurate, 0.6–0.7 faster).",
    "force.rebuild_every": "Rebuild tree every N substeps (10 = less build overhead).",
    "force.active_subset": "Evaluate forces only on active tiered particles (faster).",
    "force.mpi_local_trees": (
        "Gadget-style local octrees + LET under MPI (default True); "
        "False = full replicated tree per rank."
    ),
    "force.bh_optimizations_preset": "legacy or optimized C kernel preset.",
    "force.bh_optimizations.simd_leaves": "4-wide unrolled leaf loop in bh_c.",
    "force.bh_optimizations.omp_schedule": "OpenMP schedule: static, guided, or dynamic.",
    "checkpoints_gyr": "Times for optional checkpoint analysis [Gyr].",
}


def config_field_docs_table() -> str:
    """Markdown table of all documented JSON keys."""
    lines = ["| Key | Description |", "| --- | --- |"]
    for key, doc in CONFIG_FIELD_DOCS.items():
        lines.append(f"| `{key}` | {doc} |")
    return "\n".join(lines)


def save_walkthrough_config(path: str | Path, raw: dict[str, Any]) -> Path:
    """Write walkthrough JSON and return the path."""
    path = Path(path)
    path.write_text(json.dumps(raw, indent=2) + "\n")
    return path


def reload_walkthrough_context(
    path: str | Path | None = None,
    *,
    raw: dict[str, Any] | None = None,
    repo: Path | None = None,
) -> tuple[WalkthroughConfig, Any]:
    """
    Reload config after editing JSON on disk or an in-memory ``CONFIG`` dict.

    Pass ``raw=CONFIG`` to apply notebook edits without saving first.
    """
    config_path = Path(path) if path is not None else Path("mw_walkthrough.json")
    if raw is not None:
        cfg = WalkthroughConfig.from_raw(raw, config_path=config_path, repo=repo)
    else:
        cfg = load_walkthrough_config(config_path, repo=repo)
    cfg.ensure_artifact_dirs()
    budget = cfg.apply_parallel_env()
    return cfg, budget
