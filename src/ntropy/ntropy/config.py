"""JSON run configuration schema and loader."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Union

from ntropy.particle_types import TypeRegistry
from ntropy.integrators.timestep import TimestepConfig
from ntropy.units import (
    DEFAULT_SIM_DT,
    DEFAULT_SIM_N_STEPS,
    format_simulation_duration,
    gyr_to_code_time,
)

PathLike = Union[str, Path]


@dataclass
class ParticlesConfig:
    file: str
    types_file: str | None = None
    default_type: str | None = None


@dataclass
class ParticleTypeEntry:
    """Deprecated config shim; use :class:`~ntropy.particle_types.ParticleTypeSpec`."""

    id: int
    eps: float = 0.01
    min_timestep_bin: int = 0
    max_timestep_bin: int | None = None


@dataclass
class SofteningConfig:
    default: float = 0.01
    per_particle: bool = False
    file: str | None = None


BhOptimPreset = Literal["legacy", "optimized"]
BhOmpSchedule = Literal["static", "guided", "dynamic"]

_BH_PRESET_LEGACY: dict[str, bool | str] = {
    "fast_inv_r3": False,
    "squared_opening": False,
    "iterative_walk": False,
    "morton_build": False,
    "borrow_arrays": False,
    "fast_coincident_check": False,
    "native_pack": False,
    "accel_all_fast": False,
    "simd_leaves": False,
    "omp_schedule": "static",
}
_BH_PRESET_OPTIMIZED: dict[str, bool | str] = {
    "fast_inv_r3": True,
    "squared_opening": True,
    "iterative_walk": True,
    "morton_build": True,
    "borrow_arrays": True,
    "fast_coincident_check": True,
    "native_pack": True,
    "accel_all_fast": True,
    "simd_leaves": False,
    "omp_schedule": "static",
}


@dataclass
class BhOptimizationsConfig:
    """
    Optional C Barnes–Hut kernel optimizations (``force.bh_optimizations``).

    ``preset="legacy"`` disables all optimizations (default, backwards compatible).
    ``preset="optimized"`` enables the safe physics-preserving fast paths.
    Individual flags override the preset when set explicitly.
    """

    preset: BhOptimPreset = "legacy"
    fast_inv_r3: bool | None = None
    squared_opening: bool | None = None
    iterative_walk: bool | None = None
    morton_build: bool | None = None
    borrow_arrays: bool | None = None
    fast_coincident_check: bool | None = None
    native_pack: bool | None = None
    accel_all_fast: bool | None = None
    simd_leaves: bool | None = None
    omp_schedule: BhOmpSchedule = "static"

    def resolve(self) -> dict[str, bool | str]:
        """Return effective flag values after applying preset and overrides."""
        base = dict(_BH_PRESET_LEGACY if self.preset == "legacy" else _BH_PRESET_OPTIMIZED)
        for key in (
            "fast_inv_r3",
            "squared_opening",
            "iterative_walk",
            "morton_build",
            "borrow_arrays",
            "fast_coincident_check",
            "native_pack",
            "accel_all_fast",
            "simd_leaves",
        ):
            val = getattr(self, key)
            if val is not None:
                base[key] = bool(val)
        base["omp_schedule"] = self.omp_schedule
        return base

    def to_c_opts(self) -> dict[str, int]:
        """Map resolved flags to integer options for the C extension."""
        r = self.resolve()
        sched = {"static": 0, "guided": 1, "dynamic": 2}[str(r["omp_schedule"])]
        return {
            "fast_inv_r3": int(bool(r["fast_inv_r3"])),
            "squared_opening": int(bool(r["squared_opening"])),
            "iterative_walk": int(bool(r["iterative_walk"])),
            "morton_build": int(bool(r["morton_build"])),
            "borrow_arrays": int(bool(r["borrow_arrays"])),
            "fast_coincident_check": int(bool(r["fast_coincident_check"])),
            "native_pack": int(bool(r["native_pack"])),
            "accel_all_fast": int(bool(r["accel_all_fast"])),
            "simd_leaves": int(bool(r["simd_leaves"])),
            "omp_schedule": sched,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> BhOptimizationsConfig:
        if not raw:
            return cls()
        preset = raw.get("preset", "legacy")
        if preset not in ("legacy", "optimized"):
            raise ValueError(
                f"force.bh_optimizations.preset must be 'legacy' or 'optimized', got {preset!r}"
            )
        sched = raw.get("omp_schedule", "static")
        if sched not in ("static", "guided", "dynamic"):
            raise ValueError(
                f"force.bh_optimizations.omp_schedule must be static|guided|dynamic, got {sched!r}"
            )
        kwargs: dict[str, Any] = {"preset": preset, "omp_schedule": sched}
        for key in (
            "fast_inv_r3",
            "squared_opening",
            "iterative_walk",
            "morton_build",
            "borrow_arrays",
            "fast_coincident_check",
            "native_pack",
            "accel_all_fast",
            "simd_leaves",
        ):
            if key in raw:
                kwargs[key] = bool(raw[key])
        return cls(**kwargs)

    def to_config_dict(self) -> dict[str, Any]:
        """Serialize for ``ntropy_config.json`` (preset + explicit overrides)."""
        out: dict[str, Any] = {"preset": self.preset}
        if self.omp_schedule != "static":
            out["omp_schedule"] = self.omp_schedule
        for key in (
            "fast_inv_r3",
            "squared_opening",
            "iterative_walk",
            "morton_build",
            "borrow_arrays",
            "fast_coincident_check",
            "native_pack",
            "accel_all_fast",
            "simd_leaves",
        ):
            val = getattr(self, key)
            if val is not None:
                out[key] = bool(val)
        return out

    @classmethod
    def from_preset(cls, preset: BhOptimPreset) -> BhOptimizationsConfig:
        return cls(preset=preset)


@dataclass
class ForceConfig:
    method: Literal["brute", "bh", "bh_c"] = "bh"
    theta: float = 0.5
    rebuild_every: int = 1
    active_subset: bool = True
    bh_optimizations: BhOptimizationsConfig = field(default_factory=BhOptimizationsConfig)


IntegratorType = Literal[
    "leapfrog", "euler", "rk2", "rk3", "rk4", "tiered_leapfrog"
]


@dataclass
class IntegratorConfig:
    type: IntegratorType = "leapfrog"
    order: Literal[1, 2] = 2
    dt: float = DEFAULT_SIM_DT
    n_steps: int = DEFAULT_SIM_N_STEPS
    dt_base: float | None = None
    end_time_gyr: float | None = None
    timestep: TimestepConfig = field(default_factory=TimestepConfig)


@dataclass
class ParallelConfig:
    enabled: bool = False
    n_workers: int = 1
    mode: Literal["mpi", "domains"] = "mpi"


@dataclass
class OutputConfig:
    dir: str = "run_output"
    every: int = 0
    write_final: bool = True
    diagnostics_every: int = 1
    particle_dump_every: int = 0
    write_particle_bins: bool = True


@dataclass
class AnalysisConfig:
    density_bins: int = 20
    r_max: float | None = None


@dataclass
class RunConfig:
    seed: int = 42
    particles: ParticlesConfig = field(default_factory=lambda: ParticlesConfig(file="particles.dat"))
    softening: SofteningConfig = field(default_factory=SofteningConfig)
    force: ForceConfig = field(default_factory=ForceConfig)
    integrator: IntegratorConfig = field(default_factory=IntegratorConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    particle_types: TypeRegistry | None = None
    base_dir: Path = field(default_factory=Path.cwd)

    def resolve_path(self, relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute():
            return path
        return self.base_dir / path


def format_run_config(cfg: RunConfig, *, label: str | None = None) -> str:
    """
    Human-readable summary of a run configuration.

    Parameters
    ----------
    cfg : RunConfig
        Run configuration.
    label : str, optional
        Short name for this run (shown as a header).

    Returns
    -------
    text : str
        Multi-line summary suitable for logging or notebook output.
    """
    integ = cfg.integrator
    order_note = ""
    if integ.type in ("leapfrog", "tiered_leapfrog"):
        order_note = f", order={integ.order}"
    parallel = "on" if cfg.parallel.enabled else "off"
    if integ.type == "tiered_leapfrog":
        dt_base = integ.dt_base if integ.dt_base is not None else integ.timestep.dt_base
        end_gyr = integ.end_time_gyr if integ.end_time_gyr is not None else 1.0
        t_end_code = gyr_to_code_time(end_gyr)
        n_fine = max(1, int(round(t_end_code / dt_base)))
        duration = (
            f"{end_gyr:.4g} Gyr = {t_end_code:.4g} code units "
            f"in {n_fine} fine substeps (dt_base={dt_base:.5f})"
        )
        lines = [
            f"=== {label} ===" if label else "=== ntropy run ===",
            f"  integrator : {integ.type}{order_note}",
            f"  dt_base    : {dt_base:.5f} code units",
            f"  end_time   : {end_gyr:.4g} Gyr",
            f"  n_substeps : {n_fine} (fine, global clock)",
            f"  duration   : {duration}",
            f"  force      : {cfg.force.method} (theta={cfg.force.theta})",
            f"  parallel   : {parallel}",
            f"  seed       : {cfg.seed}",
        ]
    else:
        duration = format_simulation_duration(integ.dt, integ.n_steps)
        lines = [
            f"=== {label} ===" if label else "=== ntropy run ===",
            f"  integrator : {integ.type}{order_note}",
            f"  dt         : {integ.dt:.5f} code units",
            f"  n_steps    : {integ.n_steps}",
            f"  duration   : {duration}",
            f"  force      : {cfg.force.method} (theta={cfg.force.theta})",
            f"  parallel   : {parallel}",
            f"  seed       : {cfg.seed}",
        ]
    return "\n".join(lines)


def _require(data: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in data:
        raise ValueError(f"Missing required key '{key}' in {ctx}")
    return data[key]


def _validate_positive(name: str, value: float, *, allow_zero: bool = False) -> float:
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    elif value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return float(value)


def load_config(path: PathLike) -> RunConfig:
    """
    Load and validate a JSON run configuration.

    Parameters
    ----------
    path : path-like
        Path to the JSON configuration file.  Relative particle paths are
        resolved against this file's directory.

    Returns
    -------
    RunConfig
        Validated configuration object.

    Raises
    ------
    ValueError
        On missing keys, invalid enum values, or non-positive numerics.
    FileNotFoundError
        If the config file does not exist.
    """
    config_path = Path(path).resolve()
    with open(config_path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a JSON object")

    particles_raw = _require(raw, "particles", "config")
    if "file" not in particles_raw:
        raise ValueError("particles.file is required")

    soft_raw = raw.get("softening", {})
    force_raw = raw.get("force", {})
    integ_raw = raw.get("integrator", {})
    par_raw = raw.get("parallel", {})
    out_raw = raw.get("output", {})
    ana_raw = raw.get("analysis", {})

    method = force_raw.get("method", "bh")
    if method not in ("brute", "bh", "bh_c"):
        raise ValueError(
            f"force.method must be 'brute', 'bh', or 'bh_c', got {method!r}"
        )

    integ_type = integ_raw.get("type", "leapfrog")
    valid_types = ("leapfrog", "euler", "rk2", "rk3", "rk4", "tiered_leapfrog")
    if integ_type not in valid_types:
        raise ValueError(
            f"integrator.type must be one of {valid_types}, got {integ_type!r}"
        )

    integ_order = int(integ_raw.get("order", 2))
    if integ_type == "leapfrog" or integ_type == "tiered_leapfrog":
        if integ_order not in (1, 2):
            raise ValueError(
                f"integrator.order must be 1 or 2 for {integ_type}, got {integ_order}"
            )
    elif integ_order != 2:
        raise ValueError(
            f"integrator.order is only used for leapfrog; got order={integ_order} "
            f"with type={integ_type!r}"
        )

    par_mode = par_raw.get("mode", "mpi")
    if par_mode not in ("mpi", "domains"):
        raise ValueError(f"parallel.mode must be 'mpi' or 'domains', got {par_mode!r}")

    n_workers = int(par_raw.get("n_workers", 1))
    if n_workers < 1:
        raise ValueError(f"parallel.n_workers must be >= 1, got {n_workers}")

    types_raw = raw.get("particle_types")
    particle_types = (
        TypeRegistry.from_config_dict(types_raw) if types_raw is not None else None
    )

    rebuild_every = int(force_raw.get("rebuild_every", 1))
    if rebuild_every < 1:
        raise ValueError(f"force.rebuild_every must be >= 1, got {rebuild_every}")

    dt_base_raw = integ_raw.get("dt_base")
    end_gyr_raw = integ_raw.get("end_time_gyr")
    ts_raw = integ_raw.get("timestep", {})
    dt_base_val = float(dt_base_raw) if dt_base_raw is not None else float(
        integ_raw.get("dt", DEFAULT_SIM_DT)
    )
    ts_config = TimestepConfig(
        eta=float(ts_raw.get("eta", 0.025)),
        dt_base=float(ts_raw.get("dt_base", dt_base_val)),
        max_bin=int(ts_raw.get("max_bin", 6)),
        update_every=max(1, int(ts_raw.get("update_every", 1))),
        accel_floor=float(ts_raw.get("accel_floor", 1e-6)),
    )

    return RunConfig(
        seed=int(raw.get("seed", 42)),
        particles=ParticlesConfig(
            file=str(particles_raw["file"]),
            types_file=particles_raw.get("types_file"),
            default_type=particles_raw.get("default_type"),
        ),
        softening=SofteningConfig(
            default=_validate_positive("softening.default", float(soft_raw.get("default", 0.01))),
            per_particle=bool(soft_raw.get("per_particle", False)),
            file=soft_raw.get("file"),
        ),
        force=ForceConfig(
            method=method,
            theta=_validate_positive("force.theta", float(force_raw.get("theta", 0.5))),
            rebuild_every=rebuild_every,
            active_subset=bool(force_raw.get("active_subset", True)),
            bh_optimizations=BhOptimizationsConfig.from_dict(
                force_raw.get("bh_optimizations")
            ),
        ),
        integrator=IntegratorConfig(
            type=integ_type,  # type: ignore[arg-type]
            order=integ_order,  # type: ignore[arg-type]
            dt=_validate_positive("integrator.dt", float(integ_raw.get("dt", DEFAULT_SIM_DT))),
            n_steps=int(
                _validate_positive(
                    "integrator.n_steps",
                    float(integ_raw.get("n_steps", DEFAULT_SIM_N_STEPS)),
                    allow_zero=False,
                )
            ),
            dt_base=float(dt_base_raw) if dt_base_raw is not None else None,
            end_time_gyr=float(end_gyr_raw) if end_gyr_raw is not None else None,
            timestep=ts_config,
        ),
        parallel=ParallelConfig(
            enabled=bool(par_raw.get("enabled", False)),
            n_workers=n_workers,
            mode=par_mode,
        ),
        output=OutputConfig(
            dir=str(out_raw.get("dir", "run_output")),
            every=max(0, int(out_raw.get("every", 0))),
            write_final=bool(out_raw.get("write_final", True)),
            diagnostics_every=max(0, int(out_raw.get("diagnostics_every", 1))),
            particle_dump_every=max(0, int(out_raw.get("particle_dump_every", 0))),
            write_particle_bins=bool(out_raw.get("write_particle_bins", True)),
        ),
        analysis=AnalysisConfig(
            density_bins=max(1, int(ana_raw.get("density_bins", 20))),
            r_max=float(ana_raw["r_max"]) if "r_max" in ana_raw else None,
        ),
        particle_types=particle_types,
        base_dir=config_path.parent,
    )
