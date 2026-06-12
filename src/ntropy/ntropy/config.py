"""JSON run configuration schema and loader."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Union

PathLike = Union[str, Path]


@dataclass
class ParticlesConfig:
    file: str


@dataclass
class SofteningConfig:
    default: float = 0.01
    per_particle: bool = False
    file: str | None = None


@dataclass
class ForceConfig:
    method: Literal["brute", "bh"] = "bh"
    theta: float = 0.5


IntegratorType = Literal["leapfrog", "euler", "rk2", "rk3", "rk4"]


@dataclass
class IntegratorConfig:
    type: IntegratorType = "leapfrog"
    order: Literal[1, 2] = 2
    dt: float = 0.01
    n_steps: int = 100


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
    base_dir: Path = field(default_factory=Path.cwd)

    def resolve_path(self, relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute():
            return path
        return self.base_dir / path


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
    if method not in ("brute", "bh"):
        raise ValueError(f"force.method must be 'brute' or 'bh', got {method!r}")

    integ_type = integ_raw.get("type", "leapfrog")
    valid_types = ("leapfrog", "euler", "rk2", "rk3", "rk4")
    if integ_type not in valid_types:
        raise ValueError(
            f"integrator.type must be one of {valid_types}, got {integ_type!r}"
        )

    integ_order = int(integ_raw.get("order", 2))
    if integ_type == "leapfrog":
        if integ_order not in (1, 2):
            raise ValueError(f"integrator.order must be 1 or 2 for leapfrog, got {integ_order}")
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

    return RunConfig(
        seed=int(raw.get("seed", 42)),
        particles=ParticlesConfig(file=str(particles_raw["file"])),
        softening=SofteningConfig(
            default=_validate_positive("softening.default", float(soft_raw.get("default", 0.01))),
            per_particle=bool(soft_raw.get("per_particle", False)),
            file=soft_raw.get("file"),
        ),
        force=ForceConfig(
            method=method,
            theta=_validate_positive("force.theta", float(force_raw.get("theta", 0.5))),
        ),
        integrator=IntegratorConfig(
            type=integ_type,  # type: ignore[arg-type]
            order=integ_order,  # type: ignore[arg-type]
            dt=_validate_positive("integrator.dt", float(integ_raw.get("dt", 0.01))),
            n_steps=int(_validate_positive("integrator.n_steps", float(integ_raw.get("n_steps", 100)), allow_zero=False)),
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
        ),
        analysis=AnalysisConfig(
            density_bins=max(1, int(ana_raw.get("density_bins", 20))),
            r_max=float(ana_raw["r_max"]) if "r_max" in ana_raw else None,
        ),
        base_dir=config_path.parent,
    )
