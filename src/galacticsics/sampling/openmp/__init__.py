"""OpenMP sampler dispatch (optional C extension).

Disk, halo, and bulge rejection sampling are accelerated in ``_sampler_c`` when
the extension is built with OpenMP (``SampleConfig.use_openmp=True`` by default).
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from galacticsics.sampling.openmp.pack import (
    flat_to_particle_dtype,
    pack_bulge_context,
    pack_disk_context,
    pack_halo_context,
)
from galacticsics.sampling.particles import ParticleSet

if TYPE_CHECKING:
    from galacticsics.sampling.sampler import SampleConfig

try:
    from galacticsics.sampling import _sampler_c
except ImportError:
    _sampler_c = None  # type: ignore[assignment]


def extension_available() -> bool:
    return _sampler_c is not None


def reload_extension() -> bool:
    """Re-import the C extension (e.g. after ``pip install -e .`` without kernel restart)."""
    global _sampler_c
    try:
        mod = importlib.import_module("galacticsics.sampling._sampler_c")
        importlib.reload(mod)
        _sampler_c = mod
    except ImportError:
        _sampler_c = None  # type: ignore[assignment]
        return False
    return True


def openmp_enabled() -> bool:
    if _sampler_c is None:
        return False
    return bool(_sampler_c.extension_available())


@dataclass(frozen=True)
class SamplerBackendStatus:
    """Resolved IC sampler backend for disk/halo/bulge components."""

    backend: str  # "openmp" | "python"
    reason: str
    threads_label: str = ""

    @property
    def is_openmp(self) -> bool:
        return self.backend == "openmp"

    def log_label(self) -> str:
        if self.is_openmp:
            return f"OpenMP/{self.threads_label or 'default'}"
        return f"Python ({self.reason})"


def openmp_sampler_status(config: "SampleConfig | None" = None) -> SamplerBackendStatus:
    """Return the sampler backend that ``sample_*_python`` would use."""
    if config is not None and not config.use_openmp:
        return SamplerBackendStatus("python", "sample.use_openmp=false")
    if not extension_available():
        return SamplerBackendStatus(
            "python",
            "extension not built; pip install -e '.[dev]' + restart kernel",
        )
    if not openmp_enabled():
        return SamplerBackendStatus("python", "OpenMP disabled at compile time")
    threads = 0 if config is None else config.n_openmp_threads
    label = "all cores" if threads == 0 else f"{threads} threads"
    return SamplerBackendStatus("openmp", "C extension", threads_label=label)


def warn_python_sampler_fallback(
    component: str,
    reason: str,
    *,
    progress_log: Callable[[str], None] | None = None,
) -> None:
    """Emit a visible warning when sampling falls back to Python."""
    msg = (
        f"WARNING: {component} using slow Python sampler ({reason}). "
        "Fix: from repo root run `make install-dev` or `pip install -e '.[dev]'`, "
        "restart the Jupyter kernel, keep `sample.use_openmp=true`, "
        "and re-run sample with `skip_done=false`."
    )
    if progress_log:
        progress_log(f"  {msg}")
    else:
        warnings.warn(msg, stacklevel=3)


def sample_halo_openmp(
    work_dir: Path,
    *,
    n_particles: int,
    seed: int = -1,
    center: bool = True,
    streaming: float = 0.5,
    max_attempts: int | None = None,
    n_threads: int = 0,
    progress_log: Callable[[str], None] | None = None,
) -> ParticleSet:
    if _sampler_c is None:
        raise RuntimeError("OpenMP sampler extension not built")
    pack = pack_halo_context(work_dir)
    mass = pack["halomass"] / n_particles
    attempt_limit = max_attempts if max_attempts is not None else max(500_000, 15 * n_particles)
    flat = _sampler_c.sample_halo(
        pack=pack,
        n_particles=n_particles,
        seed=seed,
        mass=mass,
        haloedge=pack["haloedge"],
        rhomax=pack["rhomax"],
        rhomin=pack["rhomin"],
        streaming=streaming,
        n_threads=n_threads,
        max_attempts=attempt_limit,
        center=int(center),
    )
    if progress_log:
        progress_log(f"  genhalo: {n_particles:,}/{n_particles:,} particles (OpenMP)")
    data = flat_to_particle_dtype(flat, n_particles)
    return ParticleSet(data, component="halo")


def sample_disk_openmp(
    work_dir: Path,
    *,
    n_particles: int,
    seed: int = -1,
    center: bool = True,
    max_attempts: int | None = None,
    n_threads: int = 0,
    progress_log: Callable[[str], None] | None = None,
) -> ParticleSet:
    if _sampler_c is None:
        raise RuntimeError("OpenMP sampler extension not built")
    pack = pack_disk_context(work_dir)
    mass = pack["disk_mass"] / n_particles
    attempt_limit = max_attempts if max_attempts is not None else max(500_000, 15 * n_particles)
    flat = _sampler_c.sample_disk(
        pack=pack,
        n_particles=n_particles,
        seed=seed,
        mass=mass,
        rd=pack["rd"],
        zd=pack["zd"],
        rtrunc=pack["rtrunc"],
        rhomax=pack["rhomax"],
        rhomin=pack["rhomin"],
        n_threads=n_threads,
        max_attempts=attempt_limit,
        center=int(center),
    )
    if progress_log:
        progress_log(f"  gendisk: {n_particles:,}/{n_particles:,} particles (OpenMP)")
    data = flat_to_particle_dtype(flat, n_particles)
    return ParticleSet(data, component="disk")


def sample_bulge_openmp(
    work_dir: Path,
    *,
    n_particles: int,
    seed: int = -1,
    center: bool = True,
    streaming: float = 0.5,
    max_attempts: int | None = None,
    n_threads: int = 0,
    progress_log: Callable[[str], None] | None = None,
) -> ParticleSet:
    if _sampler_c is None:
        raise RuntimeError("OpenMP sampler extension not built")
    if not hasattr(_sampler_c, "sample_bulge"):
        raise RuntimeError(
            "OpenMP sampler extension lacks sample_bulge; rebuild with pip install -e '.[dev]'"
        )
    pack = pack_bulge_context(work_dir)
    mass = pack["bulgemass"] / n_particles
    attempt_limit = max_attempts if max_attempts is not None else max(500_000, 15 * n_particles)
    flat = _sampler_c.sample_bulge(
        pack=pack,
        n_particles=n_particles,
        seed=seed,
        mass=mass,
        bulgeedge=pack["bulgeedge"],
        wmax=pack["wmax"],
        wmin=pack["wmin"],
        streaming=streaming,
        n_threads=n_threads,
        max_attempts=attempt_limit,
        center=int(center),
    )
    if progress_log:
        progress_log(f"  genbulge: {n_particles:,}/{n_particles:,} particles (OpenMP)")
    data = flat_to_particle_dtype(flat, n_particles)
    return ParticleSet(data, component="bulge")
