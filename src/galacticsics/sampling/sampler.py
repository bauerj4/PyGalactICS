"""Particle samplers (Python default; legacy Fortran optional)."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from galacticsics.io.legacy_inputs import write_dbh_input, write_gendenspsi_input
from galacticsics.models import GalaxyModel
from galacticsics.physics.backend import PhysicsBackendKind
from galacticsics.sampling.particles import ParticleSet


@dataclass
class SampleConfig:
    """
    Parameters for legacy Monte Carlo particle sampling.

    Attributes
    ----------
    n_disk, n_halo, n_bulge : int
        Particle counts per component. Zero skips a component.
    seed_disk, seed_halo, seed_bulge : int
        Negative integer seeds for ``ran3`` (legacy convention).
    center : bool
        If ``True``, translate component center of mass to origin (``icofm=1``).
    stream_halo, stream_bulge : float
        Fraction of halo/bulge particles with positive v_phi (streaming).
    run_diskdf : bool
        If ``True`` and ``cordbh.dat`` is missing or invalid, run ``getfreqs`` +
        ``diskdf``.
    """

    n_disk: int = 10_000
    n_halo: int = 50_000
    n_bulge: int = 0
    seed_disk: int = -1
    seed_halo: int = -1
    seed_bulge: int = -1
    center: bool = True
    stream_halo: float = 0.5
    stream_bulge: float = 0.0
    run_diskdf: bool = True
    use_openmp: bool = True
    n_openmp_threads: int = 0


@dataclass
class SampleResult:
    """Particle sets keyed by component name."""

    particles: dict[str, ParticleSet] = field(default_factory=dict)
    work_dir: Path = Path(".")


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.is_file() and src.resolve() != dst.resolve():
        shutil.copy2(src, dst)


def prepare_model_directory(
    model: GalaxyModel,
    work_dir: Path,
    *,
    artifact_dir: Optional[Path] = None,
) -> None:
    """
    Populate *work_dir* with files required by legacy samplers.

    If *artifact_dir* is given, copy ``dbh.dat``, ``mr.dat``, ``h.dat``,
    ``cordbh.dat``, and ``freqdbh.dat`` from that directory. Otherwise write
    ``in.dbh`` / ``in.gendenspsi`` only (caller must run ``solve_potential``).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    write_gendenspsi_input(work_dir / "in.gendenspsi")
    if artifact_dir is not None:
        for name in (
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
        ):
            _copy_if_exists(artifact_dir / name, work_dir / name)
    else:
        write_dbh_input(model, work_dir / "in.dbh")


def ensure_disk_df(
    model: GalaxyModel,
    work_dir: Path,
    runner=None,
    *,
    stream_output: bool = False,
    backend: PhysicsBackendKind | str | None = None,
    progress_log: Callable[[str], None] | None = None,
) -> GalaxyModel:
    """Run frequency tabulation + disk DF correction if ``cordbh.dat`` is missing or invalid."""
    del runner
    from galacticsics.physics.dispatch import backend_ensure_disk_df

    return backend_ensure_disk_df(
        model,
        work_dir,
        backend=backend,
        stream_output=stream_output,
        progress_log=progress_log,
    )


__all__ = ["SampleConfig", "SampleResult", "ensure_disk_df", "prepare_model_directory", "sample_bulge", "sample_disk", "sample_galaxy", "sample_halo"]


def sample_disk(
    work_dir: Path,
    config: SampleConfig,
    *,
    output_name: str = "disk",
    stream_output: bool = False,
    backend: PhysicsBackendKind | str | None = None,
    progress_log: Callable[[str], None] | None = None,
) -> ParticleSet:
    """Sample stellar disk particles."""
    from galacticsics.physics.dispatch import backend_sample_disk

    ps = backend_sample_disk(
        work_dir,
        config,
        backend=backend,
        stream_output=stream_output,
        progress_log=progress_log,
    )
    out = work_dir / output_name
    if out.name != "disk" or not out.is_file():
        ps.write_ascii(out)
    return ps


def sample_halo(
    work_dir: Path,
    config: SampleConfig,
    *,
    stream_output: bool = False,
    backend: PhysicsBackendKind | str | None = None,
    progress_log: Callable[[str], None] | None = None,
) -> ParticleSet:
    """Sample halo particles."""
    from galacticsics.physics.dispatch import backend_sample_halo

    return backend_sample_halo(
        work_dir,
        config,
        backend=backend,
        stream_output=stream_output,
        progress_log=progress_log,
    )


def sample_bulge(
    work_dir: Path,
    config: SampleConfig,
    *,
    stream_output: bool = False,
    backend: PhysicsBackendKind | str | None = None,
    progress_log: Callable[[str], None] | None = None,
) -> ParticleSet:
    """Sample bulge particles."""
    from galacticsics.physics.dispatch import backend_sample_bulge

    return backend_sample_bulge(
        work_dir,
        config,
        backend=backend,
        stream_output=stream_output,
        progress_log=progress_log,
    )


def sample_galaxy(
    model: GalaxyModel,
    config: SampleConfig,
    *,
    work_dir: Optional[Path] = None,
    artifact_dir: Optional[Path] = None,
    cleanup: bool = True,
    stream_output: bool = False,
    backend: PhysicsBackendKind | str | None = None,
    progress_log: Callable[[str], None] | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> SampleResult:
    """
    Sample all requested components into a single working directory.

    Parameters
    ----------
    model : GalaxyModel
        Galaxy parameters (used for diskdf if needed).
    config : SampleConfig
        Particle counts and seeds.
    work_dir : Path, optional
        Run directory; created temporarily if omitted.
    artifact_dir : Path, optional
        Precomputed ``dbh.dat`` etc. (e.g. ``models/MilkyWay``).
    cleanup : bool
        Remove temporary directory on success.

    Returns
    -------
    SampleResult
        Dictionary of :class:`~galacticsics.sampling.particles.ParticleSet`.
    """
    tmp: tempfile.TemporaryDirectory[str] | None = None
    owned = work_dir is None
    if owned:
        tmp = tempfile.TemporaryDirectory(prefix="galacticsics_sample_")
        work_dir = Path(tmp.name)
    else:
        work_dir = Path(work_dir)

    prepare_model_directory(model, work_dir, artifact_dir=artifact_dir)

    if config.n_disk > 0 and config.run_diskdf:
        if on_stage is not None:
            on_stage("diskdf")
        model = ensure_disk_df(
            model,
            work_dir,
            stream_output=stream_output,
            backend=backend,
            progress_log=progress_log,
        )

    particles: dict[str, ParticleSet] = {}
    if config.n_disk > 0:
        if on_stage is not None:
            on_stage("gendisk")
        particles["disk"] = sample_disk(
            work_dir,
            config,
            stream_output=stream_output,
            backend=backend,
            progress_log=progress_log,
        )
    if config.n_halo > 0:
        if on_stage is not None:
            on_stage("genhalo")
        particles["halo"] = sample_halo(
            work_dir,
            config,
            stream_output=stream_output,
            backend=backend,
            progress_log=progress_log,
        )
    if config.n_bulge > 0:
        if on_stage is not None:
            on_stage("genbulge")
        particles["bulge"] = sample_bulge(
            work_dir,
            config,
            stream_output=stream_output,
            backend=backend,
            progress_log=progress_log,
        )

    if owned and cleanup and tmp is not None:
        tmp.cleanup()

    return SampleResult(particles=particles, work_dir=work_dir)
