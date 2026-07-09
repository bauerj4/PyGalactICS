"""Invoke legacy particle samplers (gendisk, genhalo, genbulge)."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from galacticsics.io.formats import cordbh_is_valid, read_particles_ascii
from galacticsics.legacy.runner import LegacyRunError
from galacticsics.io.legacy_inputs import (
    write_dbh_input,
    write_diskdf_input,
    write_gendenspsi_input,
    write_genbulge_input,
    write_gendisk_input,
    write_genhalo_input,
)
from galacticsics.legacy.runner import LegacyRunner
from galacticsics.models import GalaxyModel
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


_DISKDF_GRID_HINT = (
    "diskdf did not produce a valid cordbh.dat. This often happens when the DBH "
    "grid is too coarse for diskdf (with coarse_grid=True keep lmax≤4, or set "
    "coarse_grid=False for production lmax=6 solves)."
)


def ensure_disk_df(
    model: GalaxyModel,
    work_dir: Path,
    runner: LegacyRunner,
    *,
    stream_output: bool = False,
) -> None:
    """Run ``getfreqs`` and ``diskdf`` if ``cordbh.dat`` is missing or invalid."""
    cordbh = work_dir / "cordbh.dat"
    if cordbh_is_valid(cordbh):
        return
    if cordbh.is_file():
        cordbh.unlink()
    if not (work_dir / "h.dat").is_file():
        raise FileNotFoundError(
            "h.dat required for diskdf; run solve_potential with a halo or "
            "provide artifact_dir containing h.dat"
        )
    runner.run("getfreqs", stream_output=stream_output)
    write_diskdf_input(model, work_dir / "in.diskdf")
    runner.run("diskdf", stdin_path=work_dir / "in.diskdf", stream_output=stream_output)
    if not cordbh_is_valid(cordbh):
        raise LegacyRunError(
            f"{_DISKDF_GRID_HINT}\n"
            f"work_dir={work_dir} cordbh_bytes="
            f"{cordbh.stat().st_size if cordbh.is_file() else 0}"
        )


def sample_disk(
    work_dir: Path,
    config: SampleConfig,
    *,
    output_name: str = "disk",
    stream_output: bool = False,
) -> ParticleSet:
    """Sample stellar disk particles via ``legacy/bin/gendisk``."""
    runner = LegacyRunner(work_dir)
    stdin = work_dir / "in.disk"
    write_gendisk_input(
        stdin,
        n_particles=config.n_disk,
        seed=config.seed_disk,
        center=config.center,
    )
    out = work_dir / output_name
    result = runner.run("gendisk", stdin_path=stdin, stream_output=stream_output)
    out.write_text(result.stdout)
    return ParticleSet(read_particles_ascii(out), component="disk")


def sample_halo(work_dir: Path, config: SampleConfig, *, stream_output: bool = False) -> ParticleSet:
    """Sample halo particles via ``legacy/bin/genhalo``."""
    runner = LegacyRunner(work_dir)
    stdin = work_dir / "in.halo"
    write_genhalo_input(
        stdin,
        n_particles=config.n_halo,
        seed=config.seed_halo,
        center=config.center,
        streaming=config.stream_halo,
    )
    out = work_dir / "halo"
    result = runner.run("genhalo", stdin_path=stdin, stream_output=stream_output)
    out.write_text(result.stdout)
    return ParticleSet(read_particles_ascii(out), component="halo")


def sample_bulge(work_dir: Path, config: SampleConfig, *, stream_output: bool = False) -> ParticleSet:
    """Sample bulge particles via ``legacy/bin/genbulge``."""
    runner = LegacyRunner(work_dir)
    stdin = work_dir / "in.bulge"
    write_genbulge_input(
        stdin,
        n_particles=config.n_bulge,
        seed=config.seed_bulge,
        center=config.center,
        streaming=config.stream_bulge,
    )
    out = work_dir / "bulge"
    result = runner.run("genbulge", stdin_path=stdin, stream_output=stream_output)
    out.write_text(result.stdout)
    return ParticleSet(read_particles_ascii(out), component="bulge")


def sample_galaxy(
    model: GalaxyModel,
    config: SampleConfig,
    *,
    work_dir: Optional[Path] = None,
    artifact_dir: Optional[Path] = None,
    cleanup: bool = True,
    stream_output: bool = False,
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
    runner = LegacyRunner(work_dir)

    if config.n_disk > 0 and config.run_diskdf:
        ensure_disk_df(model, work_dir, runner, stream_output=stream_output)

    particles: dict[str, ParticleSet] = {}
    if config.n_disk > 0:
        particles["disk"] = sample_disk(work_dir, config, stream_output=stream_output)
    if config.n_halo > 0:
        particles["halo"] = sample_halo(work_dir, config, stream_output=stream_output)
    if config.n_bulge > 0:
        particles["bulge"] = sample_bulge(work_dir, config, stream_output=stream_output)

    if owned and cleanup and tmp is not None:
        tmp.cleanup()

    return SampleResult(particles=particles, work_dir=work_dir)
