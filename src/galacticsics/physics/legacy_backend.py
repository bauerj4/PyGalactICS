"""Legacy Fortran subprocess backend."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from galacticsics.io.formats import cordbh_is_valid, cordbh_needs_refresh
from galacticsics.io.legacy_inputs import (
    write_dbh_input,
    write_diskdf_input,
    write_gendenspsi_input,
    write_genbulge_input,
    write_gendisk_input,
    write_genhalo_input,
)
from galacticsics.io.formats import read_particles_ascii
from galacticsics.legacy.runner import LegacyRunner, LegacyRunError
from galacticsics.models import GalaxyModel
from galacticsics.sampling.particles import ParticleSet

if TYPE_CHECKING:
    from galacticsics.sampling.sampler import SampleConfig

_DISKDF_GRID_HINT = (
    "diskdf did not produce a valid cordbh.dat. This often happens when the DBH "
    "grid is too coarse for diskdf (with coarse_grid=True keep lmax≤4, or set "
    "coarse_grid=False for production lmax=6 solves)."
)


def legacy_solve_potential(
    model: GalaxyModel,
    work_dir: Path,
    *,
    npsi: int = 1000,
    nint: int = 20,
    timeout: float | None = 3600.0,
    stream_output: bool = False,
) -> None:
    write_gendenspsi_input(work_dir / "in.gendenspsi", npsi=npsi, nint=nint)
    write_dbh_input(model, work_dir / "in.dbh")
    runner = LegacyRunner(work_dir)
    runner.run("dbh", stdin_path=work_dir / "in.dbh", timeout=timeout, stream_output=stream_output)


def legacy_ensure_disk_df(
    model: GalaxyModel,
    work_dir: Path,
    *,
    stream_output: bool = False,
    npsi: int = 1000,
    nint: int = 20,
) -> None:
    work_dir = Path(work_dir)
    cordbh = work_dir / "cordbh.dat"
    if cordbh_is_valid(cordbh) and not cordbh_needs_refresh(work_dir):
        return
    if cordbh.is_file():
        cordbh.unlink()
    if not (work_dir / "h.dat").is_file():
        raise FileNotFoundError(
            "h.dat required for diskdf; run solve_potential with a halo or "
            "provide artifact_dir containing h.dat"
        )
    gendenspsi = work_dir / "in.gendenspsi"
    if not gendenspsi.is_file():
        write_gendenspsi_input(gendenspsi, npsi=npsi, nint=nint)
    runner = LegacyRunner(work_dir)
    freqdbh = work_dir / "freqdbh.dat"
    if not freqdbh.is_file():
        runner.run("getfreqs", stream_output=stream_output)
    write_diskdf_input(model, work_dir / "in.diskdf")
    runner.run("diskdf", stdin_path=work_dir / "in.diskdf", stream_output=stream_output)
    if not cordbh_is_valid(cordbh):
        raise LegacyRunError(
            f"{_DISKDF_GRID_HINT}\n"
            f"work_dir={work_dir} cordbh_bytes="
            f"{cordbh.stat().st_size if cordbh.is_file() else 0}"
        )


def legacy_sample_disk(
    work_dir: Path,
    config: "SampleConfig",
    *,
    stream_output: bool = False,
) -> ParticleSet:
    runner = LegacyRunner(work_dir)
    stdin = work_dir / "in.disk"
    write_gendisk_input(
        stdin,
        n_particles=config.n_disk,
        seed=config.seed_disk,
        center=config.center,
    )
    out = work_dir / "disk"
    result = runner.run("gendisk", stdin_path=stdin, stream_output=stream_output)
    out.write_text(result.stdout)
    return ParticleSet(read_particles_ascii(out), component="disk")


def legacy_sample_halo(
    work_dir: Path,
    config: "SampleConfig",
    *,
    stream_output: bool = False,
) -> ParticleSet:
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


def legacy_sample_bulge(
    work_dir: Path,
    config: "SampleConfig",
    *,
    stream_output: bool = False,
) -> ParticleSet:
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
