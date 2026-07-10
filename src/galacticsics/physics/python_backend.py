"""Python IC generation backend for halo, disk, and bulge workflows."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from galacticsics.distribution.diskdf_solve import solve_diskdf_python
from galacticsics.distribution.toomre import apply_toomre_q_target, log_toomre_q
from galacticsics.io.formats import cordbh_is_valid, cordbh_needs_refresh
from galacticsics.models import GalaxyModel
from galacticsics.physics.legacy_backend import _DISKDF_GRID_HINT
from galacticsics.io.legacy_inputs import write_gendenspsi_input
from galacticsics.potential.frequencies_tabulate import tabulate_frequencies
from galacticsics.potential.poisson.solve import solve_poisson_python
from galacticsics.sampling.python.samplers import sample_bulge_python, sample_disk_python, sample_halo_python
from galacticsics.sampling.sampler import SampleConfig


def python_solve_potential(
    model: GalaxyModel,
    work_dir: Path,
    *,
    npsi: int = 1000,
    nint: int = 20,
    max_iter: int = 100,
    n_workers: int | None = None,
    timeout: float | None = None,
    stream_output: bool = False,
) -> None:
    """
    Run the Python Poisson solver and write auxiliary frequency tables.

    Parameters
    ----------
    model : GalaxyModel
        Galaxy configuration.
    work_dir : Path
        Output directory for ``dbh.dat``, ``freqdbh.dat``, and DF tables.
    npsi, nint : int, optional
        Energy-grid resolution and Eddington quadrature count.
    max_iter : int, optional
        Poisson iteration cap.
    n_workers : int or None, optional
        Thread pool size for polar quadrature (``0`` disables parallelism).
    timeout, stream_output
        Ignored (legacy API compatibility).
    """
    del timeout, stream_output
    solve_poisson_python(
        model,
        work_dir,
        npsi=npsi,
        nint=nint,
        max_iter=max_iter,
        n_workers=n_workers,
    )
    write_gendenspsi_input(work_dir / "in.gendenspsi", npsi=npsi, nint=nint)
    tabulate_frequencies(work_dir)


def python_ensure_disk_df(
    model: GalaxyModel,
    work_dir: Path,
    *,
    stream_output: bool = False,
    progress_log: Callable[[str], None] | None = None,
    diskdf_backend: str = "python",
) -> GalaxyModel:
    """
    Ensure a valid ``cordbh.dat`` exists using the Python ``diskdf`` port only.

    Parameters
    ----------
    model : GalaxyModel
        Galaxy model (Toomre-Q scaling may be applied in place).
    work_dir : Path
        Directory containing ``dbh.dat`` and ``freqdbh.dat``.
    stream_output
        Ignored (legacy API compatibility).
    progress_log : callable or None, optional
        Optional progress sink.
    diskdf_backend : str, optional
        Must be ``"python"``; legacy Fortran ``diskdf`` is not used on this path.

    Returns
    -------
    GalaxyModel
        Possibly updated model after Toomre-Q adjustment.

    Raises
    ------
    RuntimeError
        If the Python solve does not produce a valid ``cordbh.dat``.
    ValueError
        If ``diskdf_backend`` requests legacy Fortran.
    """
    del stream_output
    if diskdf_backend not in ("python", "auto"):
        raise ValueError(
            f"Python backend does not support diskdf_backend={diskdf_backend!r}; "
            "use physics_backend=legacy for Fortran diskdf"
        )
    work_dir = Path(work_dir)
    cordbh = work_dir / "cordbh.dat"
    if cordbh_is_valid(cordbh) and not cordbh_needs_refresh(work_dir):
        return model
    if cordbh.is_file():
        cordbh.unlink()
    if not (work_dir / "freqdbh.dat").is_file():
        tabulate_frequencies(work_dir)
    model = apply_toomre_q_target(model, work_dir)
    log_toomre_q(model, work_dir, log=progress_log)
    solve_diskdf_python(model, work_dir)
    if not cordbh_is_valid(cordbh):
        raise RuntimeError(
            f"{_DISKDF_GRID_HINT}\nwork_dir={work_dir} cordbh_bytes="
            f"{cordbh.stat().st_size if cordbh.is_file() else 0}"
        )
    return model


def python_sample_disk(
    work_dir: Path,
    config: SampleConfig,
    *,
    stream_output: bool = False,
    progress_log: Callable[[str], None] | None = None,
):
    """Sample disk particles with the Python ``gendisk`` port."""
    del stream_output
    ps = sample_disk_python(
        work_dir,
        n_particles=config.n_disk,
        seed=config.seed_disk,
        center=config.center,
        progress_log=progress_log,
        config=config,
    )
    ps.write_ascii(work_dir / "disk")
    return ps


def python_sample_halo(
    work_dir: Path,
    config: SampleConfig,
    *,
    stream_output: bool = False,
    progress_log: Callable[[str], None] | None = None,
):
    """Sample halo particles with the Python ``genhalo`` port."""
    del stream_output
    ps = sample_halo_python(
        work_dir,
        n_particles=config.n_halo,
        seed=config.seed_halo,
        center=config.center,
        streaming=config.stream_halo,
        progress_log=progress_log,
        config=config,
    )
    ps.write_ascii(work_dir / "halo")
    return ps


def python_sample_bulge(
    work_dir: Path,
    config: SampleConfig,
    *,
    stream_output: bool = False,
    progress_log: Callable[[str], None] | None = None,
):
    """Sample bulge particles with the Python ``genbulge`` port."""
    del stream_output
    ps = sample_bulge_python(
        work_dir,
        n_particles=config.n_bulge,
        seed=config.seed_bulge,
        center=config.center,
        streaming=config.stream_bulge,
    )
    ps.write_ascii(work_dir / "bulge")
    return ps
