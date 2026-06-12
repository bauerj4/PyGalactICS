"""Self-consistent potential solver orchestration."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from galacticsics.io import read_harmonic_potential
from galacticsics.io.formats import read_component_masses, read_rtidal
from galacticsics.io.legacy_inputs import write_dbh_input, write_gendenspsi_input
from galacticsics.legacy.runner import LegacyRunner, LegacyRunError
from galacticsics.models import GalaxyModel
from galacticsics.potential.harmonics import HarmonicPotential


@dataclass
class SolveDiagnostics:
    """
    Convergence and mass diagnostics from a potential solve.

    Attributes
    ----------
    tidal_radius : float
        Tidal radius in kpc (from ``rtidal.dat``).
    component_masses : dict
        Disk, bulge, and halo masses and scale radii from ``mr.dat``.
        Keys are ``"disk"``, ``"bulge"``, ``"halo"``; values are
        ``(mass, radius)`` tuples in GalactICS units.
    work_dir : Path
        Directory where the legacy solver ran (useful for debugging).
    """

    tidal_radius: float
    component_masses: dict[str, tuple[float, float]]
    work_dir: Path


@dataclass
class SolveResult:
    """
    Output of :func:`solve_potential`.

    Attributes
    ----------
    potential : HarmonicPotential
        Self-consistent multipole potential read from ``dbh.dat``.
    diagnostics : SolveDiagnostics
        Tidal radius and component mass summary.
    """

    potential: HarmonicPotential
    diagnostics: SolveDiagnostics


def solve_potential(
    model: GalaxyModel,
    *,
    work_dir: Optional[Path | str] = None,
    cleanup: bool = True,
    npsi: int = 1000,
    nint: int = 20,
    timeout: float | None = 3600.0,
) -> SolveResult:
    """
    Run the legacy self-consistent Poisson solver (``dbh``).

    This function writes stdin input files from ``model``, invokes the isolated
    Fortran executable in ``legacy/bin/dbh``, and loads the resulting
    ``dbh.dat`` into a :class:`~galacticsics.potential.harmonics.HarmonicPotential`.

    Parameters
    ----------
    model : GalaxyModel
        Galaxy configuration including halo, disk, bulge, gas, grid parameters.
    work_dir : path-like, optional
        Directory for the solver run. If ``None``, a temporary directory is
        created. When ``cleanup=False``, the directory is preserved for
        inspection (contains ``refpoints.dat``, ``mr.dat``, etc.).
    cleanup : bool, optional
        If ``True`` and a temporary ``work_dir`` was created, delete it after
        reading outputs. Default is ``True``. Set ``False`` to keep artifacts.
    npsi : int, optional
        Energy grid size for ``in.gendenspsi``. Default 1000.
    nint : int, optional
        Integration steps for DF tables. Default 20.
    timeout : float or None, optional
        Wall-clock timeout in seconds for the ``dbh`` subprocess. Default 3600.

    Returns
    -------
    SolveResult
        Solved potential and diagnostics.

    Raises
    ------
    LegacyRunError
        If ``dbh`` exits with an error.
    FileNotFoundError
        If ``legacy/bin/dbh`` has not been built.

    Notes
    -----
    **Isolation.** The Fortran source under ``legacy/fortran/`` is never
    imported. Only the compiled binary is executed via
    :class:`~galacticsics.legacy.runner.LegacyRunner`.

    **Performance.** Full Milky Way models (``nr=20000``) may take several
    minutes. Use smaller ``model.grid.nr`` for quick tests.

    Examples
    --------
    >>> from galacticsics.models import GalaxyModel
    >>> from galacticsics.potential.solver import solve_potential
    >>> model = GalaxyModel.nfw_halo_only()
    >>> model.grid.nr = 500  # smaller grid for a fast test
    >>> result = solve_potential(model, cleanup=False)
    >>> result.diagnostics.tidal_radius > 0
    True
    """
    tmp: tempfile.TemporaryDirectory[str] | None = None
    owned_tmp = work_dir is None
    if owned_tmp:
        tmp = tempfile.TemporaryDirectory(prefix="galacticsics_dbh_")
        work_dir = Path(tmp.name)
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    write_gendenspsi_input(work_dir / "in.gendenspsi", npsi=npsi, nint=nint)
    write_dbh_input(model, work_dir / "in.dbh")

    runner = LegacyRunner(work_dir)
    runner.run("dbh", stdin_path=work_dir / "in.dbh", timeout=timeout)

    dbh_path = work_dir / "dbh.dat"
    if not dbh_path.is_file():
        raise LegacyRunError(f"dbh did not produce dbh.dat in {work_dir}")

    potential = read_harmonic_potential(dbh_path)
    tidal = read_rtidal(work_dir / "rtidal.dat") if (work_dir / "rtidal.dat").is_file() else 0.0
    masses = (
        read_component_masses(work_dir / "mr.dat")
        if (work_dir / "mr.dat").is_file()
        else {}
    )

    diagnostics = SolveDiagnostics(
        tidal_radius=tidal,
        component_masses=masses,
        work_dir=work_dir,
    )

    if owned_tmp and cleanup and tmp is not None:
        tmp.cleanup()

    return SolveResult(potential=potential, diagnostics=diagnostics)
