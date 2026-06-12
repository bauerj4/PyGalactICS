"""
Two-step halo-first potential workflow.

Step 1 fits or solves an axisymmetric halo multipole representation (``h.dat``)
from analytic NFW parameters or from a particle distribution.  Step 2 solves
disk, bulge, and gas in a fixed external halo, merges harmonic coefficients,
and prepares files for ``getfreqs`` / ``diskdf`` / ``gendisk``.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from galacticsics.io.formats import (
    merge_harmonic_potentials,
    read_component_masses,
    read_harmonic_potential,
    read_halo_harmonics,
    read_rtidal,
    write_harmonic_potential,
)
from galacticsics.io.legacy_inputs import write_dbh_input, write_gendenspsi_input
from galacticsics.legacy.runner import LegacyRunner, LegacyRunError
from galacticsics.models import GalaxyModel
from galacticsics.potential.harmonics import HarmonicPotential
from galacticsics.potential.solver import SolveDiagnostics, SolveResult, solve_potential


@dataclass
class HaloFirstStepResult:
    """
    Output of halo-only Poisson solve (step 1).

    Attributes
    ----------
    halo_potential : HarmonicPotential
        Halo-only multipole coefficients read from ``h.dat`` (``apot`` and
        ``fr`` blocks are pure halo contributions).
    total_potential : HarmonicPotential
        Full solve from ``dbh.dat`` when only the halo component is enabled
        (identical to ``halo_potential`` for a single-component run).
    diagnostics : SolveDiagnostics
        Tidal radius and ``mr.dat`` masses from the halo solve.
    work_dir : Path
        Directory containing ``h.dat``, ``denspsihalo.dat``, ``dfnfw.dat``, etc.
    """

    halo_potential: HarmonicPotential
    total_potential: HarmonicPotential
    diagnostics: SolveDiagnostics
    work_dir: Path


@dataclass
class BaryonsInFixedHaloResult:
    """
    Output of baryon solve in a fixed halo (step 2).

    Attributes
    ----------
    potential : HarmonicPotential
        Combined multipole potential (halo ``h.dat`` + baryon ``dbh`` harmonics)
        written to ``work_dir/dbh.dat`` for legacy samplers.
    halo_potential : HarmonicPotential
        Fixed halo harmonics from step 1 (unchanged).
    baryon_potential : HarmonicPotential
        Harmonics from the baryon-only ``dbh`` pass (disk/bulge/gas only).
    diagnostics : SolveDiagnostics
        Diagnostics from the baryon-only ``dbh`` run.
    work_dir : Path
        Run directory with merged ``dbh.dat``, ``h.dat``, and DF aux files.
    """

    potential: HarmonicPotential
    halo_potential: HarmonicPotential
    baryon_potential: HarmonicPotential
    diagnostics: SolveDiagnostics
    work_dir: Path


def _copy_halo_artifacts(src: Path, dst: Path) -> None:
    """Copy auxiliary halo DF files required by ``getfreqs`` / ``genhalo``."""
    for name in (
        "h.dat",
        "denspsihalo.dat",
        "denspsibulge.dat",
        "dfnfw.dat",
        "dfsersic.dat",
        "dfhalo.table",
        "mr.dat",
    ):
        path = src / name
        if path.is_file():
            shutil.copy2(path, dst / name)


def solve_halo_potential(
    model: GalaxyModel,
    *,
    work_dir: Optional[Path | str] = None,
    cleanup: bool = True,
    npsi: int = 1000,
    nint: int = 20,
    timeout: float | None = 3600.0,
) -> HaloFirstStepResult:
    """
    Step 1: solve an axisymmetric halo and write ``h.dat``.

    Runs ``legacy/bin/dbh`` with all baryon components disabled
    (``GalaxyModel.with_halo_only()``).  At convergence, ``dbh.f`` calls
    ``halopotential`` and writes isolated halo Legendre coefficients to
    ``h.dat``.

    Parameters
    ----------
    model : GalaxyModel
        Must include an enabled :class:`~galacticsics.models.NFWHalo`.  Disk,
        bulge, and gas flags are ignored; only the halo block is passed to
        ``dbh``.
    work_dir : path-like, optional
        Run directory.  A temporary directory is created when omitted.
    cleanup : bool, optional
        Remove a temporary ``work_dir`` after reading outputs.  Default
        ``True``.  Set ``False`` to inspect ``h.dat`` and DF tables.
    npsi, nint : int, optional
        Energy grid for ``in.gendenspsi`` (Eddington inversion tables).
    timeout : float or None, optional
        Subprocess timeout for ``dbh`` [s].

    Returns
    -------
    HaloFirstStepResult
        Halo harmonics, diagnostics, and preserved ``work_dir``.

    Raises
    ------
    ValueError
        If the model has no enabled halo.
    LegacyRunError
        If ``dbh`` fails.
    FileNotFoundError
        If ``h.dat`` is missing after the solve (build issue or bad grid).

    Notes
    -----
    The halo solve is the Poisson problem for a spherical NFW density with
    Eddington-inverted DF (see README § Halo-first workflow).  Particle-based
    fitting should update ``model.halo`` parameters *before* calling this
    function (see :func:`~galacticsics.fitting.halo_particles.estimate_nfw_from_particles`).

    Examples
    --------
    >>> from galacticsics.models import GalaxyModel
    >>> from galacticsics.potential.halo_first import solve_halo_potential
    >>> model = GalaxyModel.nfw_halo_only()
    >>> result = solve_halo_potential(model, cleanup=False)
    >>> (result.work_dir / "h.dat").is_file()
    True
    """
    if model.halo is None or not model.halo.enabled:
        raise ValueError("solve_halo_potential requires an enabled NFWHalo")

    halo_model = model.with_halo_only()
    solve = solve_potential(
        halo_model,
        work_dir=work_dir,
        cleanup=False,
        npsi=npsi,
        nint=nint,
        timeout=timeout,
    )
    h_path = solve.diagnostics.work_dir / "h.dat"
    if not h_path.is_file():
        raise FileNotFoundError(
            f"dbh did not write h.dat in {solve.diagnostics.work_dir}; "
            "ensure legacy/bin/dbh was built from legacy/fortran/dbh.f"
        )

    halo_pot = read_halo_harmonics(h_path)
    total_pot = solve.potential

    return HaloFirstStepResult(
        halo_potential=halo_pot,
        total_potential=total_pot,
        diagnostics=solve.diagnostics,
        work_dir=solve.diagnostics.work_dir,
    )


def solve_baryons_in_fixed_halo(
    model: GalaxyModel,
    halo_work_dir: Path | str,
    *,
    work_dir: Optional[Path | str] = None,
    cleanup: bool = True,
    run_getfreqs: bool = True,
    npsi: int = 1000,
    nint: int = 20,
    timeout: float | None = 3600.0,
) -> BaryonsInFixedHaloResult:
    """
    Step 2: solve baryons in a fixed halo and merge into ``dbh.dat``.

    Procedure
    ---------
    1. Copy ``h.dat`` and halo DF aux files from ``halo_work_dir``.
    2. Run ``dbh`` with halo disabled (``GalaxyModel.with_baryons_only()``).
    3. Read baryon-only harmonics from the new ``dbh.dat``.
    4. Add halo harmonics from ``h.dat`` coefficient-wise:

       .. math::

          \\Phi_{l}^{\\mathrm{tot}}(r) = \\Phi_{l}^{\\mathrm{halo}}(r)
          + \\Phi_{l}^{\\mathrm{baryon}}(r)

       and likewise for ``fr`` and ``adens``.
    5. Write the merged potential back to ``dbh.dat``.
    6. Optionally run ``getfreqs`` (requires both ``dbh.dat`` and ``h.dat``).

    Parameters
    ----------
    model : GalaxyModel
        Baryon configuration (disk, bulge, gas).  The halo block is turned
        off for the ``dbh`` stdin script; the fixed halo enters only through
        ``h.dat``.
    halo_work_dir : path-like
        Directory from :func:`solve_halo_potential` containing ``h.dat``.
    work_dir : path-like, optional
        Baryon run directory (temporary if omitted).
    cleanup : bool, optional
        Delete a temporary ``work_dir`` after completion.
    run_getfreqs : bool, optional
        Invoke ``legacy/bin/getfreqs`` after merging.  Default ``True``.
    npsi, nint, timeout
        Passed through to auxiliary file generation and ``dbh``.

    Returns
    -------
    BaryonsInFixedHaloResult
        Combined potential and preserved artifacts.

    Raises
    ------
    FileNotFoundError
        If ``halo_work_dir/h.dat`` is missing.
    ValueError
        If grid parameters disagree between halo and baryon models.
    """
    halo_root = Path(halo_work_dir)
    h_path = halo_root / "h.dat"
    if not h_path.is_file():
        raise FileNotFoundError(f"h.dat not found in {halo_root}")

    halo_pot = read_halo_harmonics(h_path)
    baryon_model = model.with_baryons_only(halo_grid=halo_pot.model.grid)

    tmp: tempfile.TemporaryDirectory[str] | None = None
    owned = work_dir is None
    if owned:
        tmp = tempfile.TemporaryDirectory(prefix="galacticsics_baryon_")
        work_dir = Path(tmp.name)
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    _copy_halo_artifacts(halo_root, work_dir)
    write_gendenspsi_input(work_dir / "in.gendenspsi", npsi=npsi, nint=nint)
    write_dbh_input(baryon_model, work_dir / "in.dbh")

    runner = LegacyRunner(work_dir)
    runner.run("dbh", stdin_path=work_dir / "in.dbh", timeout=timeout)

    baryon_path = work_dir / "dbh.dat"
    if not baryon_path.is_file():
        raise LegacyRunError(f"dbh did not produce dbh.dat in {work_dir}")

    baryon_pot = read_harmonic_potential(baryon_path)
    combined = merge_harmonic_potentials(halo_pot, baryon_pot, model=model)
    write_harmonic_potential(combined, work_dir / "dbh.dat")

    if run_getfreqs:
        runner.run("getfreqs")

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

    result = BaryonsInFixedHaloResult(
        potential=combined,
        halo_potential=halo_pot,
        baryon_potential=baryon_pot,
        diagnostics=diagnostics,
        work_dir=work_dir,
    )

    if owned and cleanup and tmp is not None:
        tmp.cleanup()

    return result


def run_halo_first_workflow(
    model: GalaxyModel,
    *,
    work_dir: Optional[Path | str] = None,
    cleanup: bool = True,
    run_getfreqs: bool = True,
    timeout: float | None = 3600.0,
) -> BaryonsInFixedHaloResult:
    """
    Execute the full two-step halo-first pipeline.

    Equivalent to :func:`solve_halo_potential` followed by
    :func:`solve_baryons_in_fixed_halo` in a single parent directory.

    Parameters
    ----------
    model : GalaxyModel
        Full galaxy model (halo + baryons).  Step 1 uses ``model.halo`` only;
        step 2 uses all other enabled components.
    work_dir : path-like, optional
        Parent directory for ``halo/`` and ``baryons/`` subdirectories.
    cleanup, run_getfreqs, timeout
        See step functions.

    Returns
    -------
    BaryonsInFixedHaloResult
        Final merged potential in ``work_dir/baryons/``.
    """
    tmp: tempfile.TemporaryDirectory[str] | None = None
    owned = work_dir is None
    if owned:
        tmp = tempfile.TemporaryDirectory(prefix="galacticsics_halo_first_")
        root = Path(tmp.name)
    else:
        root = Path(work_dir)
        root.mkdir(parents=True, exist_ok=True)

    halo_dir = root / "halo"
    baryon_dir = root / "baryons"
    solve_halo_potential(model, work_dir=halo_dir, cleanup=False, timeout=timeout)
    result = solve_baryons_in_fixed_halo(
        model,
        halo_dir,
        work_dir=baryon_dir,
        cleanup=False,
        run_getfreqs=run_getfreqs,
        timeout=timeout,
    )

    if owned and cleanup and tmp is not None:
        tmp.cleanup()

    return result
