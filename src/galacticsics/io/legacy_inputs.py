"""Generate stdin input files for legacy executables from Python models."""

from __future__ import annotations

from pathlib import Path

from galacticsics.models import GalaxyModel


def write_gendenspsi_input(path: Path, *, npsi: int = 1000, nint: int = 20) -> None:
    """
    Write ``in.gendenspsi`` for halo/bulge DF table generation.

    Parameters
    ----------
    path : Path
        Output file path (typically ``in.gendenspsi`` in the run directory).
    npsi : int, optional
        Number of points on the energy grid for Eddington inversion tables.
        Default is 1000 (matches ``models/MilkyWay/in.gendenspsi``).
    nint : int, optional
        Number of integration steps per DF table entry. Default is 20.

    Notes
    -----
    Required by ``dbh`` before the Poisson solver starts; read at line 129 of
    ``legacy/fortran/dbh.f``.
    """
    path.write_text(f"{npsi} {nint}\n")


def write_dbh_input(model: GalaxyModel, path: Path) -> None:
    """
    Write ``in.dbh`` stdin script for the legacy ``dbh`` executable.

    The file format mirrors interactive prompts in ``legacy/fortran/dbh.f``:
    yes/no flags followed by numeric parameter lines.

    Parameters
    ----------
    model : GalaxyModel
        Galaxy component configuration and multipole grid settings.
    path : Path
        Output path (typically ``in.dbh`` in the run directory).

    Examples
    --------
    >>> from pathlib import Path
    >>> from galacticsics.models import GalaxyModel
    >>> write_dbh_input(GalaxyModel.milky_way_disk_halo(), Path("in.dbh"))
    """
    lines: list[str] = []

    def yn(enabled: bool) -> str:
        return "y" if enabled else "n"

    halo = model.halo
    if halo and halo.enabled:
        lines.append("y")
        lines.append(
            f"{halo.r_outer} {halo.v0} {halo.a} {halo.dr_trunc} {halo.cusp}"
        )
    else:
        lines.append("n")

    disk = model.disk
    if disk and disk.enabled:
        lines.append("y")
        lines.append(
            f"{disk.mass} {disk.scale_length} {disk.outer_radius} "
            f"{disk.scale_height} {disk.trunc_width} {disk.hole_radius} {disk.core_radius}"
        )
    else:
        lines.append("n")

    disk2 = model.disk2
    if disk2 and disk2.enabled:
        lines.append("y")
        lines.append(
            f"{disk2.mass} {disk2.scale_length} {disk2.outer_radius} "
            f"{disk2.scale_height} {disk2.trunc_width}"
        )
    else:
        lines.append("n")

    gas = model.gas
    if gas and gas.enabled:
        lines.append("y")
        lines.append(
            f"{gas.mass} {gas.scale_length} {gas.outer_radius} {gas.z_scale} "
            f"{gas.trunc_width} {gas.rz_scale} {gas.z_max} {gas.gamma}"
        )
    else:
        lines.append("n")

    bulge = model.bulge
    if bulge and bulge.enabled:
        lines.append("y")
        lines.append(f"{bulge.n_sersic} {bulge.ppp} {bulge.v0} {bulge.a}")
    else:
        lines.append("n")

    bh = model.black_hole
    if bh and bh.enabled and bh.mass > 0:
        lines.append("y")
        lines.append(f"{bh.mass}")
    else:
        lines.append("n")

    g = model.grid
    lines.append(f"{g.dr} {g.nr}")
    lines.append(f"{g.lmax}")

    path.write_text("\n".join(lines) + "\n")


def write_diskdf_input(model: GalaxyModel, path: Path) -> None:
    """
    Write ``in.diskdf`` stdin for the legacy ``diskdf`` executable.

    Parameters
    ----------
    model : GalaxyModel
        Source of ``disk_kinematics`` parameters.
    path : Path
        Output path.
    """
    k = model.disk_kinematics
    path.write_text(
        f"{k.sigma_r0} {k.sigma_r_scale}\n{k.n_radial_steps}\n{k.n_iterations}\npsfile\n"
    )


def write_gendisk_input(
    path: Path,
    *,
    n_particles: int,
    seed: int = -1,
    center: bool = True,
) -> None:
    """
    Write stdin lines for ``gendisk`` (one value per ``iquery`` prompt).

    Parameters
    ----------
    path : Path
        Output file, typically ``in.disk``.
    n_particles : int
        Number of disk particles.
    seed : int, optional
        Negative integer RNG seed. Default ``-1``.
    center : bool, optional
        Center the disk (``icofm=1``). Default ``True``.
    """
    path.write_text(f"{n_particles}\n{seed}\n{int(center)}\n")


def write_genhalo_input(
    path: Path,
    *,
    n_particles: int,
    seed: int = -1,
    center: bool = True,
    streaming: float = 0.5,
) -> None:
    """
    Write stdin lines for ``genhalo``.

    Parameters
    ----------
    path : Path
        Output file, typically ``in.halo``.
    n_particles : int
        Number of halo particles.
    seed : int
        RNG seed (negative integer).
    center : bool
        Center particles if ``True``.
    streaming : float
        Streaming fraction for azimuthal velocities.
    """
    path.write_text(f"{streaming}\n{n_particles}\n{seed}\n{int(center)}\n")


def write_genbulge_input(
    path: Path,
    *,
    n_particles: int,
    seed: int = -1,
    center: bool = True,
    streaming: float = 0.0,
) -> None:
    """Write stdin lines for ``genbulge`` (same prompt order as ``genhalo``)."""
    path.write_text(f"{streaming}\n{n_particles}\n{seed}\n{int(center)}\n")
