"""Legacy ``appdiskdens.f`` / ``appdiskpot.f`` / ``diskpotentialestimate.f`` ports."""

from __future__ import annotations

import math

import numpy as np

from galacticsics.models import ExponentialDisk, GalaxyModel


def _truncation_factors(r: float, outer_radius: float, trunc_width: float) -> tuple[float, float]:
    """
    Legacy ``eerfc`` and Gaussian ``eexp`` from ``disksurfdens``.

    Parameters
    ----------
    r : float
        Spherical radius [kpc].
    outer_radius, trunc_width : float
        Disk truncation parameters.

    Returns
    -------
    eerfc : float
        Complementary error-function truncation factor in ``[0, 1]``.
    eexp : float
        Derivative contribution from the truncation edge.
    """
    t = math.sqrt(0.5) * (r - outer_radius) / trunc_width
    t2 = t * t
    if t < -4.0:
        return 1.0, 0.0
    if t < 4.0:
        eexp = math.exp(-t2) / math.sqrt(2.0 * math.pi) / trunc_width
        return 0.5 * math.erfc(t), eexp
    return 0.0, 0.0


def disk_surface_radial_derivatives(
    r: float,
    disk: ExponentialDisk,
) -> tuple[float, float, float]:
    """
    Return ``(f, df/dr, d²f/dr²)`` for the truncated exponential surface density.

    Parameters
    ----------
    r : float
        Spherical radius in the disk plane [kpc].
    disk : ExponentialDisk
        Disk component parameters.

    Returns
    -------
    f : float
        Surface density Σ(R) [kpc\\ :sup:`-2`].
    f1r : float
        Radial derivative ``(1/r) dΣ/dR`` at cylindrical radius ``R=r`` when ``r>0``.
    f2 : float
        Second radial derivative contribution.
    """
    eerfc, eexp = _truncation_factors(r, disk.outer_radius, disk.trunc_width)
    dc = disk.disk_const
    rd = disk.scale_length
    arg1 = -r / rd

    if disk.hole_radius == 0.0:
        sg = dc * math.exp(arg1)
        sg1 = -dc / rd * math.exp(arg1)
        sg2 = dc / (rd * rd) * math.exp(arg1)
    else:
        tmp2 = math.sqrt(r * r + disk.hole_radius**2)
        arg2 = -tmp2 / disk.core_radius
        sg = dc * (math.exp(arg1) - math.exp(arg2))
        sg1 = dc * (-math.exp(arg1) / rd + r / disk.core_radius / tmp2 * math.exp(arg2))
        sg2 = dc * (
            math.exp(arg1) / (rd * rd)
            + math.exp(arg2)
            * (disk.core_radius * disk.hole_radius**2 - r * r * tmp2)
            / (disk.core_radius**2 * tmp2**3)
        )

    f = sg * eerfc
    if r > 0.0:
        f1r = (sg1 * eerfc + eexp * sg) / r
        f2 = sg2 * eerfc + 2.0 * sg1 * eexp + eexp * ((r - disk.outer_radius) / disk.trunc_width**2) * sg
    else:
        f1r = 0.0
        f2 = 0.0
    return f, f1r, f2


def disk_vertical_derivatives(z: float, zdisk: float) -> tuple[float, float, float]:
    """
    Return ``(g, dg/dz, d²g/dz²)`` for ``g(z) = log(cosh(z/z_d))``.

    Parameters
    ----------
    z : float
        Height above the midplane [kpc].
    zdisk : float
        Disk scale height [kpc].

    Returns
    -------
    g, g1, g2 : float
        Vertical profile and its first two derivatives.
    """
    zz = z / zdisk
    if abs(zz) > 50.0:
        return abs(zz), math.copysign(1.0, zz), 0.0
    cosh_zz = math.cosh(zz)
    return math.log(cosh_zz), math.tanh(zz), 1.0 / (cosh_zz * cosh_zz)


def approximate_disk_density(s: float, z: float, model: GalaxyModel) -> float:
    """
    Laplacian of the approximate disk potential (legacy ``appdiskdens``).

    Subtracted during harmonic synthesis; see :mod:`galacticsics.potential.poisson.densities`.

    Parameters
    ----------
    s, z : float
        Cylindrical coordinates [kpc].
    model : GalaxyModel
        Galaxy with an enabled disk.

    Returns
    -------
    float
        Mass density [kpc\\ :sup:`-3`].
    """
    disk = model.disk
    if disk is None or not disk.enabled:
        return 0.0
    r = math.hypot(s, z)
    f, f1r, f2 = disk_surface_radial_derivatives(r, disk)
    g, g1, g2 = disk_vertical_derivatives(z, disk.scale_height)
    h = disk.scale_height
    return 0.5 * (f2 * h * g + 2.0 * f1r * g * h + 2.0 * f1r * g1 * z + f * g2 / h)


def approximate_disk_potential(s: float, z: float, model: GalaxyModel) -> float:
    """
    Approximate disk potential (legacy ``appdiskpot``).

    Parameters
    ----------
    s, z : float
        Cylindrical coordinates [kpc].
    model : GalaxyModel
        Galaxy with an enabled disk.

    Returns
    -------
    float
        Disk potential contribution [100 km/s]\\ :sup:`2`].
    """
    disk = model.disk
    if disk is None or not disk.enabled:
        return 0.0
    r = math.hypot(s, z)
    f, _, _ = disk_surface_radial_derivatives(r, disk)
    if f == 0.0:
        return 0.0
    g, _, _ = disk_vertical_derivatives(z, disk.scale_height)
    return -2.0 * math.pi * f * disk.scale_height * g


def disk_densestimate(s: float, z: float, model: GalaxyModel) -> float:
    """
    Legacy ``diskdensestimate(s, z)`` used by ``diskpotentialestimate``.

    This is **not** the same as :func:`approximate_disk_density` (``appdiskdens``).
    It evaluates ``0.5 * Sigma(R) * (d²g/dz²) / z_d`` with ``g = log cosh(z/z_d)``.

    Parameters
    ----------
    s, z : float
        Cylindrical coordinates [kpc].
    model : GalaxyModel
        Galaxy with an enabled exponential disk.

    Returns
    -------
    float
        Mass density estimate [GalactICS units kpc\\ :sup:`-3`].
    """
    disk = model.disk
    if disk is None or not disk.enabled:
        return 0.0
    r = math.hypot(s, z)
    f, _, _ = disk_surface_radial_derivatives(r, disk)
    _, _, g2 = disk_vertical_derivatives(z, disk.scale_height)
    return 0.5 * f * g2 / disk.scale_height


def integrated_disk_densestimate_on_shell(
    r: float,
    model: GalaxyModel,
    *,
    ntheta: int = 100,
) -> float:
    """
    Azimuthally integrated ``diskdensestimate`` on a spherical shell (``diskpotentialestimate``).

    Parameters
    ----------
    r : float
        Spherical shell radius [kpc].
    model : GalaxyModel
        Galaxy with an enabled disk.
    ntheta : int, optional
        Polar quadrature count (legacy default 100).

    Returns
    -------
    float
        Shell-averaged disk density suitable for monopole Poisson seeding.
    """
    disk = model.disk
    if disk is None or not disk.enabled or r <= 0.0:
        return 0.0
    r_work = max(r, 1e-8)
    ctheta_max = min(1.0, 10.0 * disk.scale_height / r_work)
    if ctheta_max <= 0.0:
        return 0.0
    dctheta = ctheta_max / ntheta
    acc = 0.0
    acc += disk_densestimate(r_work * math.sqrt(1.0 - ctheta_max**2), r_work * ctheta_max, model)
    acc += disk_densestimate(r_work, 0.0, model)
    for is_ in range(1, ntheta - 1, 2):
        ctheta = is_ * dctheta
        acc += 4.0 * disk_densestimate(
            r_work * math.sqrt(1.0 - ctheta**2), r_work * ctheta, model
        )
    for is_ in range(2, ntheta - 2, 2):
        ctheta = is_ * dctheta
        acc += 2.0 * disk_densestimate(
            r_work * math.sqrt(1.0 - ctheta**2), r_work * ctheta, model
        )
    return acc * dctheta / 3.0


def disk_monopole_density_grid(model: GalaxyModel, *, ntheta: int = 100) -> np.ndarray:
    """
    Tabulate legacy ``ddens1(ir)`` on the model radial grid.

    Parameters
    ----------
    model : GalaxyModel
        Galaxy with disk + grid parameters.
    ntheta : int, optional
        Polar quadrature count per shell.

    Returns
    -------
    ndarray, shape (nr + 1,)
        Monopole disk density estimate at each shell [kpc\\ :sup:`-3`].
    """
    dr = model.grid.dr
    nr = model.grid.nr
    radii = np.arange(nr + 1, dtype=float) * dr
    return np.array(
        [integrated_disk_densestimate_on_shell(float(r), model, ntheta=ntheta) for r in radii],
        dtype=float,
    )
