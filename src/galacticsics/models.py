"""
Galaxy component parameter models.

These dataclasses describe the physical and numerical parameters consumed by
the legacy GalactICS solvers and the Python orchestration layer. All lengths
are in kpc, velocities in units of 100 km/s, and masses in GalactICS mass
units (2.325 x 10^9 solar masses) unless noted otherwise.

See Also
--------
galacticsics.io.legacy_inputs.write_dbh_input
galacticsics.potential.solver.solve_potential
galacticsics.units
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

from galacticsics.units import DEFAULT_UNITS, UnitSystem


@dataclass
class NFWHalo:
    """
    Navarro-Frenk-White halo parameters.

    Corresponds to the first interactive block in ``legacy/fortran/dbh.f``.
    When enabled, the self-consistent solver includes a spherical NFW halo
    with Eddington-inverted distribution function.

    Parameters
    ----------
    r_outer : float
        Outer halo radius ``chalo`` [kpc]. Sets the extent of the halo tabulation.
    v0 : float
        Characteristic circular velocity scale [100 km/s]. Related to the NFW
        potential depth; ``psi0 = v0^2`` at the reference point.
    a : float
        NFW scale radius [kpc].
    dr_trunc : float, optional
        Truncation width ``drtrunchalo`` [kpc] for the outer halo cutoff.
        Default is 12.0.
    cusp : float, optional
        Inner slope parameter for the halo density profile. Default is 1.0
        (standard NFW).
    enabled : bool, optional
        If ``False``, the halo is omitted from the model. Default ``True``.

    Notes
    -----
    The volume normalization ``haloconst`` is computed automatically as
    ``2^(1-cusp) * v0^2 / (4 pi a^2)`` matching ``dbh.f`` line 23.
    """

    r_outer: float  # chalo [kpc]
    v0: float  # characteristic velocity [100 km/s]
    a: float  # scale radius [kpc]
    dr_trunc: float = 12.0  # drtrunchalo
    cusp: float = 1.0  # inner slope parameter
    enabled: bool = True

    @property
    def halo_const(self) -> float:
        """Volume normalization haloconst = 2^(1-cusp) v0^2 / (4 pi a^2)."""
        return (2.0 ** (1.0 - self.cusp)) * self.v0**2 / (4.0 * 3.1415926535 * self.a**2)


@dataclass
class ExponentialDisk:
    """
    Exponential/sech² stellar disk (first disk component).

    Surface density on the midplane:

    .. math::

       \\Sigma(R) = \\Sigma_0 \\exp(-R/R_d)\\,
       \\tfrac{1}{2}\\mathrm{erfc}\\!\\left(\\frac{R-R_{\\mathrm{out}}}{\\Delta R}\\right)

    with :math:`\\Sigma_0 = M/(2\\pi R_d^2)`.  Vertical structure is

    .. math::

       \\rho(R,z) = \\frac{\\Sigma(R)}{2 z_d}\\mathrm{sech}^2(z/z_d).

    Parameters match the second interactive block in ``legacy/fortran/dbh.f``.
    """

    mass: float  # rmdisk [GalactICS mass units]
    scale_length: float  # rdisk [kpc]
    outer_radius: float  # outdisk [kpc]
    scale_height: float  # zdisk [kpc]
    trunc_width: float  # drtrunc [kpc]
    hole_radius: float = 0.0  # rhole
    core_radius: float = 1.5  # rcore
    enabled: bool = True

    @property
    def disk_const(self) -> float:
        """Surface density normalization."""
        return self.mass / (2.0 * 3.1415926535 * self.scale_length**2)


@dataclass
class Sech2Disk:
    """Second disk component with sech^2 vertical profile."""

    mass: float
    scale_length: float
    outer_radius: float
    scale_height: float
    trunc_width: float
    enabled: bool = False

    @property
    def disk_const(self) -> float:
        return self.mass / (4.0 * 3.1415926535 * self.scale_length**2 * self.scale_height)


@dataclass
class GasDisk:
    """Gas disk with polytropic vertical structure."""

    mass: float
    scale_length: float
    outer_radius: float
    z_scale: float  # zgas0
    trunc_width: float
    rz_scale: float  # rzgas
    z_max: float  # zgasmax
    gamma: float  # polytropic index
    enabled: bool = False

    @property
    def gas_const(self) -> float:
        return self.mass / (2.0 * 3.1415926535 * self.scale_length**2)


@dataclass
class SersicBulge:
    """Sersic bulge (ppp < 0 selects NFW-like bulge in legacy code)."""

    n_sersic: float  # nnn
    ppp: float
    v0: float
    a: float
    enabled: bool = False

    @property
    def bulge_const(self) -> float:
        return self.v0**2 / (4.0 * 3.1415926535 * self.a**2)


@dataclass
class BlackHole:
    """Optional central black hole."""

    mass: float = 0.0
    enabled: bool = False


@dataclass
class PotentialGrid:
    """Radial grid for multipole expansion."""

    dr: float = 0.02
    nr: int = 20000
    lmax: int = 6


@dataclass
class DiskKinematics:
    """Disk velocity dispersion parameters for diskdf."""

    sigma_r0: float = 0.85  # central sigma_R [100 km/s]
    sigma_r_scale: float = 2.8  # exponential scale length for sigma_R^2 [kpc]
    n_radial_steps: int = 50
    n_iterations: int = 10


@dataclass
class GalaxyModel:
    """Complete galaxy model specification."""

    halo: Optional[NFWHalo] = None
    disk: Optional[ExponentialDisk] = None
    disk2: Optional[Sech2Disk] = None
    gas: Optional[GasDisk] = None
    bulge: Optional[SersicBulge] = None
    black_hole: BlackHole = field(default_factory=BlackHole)
    grid: PotentialGrid = field(default_factory=PotentialGrid)
    disk_kinematics: DiskKinematics = field(default_factory=DiskKinematics)

    @property
    def psi0(self) -> float:
        """Reference potential at origin from halo (if present)."""
        if self.halo and self.halo.enabled:
            return self.halo.v0**2
        if self.bulge and self.bulge.enabled:
            return self.bulge.v0**2
        return 0.0

    @classmethod
    def from_physical(
        cls,
        *,
        units: UnitSystem = DEFAULT_UNITS,
        halo_enabled: bool = True,
        halo_r_outer_kpc: float = 200.0,
        halo_v0_kms: float = 370.0,
        halo_a_kpc: float = 33.0,
        halo_dr_trunc_kpc: float = 12.0,
        halo_cusp: float = 1.0,
        disk_enabled: bool = True,
        disk_mass_msun: float = 3.9525e10,
        disk_scale_length_kpc: float = 2.5,
        disk_outer_radius_kpc: float = 20.25,
        disk_scale_height_kpc: float = 0.25,
        disk_trunc_width_kpc: float = 3.0,
        sigma_r0_kms: float = 85.0,
        sigma_r_scale_kpc: float = 2.8,
        grid_dr_kpc: float = 0.1,
        grid_nr: int = 800,
        grid_lmax: int = 2,
    ) -> GalaxyModel:
        """
        Build a :class:`GalaxyModel` from physical units.

        Internal GalactICS units are used for all solver I/O; this constructor
        accepts kpc, km/s, and solar masses and converts via :class:`UnitSystem`.

        Parameters
        ----------
        units : UnitSystem, optional
            Conversion factors.  Default matches legacy GalactICS (``G=1``).
        halo_r_outer_kpc, halo_v0_kms, halo_a_kpc, halo_dr_trunc_kpc, halo_cusp
            NFW halo parameters in physical units.
        disk_mass_msun, disk_scale_length_kpc, disk_outer_radius_kpc,
        disk_scale_height_kpc, disk_trunc_width_kpc
            Exponential disk parameters in physical units.
        sigma_r0_kms, sigma_r_scale_kpc
            Disk kinematics for ``diskdf``.
        grid_dr_kpc, grid_nr, grid_lmax
            Multipole solver grid.

        Returns
        -------
        GalaxyModel
            Model ready for :func:`~galacticsics.potential.solver.solve_potential`.

        Examples
        --------
        >>> m = GalaxyModel.from_physical(disk_mass_msun=4e10, halo_v0_kms=220)
        >>> m.halo.v0
        2.2
        """
        return cls(
            halo=NFWHalo(
                r_outer=units.length_from_kpc(halo_r_outer_kpc),
                v0=units.velocity_from_kms(halo_v0_kms),
                a=units.length_from_kpc(halo_a_kpc),
                dr_trunc=units.length_from_kpc(halo_dr_trunc_kpc),
                cusp=halo_cusp,
                enabled=halo_enabled,
            ),
            disk=ExponentialDisk(
                mass=units.mass_from_msun(disk_mass_msun),
                scale_length=units.length_from_kpc(disk_scale_length_kpc),
                outer_radius=units.length_from_kpc(disk_outer_radius_kpc),
                scale_height=units.length_from_kpc(disk_scale_height_kpc),
                trunc_width=units.length_from_kpc(disk_trunc_width_kpc),
                enabled=disk_enabled,
            ),
            grid=PotentialGrid(
                dr=units.length_from_kpc(grid_dr_kpc),
                nr=grid_nr,
                lmax=grid_lmax,
            ),
            disk_kinematics=DiskKinematics(
                sigma_r0=units.velocity_from_kms(sigma_r0_kms),
                sigma_r_scale=units.length_from_kpc(sigma_r_scale_kpc),
            ),
        )

    @classmethod
    def milky_way_disk_halo(cls) -> GalaxyModel:
        """Milky Way disk+halo parameters with the full production grid (``nr=20000``)."""
        return cls(
            halo=NFWHalo(r_outer=200.0, v0=3.7, a=33.0, dr_trunc=12.0, cusp=1.0, enabled=True),
            disk=ExponentialDisk(
                mass=17.0,
                scale_length=2.5,
                outer_radius=20.25,
                scale_height=0.25,
                trunc_width=3.0,
                hole_radius=0.0,
                core_radius=1.5,
                enabled=True,
            ),
            grid=PotentialGrid(dr=0.02, nr=20000, lmax=6),
            disk_kinematics=DiskKinematics(sigma_r0=0.85, sigma_r_scale=2.8, n_radial_steps=50, n_iterations=10),
        )

    @classmethod
    def reference_disk_halo(cls) -> GalaxyModel:
        """
        Reference disk+halo model for artifact generation and tests.

        Same disk/halo *physical* parameters as :meth:`milky_way_disk_halo` but
        a coarse grid (``nr=800``, ``dr=0.1`` kpc, ``lmax=2``) and
        ``r_outer=80`` kpc so ``dbh`` completes in seconds during
        ``make install-dev``.
        """
        base = cls.milky_way_disk_halo()
        return replace(
            base,
            halo=replace(base.halo, r_outer=80.0),
            grid=PotentialGrid(dr=0.1, nr=800, lmax=2),
        )

    @classmethod
    def nfw_halo_only(cls) -> GalaxyModel:
        """
        Spherical NFW halo test fixture.

        Returns
        -------
        GalaxyModel
            Single-component halo with ``lmax=0`` (spherical harmonics only)
            and ``nr=2000`` (minimum practical grid for the legacy solver).

        Notes
        -----
        The legacy ``dbh`` executable exits if ``nr`` is too small for the
        chosen halo extent. Use ``nr >= 2000`` for ``r_outer=100`` kpc.
        """
        return cls(
            halo=NFWHalo(r_outer=100.0, v0=2.0, a=10.0, dr_trunc=8.0, cusp=1.0, enabled=True),
            grid=PotentialGrid(dr=0.05, nr=2000, lmax=0),
        )

    @classmethod
    def sersic_bulge_only(cls) -> GalaxyModel:
        """Spherical Sersic bulge test fixture."""
        return cls(
            bulge=SersicBulge(n_sersic=4.0, ppp=0.5, v0=2.5, a=0.5, enabled=True),
            grid=PotentialGrid(dr=0.02, nr=5000, lmax=4),
        )

    @classmethod
    def nfw_plus_bulge(cls) -> GalaxyModel:
        """Combined spherical-ish halo + bulge fixture."""
        return cls(
            halo=NFWHalo(r_outer=100.0, v0=2.5, a=15.0, dr_trunc=10.0, enabled=True),
            bulge=SersicBulge(n_sersic=4.0, ppp=0.5, v0=1.5, a=0.4, enabled=True),
            grid=PotentialGrid(dr=0.02, nr=8000, lmax=4),
        )

    def with_halo_only(self) -> GalaxyModel:
        """
        Copy with only the halo enabled for step 1 of the halo-first workflow.

        Returns
        -------
        GalaxyModel
            Disk, second disk, gas, bulge, and black hole are disabled
            (``enabled=False``) so :func:`~galacticsics.io.legacy_inputs.write_dbh_input`
            emits ``n`` for each baryon prompt in ``dbh.f``.
        """
        return replace(
            self,
            disk=_disabled_optional(self.disk),
            disk2=_disabled_optional(self.disk2),
            gas=_disabled_optional(self.gas),
            bulge=_disabled_optional(self.bulge),
            black_hole=BlackHole(mass=self.black_hole.mass, enabled=False),
        )

    def with_baryons_only(self, *, halo_grid: PotentialGrid | None = None) -> GalaxyModel:
        """
        Copy with halo disabled for step 2 of the halo-first workflow.

        Parameters
        ----------
        halo_grid : PotentialGrid, optional
            If given, replace ``grid`` so baryon and halo harmonics share the
            same ``dr``, ``nr``, and ``lmax``.

        Returns
        -------
        GalaxyModel
            Halo stdin flag off; baryon components unchanged.
        """
        out = replace(
            self,
            halo=_disabled_optional(self.halo),
        )
        if halo_grid is not None:
            out = replace(out, grid=halo_grid)
        return out


def _disabled_optional(component):
    """Return *component* with ``enabled=False``, or ``None``."""
    if component is None:
        return None
    return replace(component, enabled=False)
