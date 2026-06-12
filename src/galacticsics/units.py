"""
GalactICS unit conventions and conversion helpers.

Internal (code) units
---------------------
All legacy solvers and the Python API use a single self-consistent system with
``G = 1``:

| Quantity | Internal unit | Physical meaning |
|----------|---------------|------------------|
| Length | 1 | 1 kpc |
| Velocity | 1 | 100 km/s |
| Mass | 1 | :math:`M_G = 2.325\\times 10^9\\,M_\\odot` |
| Time | 1 | :math:`\\sqrt{G M_G / \\mathrm{kpc}} \\approx 13.2\\ \\mathrm{Myr}` |

The mass unit follows from ``G=1`` with kpc and ``(100 km/s)^2``.

Physical (user) units
---------------------
Specify masses in solar masses, velocities in km/s, and lengths in kpc via
:func:`mass_from_msun`, :func:`velocity_from_kms`, and
:meth:`~galacticsics.models.GalaxyModel.from_physical`.  Internal storage always
uses GalactICS units.

Examples
--------
>>> from galacticsics.units import UnitSystem, DEFAULT_UNITS, mass_to_msun
>>> DEFAULT_UNITS.mass_msun
2325000000.0
>>> mass_to_msun(17.0)  # disk mass in internal units
39525000000.0
>>> from galacticsics.models import GalaxyModel
>>> m = GalaxyModel.from_physical(disk_mass_msun=3.95e10, halo_v0_kms=370)
>>> m.disk.mass
17.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# GalactICS mass unit [M_sun]
MASS_UNIT_MSUN = 2.325e9

# Velocity unit [km/s]
VELOCITY_UNIT_KMS = 100.0

# Length unit [kpc]
LENGTH_UNIT_KPC = 1.0

# Gravitational constant in GalactICS units (G=1 by convention)
G_GALACTICS = 1.0


@dataclass(frozen=True)
class UnitSystem:
    """
    Unit mapping between GalactICS internal values and physical units.

    Parameters
    ----------
    length_kpc : float
        One internal length unit in kpc.  Default 1.
    velocity_kms : float
        One internal velocity unit in km/s.  Default 100.
    mass_msun : float
        One internal mass unit in solar masses.  Default ``2.325e9``.
    G : float
        Gravitational constant in internal units.  Default 1 (GalactICS convention).

    Notes
    -----
    Changing ``mass_msun`` or ``velocity_kms`` independently breaks the
    ``G=1`` relation unless you are deliberately exploring alternate scalings.
    Use :data:`DEFAULT_UNITS` for faithfulness to the legacy code.
    """

    length_kpc: float = LENGTH_UNIT_KPC
    velocity_kms: float = VELOCITY_UNIT_KMS
    mass_msun: float = MASS_UNIT_MSUN
    G: float = G_GALACTICS

    def length_to_kpc(self, x: float) -> float:
        """Convert internal length to kpc."""
        return x * self.length_kpc

    def length_from_kpc(self, x_kpc: float) -> float:
        """Convert kpc to internal length."""
        return x_kpc / self.length_kpc

    def velocity_to_kms(self, v: float) -> float:
        """Convert internal velocity to km/s."""
        return v * self.velocity_kms

    def velocity_from_kms(self, v_kms: float) -> float:
        """Convert km/s to internal velocity."""
        return v_kms / self.velocity_kms

    def mass_to_msun(self, m: float) -> float:
        """Convert internal mass to solar masses."""
        return m * self.mass_msun

    def mass_from_msun(self, m_msun: float) -> float:
        """Convert solar masses to internal mass."""
        return m_msun / self.mass_msun


DEFAULT_UNITS = UnitSystem()


def mass_to_msun(mass_galactics: float, *, units: UnitSystem = DEFAULT_UNITS) -> float:
    """
    Convert GalactICS mass to solar masses.

    Parameters
    ----------
    mass_galactics : float
        Mass in GalactICS units (as stored in ``mr.dat`` and component models).
    units : UnitSystem, optional
        Unit mapping.  Default :data:`DEFAULT_UNITS`.

    Returns
    -------
    float
        Mass in solar masses.
    """
    return units.mass_to_msun(mass_galactics)


def mass_from_msun(mass_msun: float, *, units: UnitSystem = DEFAULT_UNITS) -> float:
    """Convert solar masses to GalactICS mass units."""
    return units.mass_from_msun(mass_msun)


def velocity_to_kms(v_galactics: float, *, units: UnitSystem = DEFAULT_UNITS) -> float:
    """Convert GalactICS velocity to km/s."""
    return units.velocity_to_kms(v_galactics)


def velocity_from_kms(v_kms: float, *, units: UnitSystem = DEFAULT_UNITS) -> float:
    """Convert km/s to GalactICS velocity units."""
    return units.velocity_from_kms(v_kms)


def length_to_kpc(r_galactics: float, *, units: UnitSystem = DEFAULT_UNITS) -> float:
    """Convert internal length to kpc."""
    return units.length_to_kpc(r_galactics)


def length_from_kpc(r_kpc: float, *, units: UnitSystem = DEFAULT_UNITS) -> float:
    """Convert kpc to internal length."""
    return units.length_from_kpc(r_kpc)


def circular_velocity_kms(potential_at_rc: float, radius_kpc: float) -> float:
    """
    Circular velocity [km/s] from the radial force at cylindrical radius R.

    Uses the legacy relation implemented in ``fitvc.f``:

    .. math::

        v_c^2 = -F_R \\, R

    Parameters
    ----------
    potential_at_rc : float
        Unused placeholder kept for API compatibility.
    radius_kpc : float
        Cylindrical radius [kpc].

    Returns
    -------
    float
        Circular velocity in km/s.

    Notes
    -----
    Prefer calling :func:`~galacticsics.potential.evaluate_force` directly and
    computing ``sqrt(max(0, -fr * R))`` when working in GalactICS units, then
    :func:`velocity_to_kms`.
    """
    return velocity_to_kms(math.sqrt(max(0.0, -potential_at_rc * radius_kpc)))
