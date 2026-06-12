"""
Multipole harmonic potential representation.

The self-consistent potential is stored as even-order Legendre coefficients
on a radial grid, written by ``legacy/bin/dbh`` to ``dbh.dat`` and loaded by
:func:`~galacticsics.io.read_harmonic_potential`.

Attributes on :class:`HarmonicPotential` map directly to Fortran COMMON-block
arrays in ``legacy/fortran/commonblocks``:

- ``apot[l, ir]`` — potential harmonics
- ``fr[l, ir]`` — radial force gradient harmonics
- ``adens[l, ir]`` — density harmonics

See Also
--------
galacticsics.potential.evaluate
galacticsics.potential.solver.solve_potential
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from galacticsics.models import GalaxyModel


@dataclass(frozen=True)
class ComponentFlags:
    disk: bool = False
    disk2: bool = False
    gas: bool = False
    bulge: bool = False
    halo: bool = False
    black_hole: bool = False


@dataclass
class HarmonicPotential:
    """Self-consistent multipole potential from dbh (dbh.dat)."""

    model: GalaxyModel
    psi0: float
    haloconst: float
    bulgeconst: float
    psic: float
    psid: float
    flags: ComponentFlags
    radii: np.ndarray  # shape (nr+1,)
    adens: np.ndarray  # shape (n_harm, nr+1)
    apot: np.ndarray  # shape (n_harm, nr+1)
    fr: np.ndarray  # shape (n_harm, nr+1)

    @property
    def dr(self) -> float:
        return self.model.grid.dr

    @property
    def nr(self) -> int:
        return self.model.grid.nr

    @property
    def lmax(self) -> int:
        return self.model.grid.lmax

    @property
    def n_harmonics(self) -> int:
        return self.lmax // 2 + 1

    @property
    def r_edge(self) -> float:
        return self.nr * self.dr

    def plcon(self, ell: int) -> float:
        """Normalization sqrt((2l+1)/(4 pi))."""
        return float(np.sqrt((2 * ell + 1) / (4.0 * np.pi)))
