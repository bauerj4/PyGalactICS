"""Density profile overlays vs analytic models."""

from __future__ import annotations

import numpy as np

from galacticsics.models import NFWHalo
from galacticsics.potential.poisson.densities import halo_density_spherical, nfw_density


def nfw_density_profile(r: np.ndarray, halo: NFWHalo) -> np.ndarray:
    """
    Spherical NFW mass density including legacy truncation.

    Parameters
    ----------
    r : array_like, shape (N,)
        Spherical radii [kpc].
    halo : NFWHalo
        Halo parameters.

    Returns
    -------
    rho : ndarray, shape (N,)
        Mass density [GalactICS units kpc\\ :sup:`-3`].
    """
    from galacticsics.potential.poisson.densities import halo_density_spherical_array

    return halo_density_spherical_array(np.asarray(r, dtype=float), halo)


def nfw_rho0(halo: NFWHalo) -> float:
    """Central density normalization at ``r -> 0``."""
    return nfw_density(1e-6, halo)
