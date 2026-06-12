"""Spherical density profile analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DensityProfile:
    """
    Binned spherical density profile.

    Attributes
    ----------
    r_mid : ndarray, shape (n_bins,)
        Shell center radii [kpc].
    rho : ndarray, shape (n_bins,)
        Estimated density per shell.
    counts : ndarray, shape (n_bins,)
        Particle count per shell.
    """

    r_mid: np.ndarray
    rho: np.ndarray
    counts: np.ndarray


def bin_spherical_density(
    pos: np.ndarray,
    mass: np.ndarray,
    n_bins: int = 20,
    r_max: float | None = None,
) -> DensityProfile:
    """
    Bin particle mass into spherical shells to estimate ρ(r).

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Particle positions [kpc].
    mass : ndarray, shape (N,)
        Particle masses.
    n_bins : int
        Number of logarithically uniform spherical shells.
    r_max : float, optional
        Outer binning radius.  Defaults to the maximum particle radius.

    Returns
    -------
    profile : DensityProfile
        Binned density estimate.
    """
    r = np.sqrt(np.sum(pos * pos, axis=1))
    if r_max is None:
        r_max = float(r.max()) if len(r) else 1.0
    if r_max <= 0:
        r_max = 1.0
    edges = np.linspace(0.0, r_max, n_bins + 1)
    shell_mass = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        mask = (r >= edges[i]) & (r < edges[i + 1])
        shell_mass[i] = mass[mask].sum()
        counts[i] = int(mask.sum())
    volumes = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    volumes = np.maximum(volumes, 1e-30)
    rho = shell_mass / volumes
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    return DensityProfile(r_mid=r_mid, rho=rho, counts=counts)


def compare_density_profiles(
    initial: DensityProfile,
    final: DensityProfile,
    *,
    min_count: int = 1,
) -> float:
    """
    Maximum relative deviation between two spherical density profiles.

    Parameters
    ----------
    initial, final : DensityProfile
        Profiles to compare (must share compatible binning).
    min_count : int
        Minimum particle count per shell for inclusion.

    Returns
    -------
    max_rel : float
        Maximum ``|ρ_final - ρ_initial| / ρ_initial`` over valid shells.
    """
    max_rel = 0.0
    for i in range(min(len(initial.rho), len(final.rho))):
        if initial.counts[i] < min_count or final.counts[i] < min_count:
            continue
        ref = max(initial.rho[i], 1e-30)
        rel = abs(final.rho[i] - initial.rho[i]) / ref
        max_rel = max(max_rel, rel)
    return max_rel


def theoretical_profile_from_model(
    r: np.ndarray,
    model: str,
    params: dict,
) -> np.ndarray:
    """
    Evaluate a theoretical spherical density at radii ``r``.

    Parameters
    ----------
    r : ndarray
        Radii [kpc].
    model : {'plummer', 'nfw', 'sersic'}
        Density model name.
    params : dict
        Model-specific parameters (``mass``, ``a``, ``rho0``, ``r_e``, ``n``, …).

    Returns
    -------
    rho : ndarray
        Mass density at each radius.
    """
    if model == "plummer":
        m = params["mass"]
        a = params["a"]
        return (3.0 * m / (4.0 * np.pi * a**3)) * (1.0 + (r / a) ** 2) ** (-2.5)
    if model == "nfw":
        rho0 = params["rho0"]
        a = params["a"]
        r_safe = np.maximum(r, 1e-12)
        return rho0 / (r_safe / a * (1.0 + r_safe / a) ** 2)
    if model == "sersic":
        rho0 = params["rho0"]
        r_e = params["r_e"]
        n = params["n"]
        r_safe = np.maximum(r, 1e-12)
        return rho0 * r_safe ** (-1.0 / n) * np.exp(-(r_safe / r_e) ** (1.0 / n))
    raise ValueError(f"Unknown model {model!r}")
