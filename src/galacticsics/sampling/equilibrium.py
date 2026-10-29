"""Spatial sampling profiles for isotropic equilibrium distribution functions.

For an isotropic spheroid with distribution function ``f(E)``, the spatial
density is ``rho(Psi)``.  Once a disk is added, equipotential surfaces are not
exactly spherical, so drawing positions from the original spherical NFW or
Sersic profile is inconsistent with both the converged Poisson solution and
the velocity distribution.

This module tabulates the final ``rho(Psi(R, z))`` field as an efficient
spherical-coordinate rejection proposal.  The radial proposal is proportional
to ``r^2 max_mu rho(Psi(r, mu))``; only the comparatively small angular
variation is left to rejection sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from galacticsics.potential.evaluate import evaluate_potential
from galacticsics.potential.harmonics import HarmonicPotential
from galacticsics.potential.poisson.densities import (
    bulge_density_psi_array,
    halo_density_from_psi_array,
)


@dataclass(frozen=True)
class EquilibriumSpatialProfile:
    """Tabulated proposal and mass for one spheroidal component."""

    component: str
    radius: np.ndarray
    envelope: np.ndarray
    radial_cdf: np.ndarray
    psi: np.ndarray
    density: np.ndarray
    mass: float


def _read_density_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = np.loadtxt(path, dtype=float)
    if table.ndim != 2 or table.shape[1] < 2 or table.shape[0] < 2:
        raise ValueError(f"invalid density-potential table: {path}")
    return table[:, 0], table[:, 1]


def _density_evaluator(
    potential: HarmonicPotential,
    component: str,
    energies: np.ndarray,
    densities: np.ndarray,
):
    if component == "halo":

        def evaluate(psi: np.ndarray) -> np.ndarray:
            values = np.asarray(psi, dtype=float)
            result = np.zeros_like(values)
            active = values >= potential.psic
            if np.any(active):
                result[active] = halo_density_from_psi_array(
                    np.minimum(values[active], potential.psi0),
                    energies,
                    densities,
                    psi0=potential.psi0,
                )
            return result

        return evaluate

    if component == "bulge":

        def evaluate(psi: np.ndarray) -> np.ndarray:
            values = np.asarray(psi, dtype=float)
            return bulge_density_psi_array(
                np.minimum(values, potential.psi0),
                energies,
                densities,
                psi0=potential.psi0,
                psic=potential.psic,
                psid=potential.psid,
            )

        return evaluate

    raise ValueError(f"unsupported equilibrium component: {component}")


def build_equilibrium_spatial_profile(
    potential: HarmonicPotential,
    density_table: Path,
    *,
    component: str,
    n_radius: int | None = None,
    n_mu: int = 33,
    n_psi: int = 4097,
    envelope_margin: float = 1.10,
) -> EquilibriumSpatialProfile:
    """Build the final-potential spatial proposal for ``halo`` or ``bulge``.

    The returned mass is the integral of the same ``rho(Psi)`` field used by
    the converged Poisson solve, over the solved radial domain.  Consequently,
    particle masses and positions obey the same density contract.
    """

    if n_mu < 3:
        raise ValueError("n_mu must be at least 3")
    if not 1.0 < envelope_margin:
        raise ValueError("envelope_margin must exceed one")

    energies, densities = _read_density_table(Path(density_table))
    density_at_psi = _density_evaluator(potential, component, energies, densities)

    # Both samplers evaluate rho through this regular Psi table.  Build the
    # rejection envelope from that same interpolation, while retaining the
    # component-specific evaluator below for the reported mass integral.
    psi_grid = np.linspace(potential.psic, potential.psi0, n_psi, dtype=np.float64)
    density = np.asarray(density_at_psi(psi_grid), dtype=np.float64)

    radial_count = n_radius or min(max(potential.nr + 1, 257), 4097)
    radial_coordinate = np.linspace(0.0, 1.0, radial_count, dtype=np.float64)
    radius = potential.r_edge * radial_coordinate * radial_coordinate
    mu = np.linspace(0.0, 1.0, n_mu, dtype=np.float64)
    angular_psi = np.empty((radial_count, n_mu), dtype=np.float64)
    angular_density = np.empty((radial_count, n_mu), dtype=np.float64)

    for i, r in enumerate(radius):
        psi_values = np.array(
            [
                evaluate_potential(
                    potential,
                    float(r * np.sqrt(max(0.0, 1.0 - m * m))),
                    float(r * m),
                )
                for m in mu
            ],
            dtype=float,
        )
        angular_psi[i] = psi_values
        angular_density[i] = density_at_psi(psi_values)

    proposal_angular_density = np.interp(
        angular_psi,
        psi_grid,
        density,
        left=0.0,
    )
    raw_envelope = np.max(proposal_angular_density, axis=1)
    # The tabulated rho(Psi) need not be monotone and can contain a narrow peak
    # between radial knots.  Bound each radial interval by every rho-table knot
    # in the Psi range spanned by its endpoints.  Assigning that bound to both
    # ends makes subsequent linear interpolation conservative throughout the
    # interval; the margin covers angular and potential interpolation error.
    interval_upper = np.zeros(max(len(radius) - 1, 0), dtype=np.float64)
    for i in range(len(interval_upper)):
        psi_lo = min(float(np.min(angular_psi[i])), float(np.min(angular_psi[i + 1])))
        psi_hi = max(float(np.max(angular_psi[i])), float(np.max(angular_psi[i + 1])))
        edge_density = max(
            float(np.interp(psi_lo, psi_grid, density, left=0.0)),
            float(np.interp(psi_hi, psi_grid, density, left=0.0)),
        )
        first = int(np.searchsorted(psi_grid, psi_lo, side="left"))
        last = int(np.searchsorted(psi_grid, psi_hi, side="right"))
        interior_density = (
            float(np.max(density[first:last])) if last > first else 0.0
        )
        interval_upper[i] = max(edge_density, interior_density)

    interval_safe_envelope = raw_envelope.copy()
    if len(interval_upper) > 0:
        interval_safe_envelope[0] = max(raw_envelope[0], interval_upper[0])
        interval_safe_envelope[-1] = max(raw_envelope[-1], interval_upper[-1])
    if len(interval_upper) > 1:
        interval_safe_envelope[1:-1] = np.maximum.reduce(
            (raw_envelope[1:-1], interval_upper[:-1], interval_upper[1:])
        )
    envelope = envelope_margin * interval_safe_envelope
    shell_envelope = radius * radius * envelope
    radial_cdf = np.zeros_like(radius)
    dr = np.diff(radius)
    radial_cdf[1:] = np.cumsum(
        0.5 * (shell_envelope[1:] + shell_envelope[:-1]) * dr
    )
    if not np.isfinite(radial_cdf[-1]) or radial_cdf[-1] <= 0.0:
        raise ValueError(f"empty {component} equilibrium spatial profile")
    radial_cdf /= radial_cdf[-1]

    # Reflection symmetry gives integral over mu in [-1, 1] as twice [0, 1].
    angular_integral = 2.0 * np.trapezoid(angular_density, mu, axis=1)
    shell_mass = radius * radius * angular_integral
    mass = float(2.0 * np.pi * np.trapezoid(shell_mass, radius))

    return EquilibriumSpatialProfile(
        component=component,
        radius=np.ascontiguousarray(radius),
        envelope=np.ascontiguousarray(envelope),
        radial_cdf=np.ascontiguousarray(radial_cdf),
        psi=np.ascontiguousarray(psi_grid),
        density=np.ascontiguousarray(density),
        mass=mass,
    )


def build_spherical_spatial_profile(
    potential: HarmonicPotential,
    *,
    component: str,
    n_radius: int = 4097,
) -> EquilibriumSpatialProfile:
    """Build a profile for a component solved directly as analytic ``rho(r)``.

    The disk-free fast solver synthesizes its monopole from the analytic halo
    and bulge densities, deliberately bypassing numerically degenerate
    ``rho(Psi)`` round trips.  Its sampling profile must use that same spatial
    source rather than the auxiliary ``denspsi`` table.
    """
    from galacticsics.potential.poisson.densities import (
        bulge_density_on_grid,
        halo_density_spherical_array,
    )

    if n_radius < 3:
        raise ValueError("n_radius must be at least 3")
    if component == "halo":
        halo = potential.model.halo
        if halo is None or not halo.enabled:
            raise ValueError("enabled halo required for spherical profile")
        radial_edge = halo.r_outer
        coordinate = np.linspace(0.0, 1.0, n_radius, dtype=np.float64)
        radius = radial_edge * coordinate * coordinate
        spatial_density = halo_density_spherical_array(radius, halo)
    elif component == "bulge":
        bulge = potential.model.bulge
        if bulge is None or not bulge.enabled:
            raise ValueError("enabled bulge required for spherical profile")
        radial_edge = 3.0 * bulge.a
        coordinate = np.linspace(0.0, 1.0, n_radius, dtype=np.float64)
        radius = radial_edge * coordinate * coordinate
        spatial_density = bulge_density_on_grid(radius, bulge)
    else:
        raise ValueError(f"unsupported equilibrium component: {component}")
    spatial_density[0] = spatial_density[1]

    # The radial CDF is already the exact spherical proposal. Unit lookup
    # tables make angular acceptance unconditional in both samplers.
    envelope = np.ones_like(radius)
    shell_envelope = radius * radius * spatial_density
    radial_cdf = np.zeros_like(radius)
    radial_cdf[1:] = np.cumsum(
        0.5 * (shell_envelope[1:] + shell_envelope[:-1]) * np.diff(radius)
    )
    if not np.isfinite(radial_cdf[-1]) or radial_cdf[-1] <= 0.0:
        raise ValueError(f"empty {component} equilibrium spatial profile")
    radial_cdf /= radial_cdf[-1]

    mass = float(
        4.0 * np.pi * np.trapezoid(radius * radius * spatial_density, radius)
    )
    return EquilibriumSpatialProfile(
        component=component,
        radius=np.ascontiguousarray(radius),
        envelope=np.ascontiguousarray(envelope),
        radial_cdf=np.ascontiguousarray(radial_cdf),
        psi=np.ascontiguousarray([potential.psic, potential.psi0]),
        density=np.ascontiguousarray([1.0, 1.0]),
        mass=mass,
    )


def write_equilibrium_spatial_profile(
    path: Path,
    profile: EquilibriumSpatialProfile,
) -> None:
    """Persist a profile for the OpenMP and Python samplers."""

    np.savez(
        path,
        component=np.asarray(profile.component),
        radius=profile.radius,
        envelope=profile.envelope,
        radial_cdf=profile.radial_cdf,
        psi=profile.psi,
        density=profile.density,
        mass=np.asarray(profile.mass, dtype=np.float64),
    )


def read_equilibrium_spatial_profile(path: Path) -> EquilibriumSpatialProfile:
    """Load a profile written by :func:`write_equilibrium_spatial_profile`."""

    with np.load(path, allow_pickle=False) as data:
        component = str(data["component"].item())
        return EquilibriumSpatialProfile(
            component=component,
            radius=np.ascontiguousarray(data["radius"], dtype=np.float64),
            envelope=np.ascontiguousarray(data["envelope"], dtype=np.float64),
            radial_cdf=np.ascontiguousarray(data["radial_cdf"], dtype=np.float64),
            psi=np.ascontiguousarray(data["psi"], dtype=np.float64),
            density=np.ascontiguousarray(data["density"], dtype=np.float64),
            mass=float(data["mass"]),
        )


def sample_equilibrium_position(
    rng: np.random.Generator,
    potential: HarmonicPotential,
    profile: EquilibriumSpatialProfile,
) -> tuple[float, float, float] | None:
    """Draw one accepted Cartesian position, or ``None`` after rejection."""

    u = rng.random()
    radius = float(np.interp(u, profile.radial_cdf, profile.radius))
    mu = 2.0 * rng.random() - 1.0
    phi = 2.0 * np.pi * rng.random()
    r_cyl = radius * np.sqrt(max(0.0, 1.0 - mu * mu))
    z = radius * mu
    psi = evaluate_potential(potential, float(r_cyl), float(z))
    density = float(np.interp(psi, profile.psi, profile.density, left=0.0))
    envelope = float(np.interp(radius, profile.radius, profile.envelope))
    if density > envelope * (1.0 + 1.0e-10):
        raise RuntimeError(
            f"{profile.component} spatial rejection envelope exceeded: "
            f"rho={density:.8g} > envelope={envelope:.8g} at r={radius:.8g}"
        )
    if envelope <= 0.0 or envelope * rng.random() > density:
        return None
    return r_cyl * np.cos(phi), r_cyl * np.sin(phi), z
