r"""Normalize the potential-based stellar-disk vertical profile.

The GalactICS disk DF uses an isothermal-in-vertical-energy profile whose
shape is set by the current potential.  Its legacy midplane prefactor,
``1 / (2 z_d)``, is exact only for a ``sech^2(z / z_d)`` profile.  A central
bulge or halo changes the vertical shape, so that prefactor does not in
general preserve the requested projected surface density.

This module computes the radius-dependent prefactor

.. math::

   A(R) = \left[\int_{-30 z_d}^{30 z_d} C(R,z)\,dz\right]^{-1},

where ``C`` is the same potential-dependent vertical factor used by the
Poisson solver and particle samplers.  Multiplying ``Sigma(R) C(R,z)`` by
this prefactor makes its vertical integral equal ``Sigma(R)`` while retaining
the equilibrium-motivated vertical shape.
"""

from __future__ import annotations

import numpy as np

from galacticsics.models import GalaxyModel


_VERTICAL_LIMIT_SCALE_HEIGHTS = 30.0
_DEFAULT_QUADRATURE_ORDER = 108


def _vertical_factor(
    psi: np.ndarray,
    psi_mid: np.ndarray,
    psi_at_3zd: np.ndarray,
) -> np.ndarray:
    """Return the GalactICS potential-dependent vertical factor."""
    dpsizh = psi_mid - psi_at_3zd
    dpsi = psi_mid - psi
    coeff = np.divide(
        dpsi,
        dpsizh,
        out=np.zeros_like(dpsi),
        where=np.abs(dpsizh) > 0.0,
    )
    factor = np.zeros_like(coeff)
    valid = (coeff >= 0.0) & (coeff <= 16.0) & (np.abs(dpsizh) > 0.0)
    factor[valid] = 0.009866**coeff[valid]
    return factor


def disk_vertical_normalization_table(
    potential,
    model: GalaxyModel,
    *,
    radial_step: float | None = None,
    radial_max: float | None = None,
    quadrature_order: int = _DEFAULT_QUADRATURE_ORDER,
) -> tuple[np.ndarray, np.ndarray]:
    """Tabulate the prefactor that gives the requested disk surface density.

    Parameters
    ----------
    potential
        A :class:`~galacticsics.potential.harmonics.HarmonicPotential` or the
        current Poisson-iteration arrays accepted by ``potential_at_batch``.
    model
        Galaxy model associated with ``potential``.
    radial_step, radial_max
        Optional normalization-table spacing and extent.  The default spacing
        is half a Poisson cell and the default extent covers the disk taper.
    quadrature_order
        Gauss-Legendre order on one side of the midplane.

    Returns
    -------
    radii, normalization
        Uniform cylindrical-radius grid and values with units of inverse
        length.  ``Sigma(R) * normalization(R) * C(R,z)`` integrates to
        ``Sigma(R)`` over the same ``|z| <= 30 z_d`` support used by the
        density routine.
    """
    disk = model.disk
    if disk is None or not disk.enabled:
        return np.array([0.0], dtype=float), np.array([0.0], dtype=float)
    if disk.scale_height <= 0.0:
        raise ValueError("disk scale_height must be positive")
    if quadrature_order < 8:
        raise ValueError("quadrature_order must be at least 8")

    grid_edge = float(model.grid.nr) * float(model.grid.dr)
    if radial_max is None:
        radial_max = min(
            grid_edge,
            float(disk.outer_radius)
            + 4.0 * np.sqrt(2.0) * float(disk.trunc_width),
        )
    if radial_step is not None:
        if radial_step <= 0.0:
            raise ValueError("radial_step must be positive")
        radial_max = max(float(radial_max), float(radial_step))
        n_radial = int(np.ceil(radial_max / radial_step))
        radii = np.arange(n_radial + 1, dtype=float) * float(radial_step)
    else:
        # The normalization changes most rapidly inside one scale height.  A
        # fine central grid and coarser outer grid keep interpolation errors
        # below the vertical quadrature error without evaluating thousands of
        # unnecessary outer-disk radii.
        inner_step = max(
            min(0.25 * float(model.grid.dr), float(disk.scale_height) / 32.0),
            1.0e-4,
        )
        outer_step = max(
            min(float(model.grid.dr), float(disk.scale_height) / 4.0),
            inner_step,
        )
        # Bulge compression can change rapidly for several scale heights, not
        # only inside one z_d.  Keep the fine mesh through at least 1 kpc so
        # interpolation does not manufacture a few-percent column error near
        # the bulge/disk transition.
        inner_max = min(
            max(4.0 * float(disk.scale_height), 1.0),
            float(radial_max),
        )
        n_inner = int(np.ceil(inner_max / inner_step))
        inner = np.linspace(0.0, inner_max, n_inner + 1)
        if inner_max < float(radial_max):
            n_outer = int(np.ceil((float(radial_max) - inner_max) / outer_step))
            outer = np.linspace(inner_max, float(radial_max), n_outer + 1)[1:]
            radii = np.concatenate([inner, outer])
        else:
            radii = inner

    if disk.vertical_density_mode == "legacy_midplane":
        normalization = np.full_like(radii, 0.5 / float(disk.scale_height))
        return np.ascontiguousarray(radii), np.ascontiguousarray(normalization)

    # Resolve the narrow central profile without spending most nodes in its
    # negligible far tail.  The four intervals still cover the exact support
    # used by disk_density_psi.
    interval_boundaries = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0])
    nodes_per_interval = max(8, int(np.ceil(quadrature_order / (interval_boundaries.size - 1))))
    nodes, weights = np.polynomial.legendre.leggauss(nodes_per_interval)
    edges = interval_boundaries * float(disk.scale_height)
    node_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        node_parts.append(0.5 * (nodes + 1.0) * (hi - lo) + lo)
        weight_parts.append(0.5 * (hi - lo) * weights)
    z_nodes = np.concatenate(node_parts)
    z_weights = np.concatenate(weight_parts)

    from types import SimpleNamespace

    from galacticsics.potential.poisson.potential import potential_at_batch

    arrays = potential
    if not hasattr(potential, "lmax_active"):
        arrays = SimpleNamespace(
            apot=potential.apot,
            lmax_active=potential.lmax,
            flags=potential.flags,
        )

    nr = radii.size
    nz = z_nodes.size
    s_grid = np.repeat(radii, nz)
    z_grid = np.tile(z_nodes, nr)
    psi = potential_at_batch(arrays, model, s_grid, z_grid).reshape(nr, nz)
    psi_mid = potential_at_batch(
        arrays,
        model,
        radii,
        np.zeros_like(radii),
    )[:, None]
    psi_3zd = potential_at_batch(
        arrays,
        model,
        radii,
        np.full_like(radii, 3.0 * float(disk.scale_height)),
    )[:, None]
    factor = _vertical_factor(psi, psi_mid, psi_3zd)
    column = 2.0 * np.sum(factor * z_weights[None, :], axis=1)
    invalid = ~(np.isfinite(column) & (column > 0.0))
    if np.any(invalid):
        bad = radii[invalid]
        raise RuntimeError(
            "invalid disk vertical column normalization at cylindrical radii "
            f"{bad[:8].tolist()}"
        )
    normalization = 1.0 / column
    return np.ascontiguousarray(radii), np.ascontiguousarray(normalization)


def interpolate_disk_vertical_normalization(
    cylindrical_radius: np.ndarray | float,
    radii: np.ndarray,
    normalization: np.ndarray,
) -> np.ndarray | float:
    """Interpolate a normalization table at cylindrical radius ``R``."""
    query = np.asarray(cylindrical_radius, dtype=float)
    values = np.interp(query, radii, normalization)
    if np.ndim(cylindrical_radius) == 0:
        return float(values)
    return values


def cached_disk_vertical_normalization(potential) -> tuple[np.ndarray, np.ndarray]:
    """Return, and cache, the normalization table for a solved potential."""
    radii = getattr(potential, "disk_vertical_norm_radii", None)
    values = getattr(potential, "disk_vertical_norm_values", None)
    if radii is None or values is None:
        radii, values = disk_vertical_normalization_table(potential, potential.model)
        potential.disk_vertical_norm_radii = radii
        potential.disk_vertical_norm_values = values
    return radii, values


def disk_vertical_normalization_at(potential, cylindrical_radius: float) -> float:
    """Return the cached vertical normalization at one cylindrical radius."""
    radii, values = cached_disk_vertical_normalization(potential)
    return float(interpolate_disk_vertical_normalization(cylindrical_radius, radii, values))
