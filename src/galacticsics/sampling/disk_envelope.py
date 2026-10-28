"""Safe rejection envelope for the potential-dependent stellar disk."""

from __future__ import annotations

import math

import numpy as np


def disk_rejection_envelope(
    potential,
    *,
    radial_limit: float,
    proposal_scale_length: float,
    proposal_scale_height: float,
    safety_factor: float = 1.25,
) -> tuple[float, float]:
    """Return conservative upper/lower bounds for disk rejection sampling.

    The proposal is exponential in cylindrical radius and ``sech^2`` in
    height.  A bulge-compressed, column-normalized disk need not have its
    largest target/proposal ratio at ``z=0``, so scan both dimensions.
    """
    disk = potential.model.disk
    if disk is None or not disk.enabled:
        raise ValueError("disk component missing")

    from galacticsics.distribution.diskdf_solve import _disk_midplane_density
    from galacticsics.potential.poisson.disk_vertical import (
        cached_disk_vertical_normalization,
    )

    norm_radii, _ = cached_disk_vertical_normalization(potential)
    if (
        disk.vertical_density_mode == "surface_normalized"
        and norm_radii[-1] + 1.0e-12 < radial_limit
    ):
        raise ValueError(
            "Poisson grid does not cover the normalized disk sampler: "
            f"normalization edge={norm_radii[-1]:g}, "
            f"sampler edge={radial_limit:g}"
        )

    radial_grid = norm_radii[norm_radii <= radial_limit]
    if radial_grid.size == 0 or radial_grid[-1] < radial_limit:
        radial_grid = np.append(radial_grid, radial_limit)

    # The sampler draws v uniformly and maps z = zd * atanh(v), clipping
    # |v| at 0.999.  Sampling uniformly in v therefore resolves both the
    # midplane and the farthest height the proposal can actually generate.
    v_grid = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 0.95, 49),
                np.linspace(0.95, 0.999, 49),
            ]
        )
    )
    height_grid = proposal_scale_height * np.arctanh(v_grid)

    maximum_ratio = 0.0
    for radius in radial_grid:
        radial_proposal = math.exp(-float(radius) / proposal_scale_length)
        for height in height_grid:
            vertical_proposal = 1.0 / math.cosh(
                float(height) / proposal_scale_height
            ) ** 2
            proposal = max(radial_proposal * vertical_proposal, 1.0e-300)
            ratio = _disk_midplane_density(
                float(radius), float(height), potential
            ) / proposal
            if not math.isfinite(ratio) or ratio < 0.0:
                raise RuntimeError(
                    "invalid disk rejection ratio at "
                    f"R={radius:g}, z={height:g}: {ratio!r}"
                )
            maximum_ratio = max(maximum_ratio, float(ratio))

    if maximum_ratio <= 0.0:
        raise RuntimeError("disk rejection envelope is non-positive")
    upper = safety_factor * maximum_ratio
    return upper, 1.0e-10 * upper
