"""Composite galaxy initial conditions (halo + bulge + disk)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ntropy.ics.disk import ExponentialDiskParams, sample_exponential_disk
from ntropy.ics.nfw import NFWParams, sample_nfw
from ntropy.ics.sersic import SersicParams, sample_sersic
from ntropy.particles import ParticleState


@dataclass
class CompositeICSpec:
    """
    Specification for a multi-component initial condition.

    Parameters
    ----------
    halo : NFWParams or None
        Truncated NFW halo parameters.
    bulge : SersicParams or None
        Spherical Sersic bulge parameters.
    disk : ExponentialDiskParams or None
        Exponential disk parameters.
    """

    halo: NFWParams | None = None
    bulge: SersicParams | None = None
    disk: ExponentialDiskParams | None = None


def _small_defaults() -> CompositeICSpec:
    """Small-N defaults for stability testing."""
    return CompositeICSpec(
        halo=NFWParams(n_particles=128, mass=80.0, eps=0.05),
        bulge=SersicParams(n_particles=64, mass=8.0, eps=0.02),
        disk=ExponentialDiskParams(n_particles=128, mass=15.0, eps=0.02),
    )


def sample_composite(
    spec: CompositeICSpec | None = None,
    *,
    seed: int = 42,
    components: frozenset[str] | None = None,
) -> ParticleState:
    """
    Sample and merge multiple stellar components into one particle set.

    Each enabled component is sampled independently with a deterministic
    sub-seed derived from ``seed`` and the component name.

    Parameters
    ----------
    spec : CompositeICSpec or None
        Per-component parameters.  When ``None``, small test defaults are used.
    seed : int
        Master random seed.
    components : frozenset of str, optional
        Subset of ``{'halo', 'bulge', 'disk'}`` to include.  When ``None``,
        all non-``None`` fields in ``spec`` are included.

    Returns
    -------
    state : ParticleState
        Combined particle set with center-of-mass removed.

    Raises
    ------
    ValueError
        If no components are requested or available.
    """
    base = spec or _small_defaults()
    active = components
    if active is None:
        active = frozenset(
            name
            for name, params in (
                ("halo", base.halo),
                ("bulge", base.bulge),
                ("disk", base.disk),
            )
            if params is not None
        )

    _seed_offsets = {"bulge": 1, "disk": 2, "halo": 3}
    parts: list[tuple[str, ParticleState]] = []
    for name in sorted(active):
        sub_seed = seed + 1000 * _seed_offsets[name]
        if name == "halo":
            if base.halo is None:
                raise ValueError("halo requested but halo params are None")
            parts.append((name, sample_nfw(base.halo, seed=sub_seed)))
        elif name == "bulge":
            if base.bulge is None:
                raise ValueError("bulge requested but bulge params are None")
            parts.append((name, sample_sersic(base.bulge, seed=sub_seed)))
        elif name == "disk":
            if base.disk is None:
                raise ValueError("disk requested but disk params are None")
            parts.append((name, sample_exponential_disk(base.disk, seed=sub_seed)))
        else:
            raise ValueError(f"Unknown component {name!r}")

    if not parts:
        raise ValueError("No components to sample")

    pos = np.vstack([p.pos for _, p in parts])
    vel = np.vstack([p.vel for _, p in parts])
    mass = np.concatenate([p.mass for _, p in parts])
    eps = np.concatenate([p.eps for _, p in parts])
    tags = np.concatenate([[name] * p.n for name, p in parts])
    state = ParticleState.from_arrays(pos, vel, mass, eps)
    state.tags = tags
    state.remove_center_of_mass()
    return state
