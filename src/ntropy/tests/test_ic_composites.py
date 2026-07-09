"""Stability tests for composite halo / bulge / disk initial conditions."""

from __future__ import annotations

import numpy as np
import pytest

from ntropy.analysis.density import bin_spherical_density, compare_density_profiles
from ntropy.analysis.disk_density import bin_midplane_surface_density, compare_surface_density
from ntropy.config import ForceConfig, IntegratorConfig, ParallelConfig, RunConfig
from ntropy.ics.composite import CompositeICSpec, sample_composite
from ntropy.ics.disk import ExponentialDiskParams
from ntropy.ics.nfw import NFWParams
from ntropy.ics.sersic import SersicParams
from ntropy.simulation import Simulation

MAX_REL_SPHERICAL = 0.30
MAX_REL_DISK_SIGMA = 0.45

SMALL_SPEC = CompositeICSpec(
    halo=NFWParams(n_particles=96, mass=60.0, a=8.0, r_trunc=50.0, eps=0.05),
    bulge=SersicParams(n_particles=48, mass=6.0, r_e=0.4, eps=0.02),
    disk=ExponentialDiskParams(
        n_particles=256,
        mass=12.0,
        scale_length=2.5,
        outer_radius=20.0,
        scale_height=0.25,
        trunc_width=1.5,
        eps=0.02,
    ),
)

# Disk-only is excluded: exponential disks are not spherically symmetric.
SPHERICAL_COMBOS = [
    frozenset({"halo"}),
    frozenset({"bulge"}),
    frozenset({"halo", "bulge"}),
    frozenset({"halo", "disk"}),
    frozenset({"bulge", "disk"}),
    frozenset({"halo", "bulge", "disk"}),
]

DISK_SIGMA_COMBOS = [
    frozenset({"disk"}),
    frozenset({"halo", "disk"}),
    frozenset({"bulge", "disk"}),
    frozenset({"halo", "bulge", "disk"}),
]


def _run_short(state, *, n_steps: int = 20, dt: float = 0.002):
    cfg = RunConfig()
    cfg.integrator = IntegratorConfig(dt=dt, n_steps=n_steps)
    cfg.force = ForceConfig(method="brute", theta=0.4)
    cfg.parallel = ParallelConfig(enabled=False)
    cfg.output.write_final = False
    cfg.output.every = 0
    return Simulation(cfg, state=state.copy()).run()


def _disk_subset(state):
    """Return disk-tagged particles for surface-density analysis."""
    if state.tags is None:
        return state
    return state.mask("disk")


@pytest.mark.parametrize("components", SPHERICAL_COMBOS, ids=lambda c: "+".join(sorted(c)))
def test_composite_spherical_density_stability(components):
    """Spherical ρ(r) should not drift excessively over a short self-gravitating run."""
    state = sample_composite(SMALL_SPEC, seed=42, components=components)
    r_max = float(np.sqrt(np.sum(state.pos**2, axis=1)).max()) * 1.2
    init_prof = bin_spherical_density(state.pos, state.mass, n_bins=12, r_max=r_max)
    result = _run_short(state, n_steps=18, dt=0.002)
    final_prof = bin_spherical_density(
        result.final_state.pos,
        result.final_state.mass,
        n_bins=12,
        r_max=r_max,
    )
    max_rel = compare_density_profiles(init_prof, final_prof, min_count=2)
    assert max_rel < MAX_REL_SPHERICAL, (
        f"Components {components}: spherical drift {max_rel:.3f}"
    )


@pytest.mark.parametrize(
    "components",
    DISK_SIGMA_COMBOS,
    ids=lambda c: "sigma-" + "+".join(sorted(c)),
)
def test_exponential_disk_surface_density_stability(components):
    """Midplane Σ(R) for disk particles stays near the legacy target."""
    state = sample_composite(SMALL_SPEC, seed=42, components=components)
    disk_params = SMALL_SPEC.disk
    assert disk_params is not None
    disk_state = _disk_subset(state)

    r_max = disk_params.outer_radius * 0.85
    init_sigma = bin_midplane_surface_density(
        disk_state.pos, disk_state.mass, n_bins=10, r_max=r_max
    )
    init_err = compare_surface_density(init_sigma, disk_params, min_count=5, skip_edges=2)
    assert init_err < MAX_REL_DISK_SIGMA, f"Initial disk Σ(R) mismatch {init_err:.3f}"

    result = _run_short(state, n_steps=15, dt=0.002)
    final_disk = _disk_subset(result.final_state)
    final_sigma = bin_midplane_surface_density(
        final_disk.pos, final_disk.mass, n_bins=10, r_max=r_max
    )
    final_err = compare_surface_density(final_sigma, disk_params, min_count=5, skip_edges=2)
    drift = compare_density_profiles(
        _sigma_to_density_profile(init_sigma),
        _sigma_to_density_profile(final_sigma),
        min_count=3,
    )
    assert final_err < MAX_REL_DISK_SIGMA * 1.5, f"Final disk Σ(R) mismatch {final_err:.3f}"
    assert drift < MAX_REL_DISK_SIGMA, f"Disk Σ(R) drift {drift:.3f}"


def _sigma_to_density_profile(profile):
    """Adapter: treat surface density rings like DensityProfile for drift metric."""
    from ntropy.analysis.density import DensityProfile

    return DensityProfile(r_mid=profile.r_mid, rho=profile.sigma, counts=profile.counts)
