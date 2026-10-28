"""Regression tests for the final-potential spheroid sampling contract."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from galacticsics.io import read_harmonic_potential
from galacticsics.models import GalaxyModel, NFWHalo, PotentialGrid, SersicBulge
from galacticsics.physics.backend import PhysicsBackendKind
from galacticsics.potential.evaluate import evaluate_potential
from galacticsics.potential.solver import solve_potential
from galacticsics.sampling.equilibrium import read_equilibrium_spatial_profile
from galacticsics.sampling.sampler import SampleConfig, sample_galaxy


def _disk_halo_bulge_model() -> GalaxyModel:
    base = GalaxyModel.reference_disk_halo()
    assert base.disk is not None
    return replace(
        base,
        disk=replace(
            base.disk,
            mass=6.0,
            scale_length=2.0,
            scale_height=0.25,
            outer_radius=8.0,
            trunc_width=1.0,
            vertical_density_mode="surface_normalized",
        ),
        halo=NFWHalo(
            r_outer=30.0,
            v0=3.2,
            a=15.0,
            dr_trunc=5.0,
            enabled=True,
        ),
        bulge=SersicBulge(
            n_sersic=4.0,
            ppp=0.5,
            v0=1.5,
            a=0.4,
            enabled=True,
        ),
        grid=PotentialGrid(dr=0.1, nr=300, lmax=2),
    )


@pytest.fixture(scope="module")
def equilibrium_work(tmp_path_factory: pytest.TempPathFactory) -> tuple[GalaxyModel, Path]:
    model = _disk_halo_bulge_model()
    work = tmp_path_factory.mktemp("equilibrium_spatial")
    solve_potential(
        model,
        work_dir=work,
        cleanup=False,
        npsi=256,
        nint=12,
        backend=PhysicsBackendKind.PYTHON,
    )
    return model, work


@pytest.mark.physics_python
@pytest.mark.essential
def test_mr_masses_match_final_density_profiles(equilibrium_work) -> None:
    """Particle masses come from the converged rho(Psi), not analytic totals."""

    model, work = equilibrium_work
    rows = (work / "mr.dat").read_text().splitlines()
    assert float(rows[0].split()[0]) == pytest.approx(model.disk.mass)
    for component, row in (("bulge", 1), ("halo", 2)):
        profile = read_equilibrium_spatial_profile(
            work / f"{component}_spatial_profile.npz"
        )
        mass, edge = (float(value) for value in rows[row].split()[:2])
        assert mass == pytest.approx(profile.mass, rel=1e-12)
        assert edge == pytest.approx(profile.radius[-1], rel=1e-12)
        assert profile.radial_cdf[0] == 0.0
        assert profile.radial_cdf[-1] == pytest.approx(1.0)
        assert np.all(np.diff(profile.radial_cdf) >= 0.0)


def _fixed_potential_virial_ratio(data: np.ndarray, potential) -> float:
    pos = np.column_stack((data["x"], data["y"], data["z"]))
    vel = np.column_stack((data["vx"], data["vy"], data["vz"]))
    mass = data["mass"]
    kinetic = 0.5 * float(np.sum(mass * np.sum(vel * vel, axis=1)))
    eps = 1.0e-4
    virial = 0.0
    for (x, y, z), particle_mass in zip(pos, mass):
        radius = float(np.hypot(x, y))
        psi_r_plus = evaluate_potential(potential, radius + eps, float(z))
        psi_r_minus = evaluate_potential(
            potential, max(radius - eps, 0.0), float(z)
        )
        radial_denominator = 2.0 * eps if radius > eps else eps
        dpsi_dr = (psi_r_plus - psi_r_minus) / radial_denominator
        dpsi_dz = (
            evaluate_potential(potential, radius, float(z + eps))
            - evaluate_potential(potential, radius, float(z - eps))
        ) / (2.0 * eps)
        virial += float(particle_mass) * (radius * dpsi_dr + float(z) * dpsi_dz)
    return 2.0 * kinetic / abs(virial)


@pytest.mark.physics_python
@pytest.mark.essential
def test_spheroid_samples_are_in_final_potential_equilibrium(equilibrium_work) -> None:
    """Halo and bulge positions and velocities obey one final-potential DF."""

    model, work = equilibrium_work
    result = sample_galaxy(
        model,
        SampleConfig(
            n_disk=0,
            n_halo=4000,
            n_bulge=3000,
            run_diskdf=False,
            seed_halo=-701,
            seed_bulge=-702,
        ),
        work_dir=work,
        cleanup=False,
        backend=PhysicsBackendKind.PYTHON,
    )
    potential = read_harmonic_potential(work / "dbh.dat")
    for component in ("halo", "bulge"):
        ratio = _fixed_potential_virial_ratio(
            result.particles[component].data,
            potential,
        )
        assert 0.90 <= ratio <= 1.10, (component, ratio)

