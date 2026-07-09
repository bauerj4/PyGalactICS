"""Tests for leapfrog integrator and energy behavior."""

from __future__ import annotations

import numpy as np
import pytest

from ntropy.config import IntegratorConfig, RunConfig
from ntropy.ics.plummer import sample_plummer
from ntropy.integrators.leapfrog import leapfrog1_step, leapfrog_step
from ntropy.particles import ParticleState
from ntropy.simulation import Simulation
from ntropy.softening import total_energy
from ntropy.units import GYR_PER_CODE_TIME, code_time_to_gyr

TWO_BODY_DURATION_GYR = 10.0
# Order-2 leapfrog: notebook cadence (2500 steps/Gyr).
TWO_BODY_STEPS_PER_GYR_ORDER2 = 2500
# Order-1 needs a smaller Δt over 10 Gyr to control accumulated drift.
TWO_BODY_STEPS_PER_GYR_ORDER1 = 25_000


def _max_relative_energy_error(energies: list[float]) -> float:
    e0 = energies[0]
    denom = max(abs(e0), 1e-30)
    return max(abs(e - e0) / denom for e in energies)


def _two_body_kepler_state(
    *,
    semi_major: float = 0.2,
    eccentricity: float = 0.0,
    mass: float = 1.0,
    eps_frac: float = 0.1,
) -> ParticleState:
    """
    Equal-mass softened two-body binary in the x-y plane.

    ``eccentricity = 0`` places the stars on a circular orbit with separation
    ``semi_major``.  For ``eccentricity > 0`` the stars start at apocenter
    (purely tangential motion) on a mildly eccentric Kepler ellipse.
    """
    if not 0.0 <= eccentricity < 1.0:
        raise ValueError(f"eccentricity must be in [0, 1), got {eccentricity}")

    eps = eps_frac * semi_major
    separation = semi_major if eccentricity == 0.0 else semi_major * (1.0 + eccentricity)
    half = 0.5 * separation
    total_mass = 2.0 * mass
    v_rel = np.sqrt(total_mass * (2.0 / separation - 1.0 / semi_major))
    speed = 0.5 * v_rel

    return ParticleState(
        pos=np.array([[-half, 0.0, 0.0], [half, 0.0, 0.0]], dtype=float),
        vel=np.array([[0.0, speed, 0.0], [0.0, -speed, 0.0]], dtype=float),
        mass=np.array([mass, mass], dtype=float),
        eps=np.array([eps, eps], dtype=float),
    )


def _two_body_timestep_and_tolerance(order: int) -> tuple[float, int, float]:
    """Return (dt, n_steps, max_allowed_rel_energy_error) for a 10 Gyr run."""
    if order == 1:
        steps_per_gyr = TWO_BODY_STEPS_PER_GYR_ORDER1
        max_rel = 1e-2
    elif order == 2:
        steps_per_gyr = TWO_BODY_STEPS_PER_GYR_ORDER2
        max_rel = 5e-2
    else:
        raise ValueError(f"unsupported leapfrog order: {order}")

    dt = (1.0 / steps_per_gyr) / GYR_PER_CODE_TIME
    n_steps = int(TWO_BODY_DURATION_GYR * steps_per_gyr)
    return dt, n_steps, max_rel


def test_leapfrog1_step_matches_symplectic_euler():
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    vel = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    acc = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]])
    dt = 0.1
    pos_new, vel_new = leapfrog1_step(pos, vel, acc, dt)
    np.testing.assert_allclose(vel_new, vel + dt * acc)
    np.testing.assert_allclose(pos_new, pos + dt * vel_new)


def test_leapfrog2_step_is_velocity_verlet_half_kick():
    pos = np.array([[0.0, 0.0, 0.0]])
    vel = np.array([[1.0, 0.0, 0.0]])
    acc = np.array([[0.0, -2.0, 0.0]])
    dt = 0.2
    pos_new, vel_half = leapfrog_step(pos, vel, acc, dt)
    np.testing.assert_allclose(vel_half, vel + 0.5 * dt * acc)
    np.testing.assert_allclose(pos_new, pos + dt * vel_half)


@pytest.mark.parametrize(
    ("orbit", "eccentricity"),
    [
        pytest.param("circular", 0.0, id="circular"),
        pytest.param("eccentric", 0.1, id="eccentric"),
    ],
)
@pytest.mark.parametrize("order", [1, 2])
def test_two_body_symplectic_energy_preserved(
    orbit: str,
    eccentricity: float,
    order: int,
):
    """
    Leapfrog should conserve the softened two-body Hamiltonian over 10 Gyr.

    Uses brute force (exact for N=2) and thousands of orbital periods at
    semi-major axis 0.2 kpc (period ~4 Myr in GalactICS units).
    """
    state = _two_body_kepler_state(eccentricity=eccentricity)
    dt, n_steps, max_rel = _two_body_timestep_and_tolerance(order)

    cfg = RunConfig()
    cfg.integrator = IntegratorConfig(
        type="leapfrog",
        order=order,
        dt=dt,
        n_steps=n_steps,
    )
    cfg.force.method = "brute"
    cfg.parallel.enabled = False
    cfg.output.write_final = False

    run = Simulation(cfg, state=state).run()
    rel = _max_relative_energy_error(run.energies)
    span_gyr = code_time_to_gyr(dt * n_steps)

    assert rel < max_rel, (
        f"{orbit} leapfrog-{order} over {span_gyr:.1f} Gyr "
        f"({n_steps} steps): max |ΔE/E0| = {rel:.3e}"
    )


def test_plummer_energy_drift_bounded():
    state = sample_plummer(seed=42)
    cfg = RunConfig()
    cfg.integrator = IntegratorConfig(dt=0.002, n_steps=30)
    cfg.force.method = "brute"
    cfg.parallel.enabled = False
    sim = Simulation(cfg, state=state)
    e0 = total_energy(state.pos, state.vel, state.mass, state.eps)
    for _ in range(cfg.integrator.n_steps):
        sim.step()
    e1 = total_energy(sim.state.pos, sim.state.vel, sim.state.mass, sim.state.eps)
    rel_drift = abs(e1 - e0) / max(abs(e0), 1e-30)
    assert rel_drift < 0.5
