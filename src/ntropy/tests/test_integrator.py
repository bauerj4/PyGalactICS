"""Tests for leapfrog integrator and energy behavior."""

from __future__ import annotations

import numpy as np

from ntropy.config import IntegratorConfig, RunConfig
from ntropy.ics.plummer import sample_plummer
from ntropy.integrators.leapfrog import leapfrog1_step, leapfrog_step
from ntropy.simulation import Simulation
from ntropy.softening import total_energy


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
