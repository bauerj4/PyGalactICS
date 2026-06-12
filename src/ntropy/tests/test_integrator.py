"""Tests for leapfrog integrator and energy behavior."""

from __future__ import annotations

import numpy as np

from ntropy.config import IntegratorConfig, RunConfig
from ntropy.ics.plummer import sample_plummer
from ntropy.simulation import Simulation
from ntropy.softening import total_energy


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
