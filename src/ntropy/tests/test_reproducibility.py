"""Reproducibility tests with fixed random seed."""

from __future__ import annotations

import numpy as np

from ntropy.config import IntegratorConfig, RunConfig
from ntropy.ics.plummer import sample_plummer
from ntropy.simulation import Simulation


def test_same_seed_same_final_positions():
    cfg = RunConfig()
    cfg.integrator = IntegratorConfig(dt=0.005, n_steps=10)
    cfg.force.method = "brute"
    cfg.parallel.enabled = False
    cfg.output.write_final = False
    cfg.output.every = 0

    state1 = sample_plummer(seed=42)
    state2 = sample_plummer(seed=42)
    r1 = Simulation(cfg, state=state1).run()
    r2 = Simulation(cfg, state=state2).run()
    np.testing.assert_allclose(r1.final_state.pos, r2.final_state.pos, rtol=0, atol=0)
