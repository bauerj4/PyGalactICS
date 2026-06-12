"""Tests for Euler and Runge–Kutta integrators."""

from __future__ import annotations

import numpy as np
import pytest

from ntropy.config import IntegratorConfig, RunConfig
from ntropy.ics.plummer import sample_plummer
from ntropy.integrators.euler import euler_step
from ntropy.integrators.registry import integrator_force_evaluations, is_symplectic
from ntropy.integrators.rk import rk2_step, rk3_step, rk4_step
from ntropy.simulation import Simulation
from ntropy.softening import total_energy


def _harmonic_accel(pos: np.ndarray) -> np.ndarray:
    """Unit harmonic oscillator acceleration a = -x."""
    return -pos


def test_euler_step_forward():
    pos = np.array([[1.0, 0.0, 0.0]])
    vel = np.array([[0.0, 1.0, 0.0]])
    acc = np.array([[0.0, -1.0, 0.0]])
    pos_new, vel_new = euler_step(pos, vel, acc, dt=0.1)
    np.testing.assert_allclose(pos_new, pos + 0.1 * vel)
    np.testing.assert_allclose(vel_new, vel + 0.1 * acc)


@pytest.mark.parametrize(
    ("step_fn", "expected_order"),
    [
        (rk2_step, 2),
        (rk3_step, 3),
        (rk4_step, 4),
    ],
)
def test_rk_linear_oscillator_converges_with_order(step_fn, expected_order):
    """RK integrators should converge at the expected rate for a = -x."""
    pos0 = np.array([[1.0, 0.0, 0.0]])
    vel0 = np.array([[0.0, 0.0, 0.0]])
    dt = 0.05
    t_final = 0.5

    errors = []
    for n_sub in (1, 2, 4, 8):
        dt_sub = t_final / n_sub
        pos, vel = pos0.copy(), vel0.copy()
        for _ in range(n_sub):
            pos, vel = step_fn(pos, vel, _harmonic_accel, dt_sub)
        x_exact = np.cos(t_final)
        vx_exact = -np.sin(t_final)
        err = float(
            np.hypot(pos[0, 0] - x_exact, vel[0, 0] - vx_exact)
        )
        errors.append(err)

    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    expected_ratio = 2**expected_order
    for ratio in ratios:
        assert ratio > expected_ratio * 0.35


def _max_relative_energy_error(energies: list[float]) -> float:
    e0 = energies[0]
    denom = max(abs(e0), 1e-30)
    return max(abs(e - e0) / denom for e in energies)


def test_integrator_metadata():
    assert is_symplectic("leapfrog")
    assert not is_symplectic("euler")
    assert integrator_force_evaluations("rk4") == 4
    assert integrator_force_evaluations("leapfrog", order=2) == 2


def test_euler_energy_drift_exceeds_symplectic_leapfrog():
    """Forward Euler should accumulate larger |ΔE/E0| than leapfrog-2."""
    state = sample_plummer(seed=7)
    dt = 0.002
    n_steps = 80

    cfg_lf = RunConfig()
    cfg_lf.integrator = IntegratorConfig(type="leapfrog", order=2, dt=dt, n_steps=n_steps)
    cfg_lf.force.method = "brute"
    cfg_lf.parallel.enabled = False
    cfg_lf.output.write_final = False
    run_lf = Simulation(cfg_lf, state=state.copy()).run()
    lf_max = _max_relative_energy_error(run_lf.energies)

    cfg_euler = RunConfig()
    cfg_euler.integrator = IntegratorConfig(type="euler", dt=dt, n_steps=n_steps)
    cfg_euler.force.method = "brute"
    cfg_euler.parallel.enabled = False
    cfg_euler.output.write_final = False
    run_euler = Simulation(cfg_euler, state=state.copy()).run()
    euler_max = _max_relative_energy_error(run_euler.energies)

    assert euler_max > lf_max * 5.0, f"euler max {euler_max:.3e} vs leapfrog {lf_max:.3e}"
