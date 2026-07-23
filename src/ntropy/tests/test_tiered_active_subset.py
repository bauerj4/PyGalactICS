"""Tests for active-subset force evaluation in tiered integration."""

from __future__ import annotations

import numpy as np
import pytest

from ntropy.config import ForceConfig
from ntropy.forces.context import ForceContext
from ntropy.ics.plummer import sample_plummer
from ntropy.integrators.timestep import TimestepConfig
from ntropy.integrators.tiered import run_tiered_leapfrog
from ntropy.particle_types import ParticleTypeSpec, TypeRegistry


def test_active_subset_matches_full_when_all_active():
    state = sample_plummer(seed=3)
    registry = TypeRegistry.from_specs(
        [ParticleTypeSpec(id=1, label="all", eps=0.05, min_timestep_bin=0, max_timestep_bin=2)]
    )
    state.type_id = np.ones(state.n, dtype=np.int32)
    ts_config = TimestepConfig(eta=0.05, dt_base=0.05, max_bin=0, update_every=1)

    def full_accel(pos: np.ndarray, active_idx=None) -> np.ndarray:
        from ntropy.forces.brute import compute_forces_brute
        return compute_forces_brute(pos, state.mass, state.eps)

    def subset_accel(pos: np.ndarray, active_idx=None) -> np.ndarray:
        ctx = ForceContext(config=ForceConfig(method="brute", active_subset=True))
        if active_idx is None:
            acc = ctx.accel_at_pos(state, pos)
        else:
            cache = np.zeros((state.n, 3))
            cache[active_idx] = ctx.accel_at_pos(state, pos, target_indices=active_idx)
            acc = cache
        ctx.after_force_eval()
        return acc

    s_full = state.copy()
    s_sub = state.copy()
    _, e_full, _ = run_tiered_leapfrog(
        s_full, registry, full_accel, ts_config=ts_config, end_time_gyr=0.0002, order=2
    )
    _, e_sub, _ = run_tiered_leapfrog(
        s_sub, registry, subset_accel, ts_config=ts_config, end_time_gyr=0.0002, order=2
    )
    np.testing.assert_allclose(s_full.pos, s_sub.pos, rtol=0, atol=1e-10)
    np.testing.assert_allclose(s_full.vel, s_sub.vel, rtol=0, atol=1e-10)
    assert e_full[-1] == pytest.approx(e_sub[-1], rel=1e-10)


def test_force_context_active_subset_reduces_targets():
    from ntropy.forces.brute import compute_forces_brute

    state = sample_plummer(seed=4)
    targets = np.array([0, 2, 4], dtype=int)
    ctx = ForceContext(config=ForceConfig(method="brute", active_subset=True))
    full = ctx.accel_at_pos(state, state.pos)
    ctx.after_force_eval()
    partial = ctx.accel_at_pos(state, state.pos, target_indices=targets)
    np.testing.assert_allclose(partial, full[targets], rtol=0, atol=1e-12)
