"""Tests for GADGET-style dynamic timestep bins."""

from __future__ import annotations

import numpy as np

from ntropy.integrators.timestep import (
    TimestepConfig,
    active_mask_for_step,
    bin_from_timestep,
    ideal_timestep,
    update_timestep_bins,
)
from ntropy.particle_types import ParticleTypeSpec, TypeRegistry


def test_ideal_timestep_scales_with_softening():
    acc = np.array([[0.0, 1.0, 0.0], [0.0, 4.0, 0.0]])
    eps = np.array([0.01, 0.04])
    dt = ideal_timestep(acc, eps, eta=0.025)
    assert dt[1] > dt[0]


def test_bin_from_timestep_power_of_two():
    bins = bin_from_timestep(np.array([0.02, 0.05, 0.2]), 0.025, max_bin=6)
    assert bins.tolist() == [0, 1, 3]


def test_update_timestep_bins_damping():
    registry = TypeRegistry.from_specs(
        [ParticleTypeSpec(id=1, label="a", eps=0.05, max_timestep_bin=6)]
    )
    acc = np.array([[0.0, 1.0, 0.0], [0.0, 0.01, 0.0]])
    eps = np.array([0.05, 0.05])
    type_id = np.array([1, 1], dtype=np.int32)
    cfg = TimestepConfig(eta=0.025, dt_base=0.025, max_bin=6)

    b0 = update_timestep_bins(acc, eps, type_id, registry, None, cfg)
    # Stronger accel -> finer bin on particle 0
    assert b0[0] <= b0[1]

    # Second update: change limited to one bin
    acc2 = np.array([[0.0, 100.0, 0.0], [0.0, 0.01, 0.0]])
    b1 = update_timestep_bins(acc2, eps, type_id, registry, b0, cfg)
    assert abs(int(b1[0]) - int(b0[0])) <= 1


def test_active_mask_respects_bins():
    bins = np.array([0, 1, 2], dtype=np.int32)
    assert active_mask_for_step(2, bins).tolist() == [True, True, True]
    assert active_mask_for_step(1, bins).tolist() == [True, False, False]


def test_type_max_bin_clamp():
    registry = TypeRegistry.from_specs(
        [
            ParticleTypeSpec(
                id=1, label="disk", eps=0.01, min_timestep_bin=0, max_timestep_bin=1
            )
        ]
    )
    acc = np.array([[0.0, 0.001, 0.0]])  # tiny accel -> huge ideal dt
    eps = np.array([0.01])
    type_id = np.array([1], dtype=np.int32)
    cfg = TimestepConfig(eta=0.025, dt_base=0.025, max_bin=6)
    bins = update_timestep_bins(acc, eps, type_id, registry, None, cfg)
    assert bins[0] <= 1
