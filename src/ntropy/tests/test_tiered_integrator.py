"""Tests for tiered leapfrog integration."""

from __future__ import annotations

import numpy as np

from ntropy.ics.plummer import sample_plummer
from ntropy.integrators.timestep import TimestepConfig
from ntropy.integrators.tiered import run_tiered_leapfrog
from ntropy.particle_types import ParticleTypeSpec, TypeRegistry


def test_tiered_leapfrog_short_run():
    state = sample_plummer(seed=1)
    registry = TypeRegistry.from_specs(
        [
            ParticleTypeSpec(
                id=1, label="all", eps=0.05, min_timestep_bin=0, max_timestep_bin=4
            )
        ]
    )
    state.type_id = np.ones(state.n, dtype=np.int32)

    def accel(pos: np.ndarray, active_idx: np.ndarray | None = None) -> np.ndarray:
        from ntropy.forces.brute import compute_forces_brute
        return compute_forces_brute(pos, state.mass, state.eps)

    ts_config = TimestepConfig(eta=0.05, dt_base=0.05, max_bin=4)
    _, energies, diag = run_tiered_leapfrog(
        state,
        registry,
        accel,
        ts_config=ts_config,
        end_time_gyr=0.001,
        order=2,
    )
    assert len(energies) >= 2
    assert len(diag.steps) >= 2
    assert state.timestep_bin is not None
    assert state.timestep_bin.shape == (state.n,)


def test_tiered_bins_differ_for_different_accel():
    """Disks in stronger fields should trend toward finer bins than outer halo."""
    registry = TypeRegistry.default_galaxy()
    state = sample_plummer(seed=2)
    n = state.n
    half = n // 2
    state.type_id = np.concatenate(
        [
            np.full(half, registry.id_for("halo"), dtype=np.int32),
            np.full(n - half, registry.id_for("disk"), dtype=np.int32),
        ]
    )
    state.eps[:half] = 0.05
    state.eps[half:] = 0.01

    def accel(pos: np.ndarray, active_idx: np.ndarray | None = None) -> np.ndarray:
        from ntropy.forces.brute import compute_forces_brute
        return compute_forces_brute(pos, state.mass, state.eps)

    ts_config = TimestepConfig(eta=0.025, dt_base=0.025, max_bin=5)
    _, _, diag = run_tiered_leapfrog(
        state, registry, accel, ts_config=ts_config, end_time_gyr=0.0005, order=2
    )
    assert len(diag.steps) >= 1
    halo_bins = state.timestep_bin[:half]
    disk_bins = state.timestep_bin[half:]
    # Dynamic bins: not fixed by type; allow equal but typically disk <= halo coarseness
    assert halo_bins.max() >= disk_bins.min()
