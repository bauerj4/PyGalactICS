"""Tests for tiered integrator diagnostics output."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ntropy.analysis.tiered_diagnostics import (
    collect_step_diagnostics,
    load_diagnostics_log,
    write_diagnostics_log,
    write_particle_bin_dump,
)
from ntropy.ics.plummer import sample_plummer
from ntropy.integrators.timestep import TimestepConfig
from ntropy.integrators.tiered import run_tiered_leapfrog
from ntropy.particle_types import ParticleTypeSpec, TypeRegistry


def test_collect_step_diagnostics_histogram():
    registry = TypeRegistry.from_specs(
        [ParticleTypeSpec(id=1, label="all", eps=0.05, min_timestep_bin=0, max_timestep_bin=4)]
    )
    n = 16
    bins = np.array([0, 0, 1, 1, 2, 2, 2, 2, 0, 0, 1, 1, 3, 3, 3, 3], dtype=np.int32)
    type_id = np.ones(n, dtype=np.int32)
    rec = collect_step_diagnostics(
        step=5,
        bins=bins,
        type_id=type_id,
        registry=registry,
        energy=-1.0,
        e0=-1.0,
        n_active=8,
        dt_base=0.025,
        max_bin=4,
        acc=np.ones((n, 3)),
    )
    assert rec.step == 5
    assert rec.n_active == 8
    assert sum(rec.bin_counts) == n
    assert rec.by_type["all"]["mean_bin"] == pytest.approx(bins.mean())


def test_run_tiered_writes_diagnostics_and_callback(tmp_path: Path):
    state = sample_plummer(seed=3)
    registry = TypeRegistry.from_specs(
        [ParticleTypeSpec(id=1, label="all", eps=0.05, min_timestep_bin=0, max_timestep_bin=3)]
    )
    state.type_id = np.ones(state.n, dtype=np.int32)
    dumps: list[int] = []

    def accel(pos: np.ndarray) -> np.ndarray:
        from ntropy.forces.brute import compute_forces_brute
        return compute_forces_brute(pos, state.mass, state.eps)

    def on_record(step: int, snap, acc: np.ndarray) -> None:
        dumps.append(step)
        write_particle_bin_dump(tmp_path / f"step_{step:06d}.npz", snap, acc=acc)

    ts = TimestepConfig(eta=0.05, dt_base=0.05, max_bin=3, update_every=1)
    _, _, diag = run_tiered_leapfrog(
        state,
        registry,
        accel,
        ts_config=ts,
        end_time_gyr=0.002,
        diagnostics_every=1,
        particle_dump_every=1,
        on_record=on_record,
    )
    write_diagnostics_log(diag, tmp_path)
    assert (tmp_path / "diagnostics.csv").exists()
    assert (tmp_path / "diagnostics.jsonl").exists()
    loaded = load_diagnostics_log(tmp_path)
    assert len(loaded.steps) == diag.n_recorded
    assert len(dumps) >= 2
    assert (tmp_path / "step_000000.npz").exists()


def test_run_tiered_streams_diagnostics_without_ram_growth(tmp_path: Path):
    state = sample_plummer(seed=5)
    registry = TypeRegistry.from_specs(
        [ParticleTypeSpec(id=1, label="all", eps=0.05, min_timestep_bin=0, max_timestep_bin=3)]
    )
    state.type_id = np.ones(state.n, dtype=np.int32)
    jsonl_path = tmp_path / "diagnostics.jsonl"
    jsonl_path.write_text("")

    def accel(pos: np.ndarray) -> np.ndarray:
        from ntropy.forces.brute import compute_forces_brute
        return compute_forces_brute(pos, state.mass, state.eps)

    ts = TimestepConfig(eta=0.05, dt_base=0.05, max_bin=3, update_every=1)
    _, _, diag = run_tiered_leapfrog(
        state,
        registry,
        accel,
        ts_config=ts,
        end_time_gyr=0.002,
        diagnostics_every=1,
        diagnostics_jsonl=jsonl_path,
    )
    assert diag.n_recorded > 0
    assert len(diag.steps) == 0
    assert jsonl_path.stat().st_size > 0
    write_diagnostics_log(diag, tmp_path)
    loaded = load_diagnostics_log(tmp_path)
    assert len(loaded.steps) == diag.n_recorded
