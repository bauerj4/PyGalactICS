"""Tests for optional C Barnes–Hut optimizations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ntropy.config import BhOptimizationsConfig, load_config
from ntropy.forces.bhtree import compute_forces_bh
from ntropy.forces.bhtree_c import (
    PACK_FORMAT_LEGACY,
    PACK_FORMAT_NATIVE,
    BarnesHutTreeC,
    extension_available,
)

pytestmark = pytest.mark.skipif(
    not extension_available(),
    reason="C Barnes–Hut extension not built",
)


def test_bh_opts_legacy_matches_python_bh(small_plummer_state):
    state = small_plummer_state
    legacy = BhOptimizationsConfig(preset="legacy")
    tree = BarnesHutTreeC.build(state.pos, state.mass, state.eps, bh_opts=legacy)
    acc_c = tree.accel_all(theta=0.3)
    acc_py = compute_forces_bh(state.pos, state.mass, state.eps, theta=0.3)
    np.testing.assert_allclose(acc_c, acc_py, rtol=1e-10, atol=1e-10)


def test_bh_opts_optimized_matches_legacy(small_plummer_state):
    state = small_plummer_state
    legacy = BarnesHutTreeC.build(
        state.pos, state.mass, state.eps, bh_opts=BhOptimizationsConfig(preset="legacy")
    )
    optimized = BarnesHutTreeC.build(
        state.pos, state.mass, state.eps, bh_opts=BhOptimizationsConfig(preset="optimized")
    )
    acc_legacy = legacy.accel_all(theta=0.5)
    acc_opt = optimized.accel_all(theta=0.5)
    np.testing.assert_allclose(acc_opt, acc_legacy, rtol=1e-12, atol=1e-12)


def test_bh_native_pack_roundtrip(small_plummer_state):
    state = small_plummer_state
    opts = BhOptimizationsConfig(preset="optimized")
    tree = BarnesHutTreeC.build(state.pos, state.mass, state.eps, bh_opts=opts)
    packed = tree.pack_buffers()
    assert packed["pack_format"] == PACK_FORMAT_NATIVE
    assert packed["nodes_native"].size > 0
    restored = BarnesHutTreeC.from_packed(packed, bh_opts=opts)
    np.testing.assert_allclose(
        restored.accel_all(theta=0.5),
        tree.accel_all(theta=0.5),
        rtol=1e-12,
        atol=1e-12,
    )


def test_bh_legacy_pack_still_works(small_plummer_state):
    state = small_plummer_state
    opts = BhOptimizationsConfig(preset="legacy")
    tree = BarnesHutTreeC.build(state.pos, state.mass, state.eps, bh_opts=opts)
    packed = tree.pack_buffers()
    assert packed["pack_format"] == PACK_FORMAT_LEGACY
    assert packed["nodes"].shape[1] == 19
    restored = BarnesHutTreeC.from_packed(packed, bh_opts=opts)
    np.testing.assert_allclose(
        restored.accel_all(theta=0.5),
        tree.accel_all(theta=0.5),
        rtol=1e-10,
        atol=1e-10,
    )


def test_load_config_bh_optimizations(tmp_path: Path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "particles": {"file": "p.dat"},
                "force": {
                    "method": "bh_c",
                    "bh_optimizations": {
                        "preset": "optimized",
                        "simd_leaves": True,
                        "omp_schedule": "guided",
                    },
                },
            }
        )
    )
    cfg = load_config(cfg_path)
    resolved = cfg.force.bh_optimizations.resolve()
    assert resolved["fast_inv_r3"] is True
    assert resolved["simd_leaves"] is True
    assert resolved["omp_schedule"] == "guided"
