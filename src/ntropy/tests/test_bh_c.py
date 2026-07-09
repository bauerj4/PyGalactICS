"""Tests for the C Barnes–Hut extension and Python wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from ntropy.forces.bhtree import compute_forces_bh
from ntropy.forces.brute import compute_forces_brute
from ntropy.forces.bhtree_c import (
    BarnesHutTreeC,
    compute_forces_bh_c,
    extension_available,
)

pytestmark = pytest.mark.skipif(
    not extension_available(),
    reason="C Barnes–Hut extension not built",
)


def test_bh_c_matches_python_bh(small_plummer_state):
    state = small_plummer_state
    acc_py = compute_forces_bh(state.pos, state.mass, state.eps, theta=0.3)
    acc_c = compute_forces_bh_c(state.pos, state.mass, state.eps, theta=0.3)
    np.testing.assert_allclose(acc_c, acc_py, rtol=1e-10, atol=1e-10)


def test_bh_c_matches_brute_small(small_plummer_state):
    state = small_plummer_state
    acc_brute = compute_forces_brute(state.pos, state.mass, state.eps)
    acc_c = compute_forces_bh_c(state.pos, state.mass, state.eps, theta=0.3)
    np.testing.assert_allclose(acc_c, acc_brute, rtol=0.15, atol=1e-3)


def test_bh_c_targets_subset(small_plummer_state):
    state = small_plummer_state
    targets = np.array([0, 2, 4, 7, 11], dtype=np.int32)
    tree = BarnesHutTreeC.build(state.pos, state.mass, state.eps)
    partial = tree.accel_targets(targets, theta=0.4)
    full = tree.accel_all(theta=0.4)
    np.testing.assert_allclose(partial, full[targets], rtol=1e-10, atol=1e-10)


def test_bh_c_pack_roundtrip(small_plummer_state):
    state = small_plummer_state
    tree = BarnesHutTreeC.build(state.pos, state.mass, state.eps)
    packed = tree.pack_buffers()
    assert packed["nodes"].shape[1] == 19
    restored = BarnesHutTreeC.from_packed(packed)
    acc_orig = tree.accel_all(theta=0.5)
    acc_restored = restored.accel_all(theta=0.5)
    np.testing.assert_allclose(acc_restored, acc_orig, rtol=1e-10, atol=1e-10)
