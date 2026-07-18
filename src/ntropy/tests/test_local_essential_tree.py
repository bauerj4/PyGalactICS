"""Tests for Gadget-style local trees + Local Essential Tree exchange."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from ntropy.forces.bhtree_c import extension_available
from ntropy.parallel.local_essential_tree import (
    LetPayload,
    apply_remote_accelerations,
    domain_bbox,
    export_let_from_packed,
    min_dist_point_to_aabb,
)
from ntropy.parallel.mpi import compute_forces_mpi, mpi_available
from ntropy.benchmark.mpi_subprocess import mpirun_env


def test_min_dist_point_to_aabb():
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([1.0, 1.0, 1.0])
    assert min_dist_point_to_aabb(np.array([0.5, 0.5, 0.5]), lo, hi) == 0.0
    assert min_dist_point_to_aabb(np.array([2.0, 0.5, 0.5]), lo, hi) == pytest.approx(1.0)
    assert min_dist_point_to_aabb(np.array([-1.0, -1.0, -1.0]), lo, hi) == pytest.approx(
        np.sqrt(3.0)
    )


def test_let_export_accepts_distant_root():
    """A compact local clump should export as a single monopole to a far box."""
    if not extension_available():
        pytest.skip("C Barnes–Hut extension not built")
    from ntropy.forces.bhtree_c import BarnesHutTreeC

    rng = np.random.default_rng(0)
    pos = rng.normal(scale=0.1, size=(32, 3))
    mass = np.ones(32)
    eps = np.full(32, 0.05)
    tree = BarnesHutTreeC.build(pos, mass, eps)
    packed = tree.pack_buffers()
    local_to_global = np.arange(32, dtype=np.int64)
    far_lo = np.array([50.0, 50.0, 50.0])
    far_hi = np.array([51.0, 51.0, 51.0])
    payload = export_let_from_packed(
        packed["nodes"],
        packed["leaf_indices"],
        local_to_global,
        far_lo,
        far_hi,
        theta=0.5,
    )
    assert payload.monopoles.shape[0] >= 1
    assert payload.particle_indices.size == 0
    assert payload.monopoles[:, 3].sum() == pytest.approx(32.0)


def test_apply_remote_monopole_matches_pairwise():
    targets = np.array([[0.0, 0.0, 0.0]])
    target_eps = np.array([0.1])
    monos = np.array([[1.0, 0.0, 0.0, 2.0, 0.5]])  # com, mass, size
    acc = apply_remote_accelerations(
        targets,
        target_eps,
        monopoles=monos,
        source_pos=np.zeros((0, 3)),
        source_mass=np.zeros(0),
        source_eps=np.zeros(0),
        source_indices=np.zeros(0, dtype=np.int64),
    )
    dr = np.array([1.0, 0.0, 0.0])
    r2 = 1.0
    h2 = 0.01
    expected = 2.0 * dr / (r2 + h2) ** 1.5
    np.testing.assert_allclose(acc[0], expected, rtol=1e-12)


@pytest.mark.skipif(not mpi_available(), reason="mpi4py not installed")
def test_mpi_local_trees_matches_replicated(small_plummer_state):
    """Single-process path: local-tree flag is a no-op at size==1."""
    if not extension_available():
        pytest.skip("C Barnes–Hut extension not built")
    state = small_plummer_state
    a = compute_forces_mpi(
        state.pos, state.mass, state.eps, method="bh_c", theta=0.3,
        mpi_local_trees=True,
    )
    b = compute_forces_mpi(
        state.pos, state.mass, state.eps, method="bh_c", theta=0.3,
        mpi_local_trees=False,
    )
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not mpi_available(), reason="mpi4py not installed")
def test_mpirun_local_trees_matches_replicated(tmp_path, small_plummer_state):
    """Multi-rank LET path agrees with replicated full-tree BH."""
    if shutil.which("mpirun") is None:
        pytest.skip("mpirun not available")
    if not extension_available():
        pytest.skip("C Barnes–Hut extension not built")

    state = small_plummer_state
    state_path = tmp_path / "state.npz"
    out_path = tmp_path / "out.npz"
    np.savez(state_path, pos=state.pos, mass=state.mass, eps=state.eps)

    worker = Path(__file__).resolve().parent / "mpi_local_tree_worker.py"
    cmd = [
        "mpirun",
        "--oversubscribe",
        "-n",
        "4",
        sys.executable,
        str(worker),
        str(state_path),
        str(out_path),
        "0.3",
    ]
    import subprocess

    subprocess.run(
        cmd,
        check=True,
        cwd=Path(__file__).resolve().parents[3],
        env=mpirun_env(Path(sys.executable).resolve().parent),
    )
    with np.load(out_path) as data:
        acc_let = data["acc_let"]
        acc_rep = data["acc_rep"]
    # LET uses a conservative AABB opening criterion, so forces match the
    # replicated walk to BH accuracy (not bit-identical).
    rel = np.linalg.norm(acc_let - acc_rep, axis=1) / np.maximum(
        np.linalg.norm(acc_rep, axis=1), 1e-30
    )
    assert np.median(rel) < 0.05
    assert rel.max() < 0.35
