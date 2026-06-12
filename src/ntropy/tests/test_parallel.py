"""Tests for MPI parallel force computation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ntropy.benchmark.mpi_subprocess import run_mpirun_benchmark, run_mpirun_simulation
from ntropy.forces.brute import compute_forces_brute
from ntropy.parallel.mpi import compute_forces_mpi, mpi_available
from ntropy.parallel.pool import compute_forces_parallel


def test_mpi_serial_matches_brute(small_plummer_state):
    """Single-rank MPI path matches brute force."""
    state = small_plummer_state
    serial = compute_forces_brute(state.pos, state.mass, state.eps)
    mpi_acc = compute_forces_mpi(state.pos, state.mass, state.eps, method="brute")
    np.testing.assert_allclose(serial, mpi_acc, rtol=1e-10, atol=1e-12)


def test_pool_dispatch_matches_brute(small_plummer_state):
    """compute_forces_parallel dispatches to MPI backend."""
    state = small_plummer_state
    serial = compute_forces_brute(state.pos, state.mass, state.eps)
    parallel = compute_forces_parallel(
        state.pos, state.mass, state.eps, method="brute", n_workers=2
    )
    np.testing.assert_allclose(serial, parallel, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not mpi_available(), reason="mpi4py not installed")
def test_mpirun_benchmark_worker(tmp_path):
    """Notebook-style mpirun worker writes timing JSON."""
    import shutil

    if shutil.which("mpirun") is None:
        pytest.skip("mpirun not available")

    state = np.random.default_rng(0)
    pos = state.normal(size=(48, 3))
    mass = np.ones(48)
    eps = np.full(48, 0.05)
    state_path = tmp_path / "state.npz"
    out_path = tmp_path / "out.json"
    np.savez(state_path, pos=pos, mass=mass, eps=eps)

    run_mpirun_benchmark(
        2,
        [str(state_path), "brute", "0.5", "3", str(out_path)],
        cwd=Path(__file__).resolve().parents[3],
        python=sys.executable,
        venv_bin=Path(sys.executable).resolve().parent,
        timeout_s=120.0,
    )
    payload = json.loads(out_path.read_text())
    assert payload["n_ranks"] == 2
    assert payload["time_per_force_s"] > 0


@pytest.mark.skipif(not mpi_available(), reason="mpi4py not installed")
def test_mpirun_simulation_worker(tmp_path):
    """MPI simulation worker records energies for notebook energy-drift runs."""
    import shutil

    if shutil.which("mpirun") is None:
        pytest.skip("mpirun not available")

    rng = np.random.default_rng(0)
    n = 32
    pos = rng.normal(size=(n, 3))
    vel = rng.normal(scale=0.1, size=(n, 3))
    mass = np.ones(n)
    eps = np.full(n, 0.05)
    state_path = tmp_path / "state.npz"
    config_path = tmp_path / "config.json"
    out_path = tmp_path / "out.json"
    np.savez(state_path, pos=pos, vel=vel, mass=mass, eps=eps)
    config_path.write_text(
        json.dumps(
            {
                "integrator": {
                    "type": "leapfrog",
                    "order": 2,
                    "dt": 0.01,
                    "n_steps": 3,
                },
                "force": {"method": "bh", "theta": 0.5},
                "parallel": {"enabled": True, "n_workers": 2},
            }
        )
    )

    run_mpirun_simulation(
        2,
        [str(state_path), str(config_path), str(out_path)],
        cwd=Path(__file__).resolve().parents[3],
        python=sys.executable,
        venv_bin=Path(sys.executable).resolve().parent,
        timeout_s=120.0,
    )
    payload = json.loads(out_path.read_text())
    assert payload["n_ranks"] == 2
    assert len(payload["energies"]) == 4  # initial + 3 steps


@pytest.mark.skipif(not mpi_available(), reason="mpi4py not installed")
def test_mpi_bh_c_matches_python_bh(small_plummer_state):
    """MPI C Barnes–Hut path matches the Python reference."""
    from ntropy.forces.bhtree import compute_forces_bh
    from ntropy.forces.bhtree_c import extension_available

    if not extension_available():
        pytest.skip("C Barnes–Hut extension not built")

    state = small_plummer_state
    serial = compute_forces_bh(state.pos, state.mass, state.eps, theta=0.3)
    mpi_acc = compute_forces_mpi(
        state.pos, state.mass, state.eps, method="bh_c", theta=0.3
    )
    np.testing.assert_allclose(serial, mpi_acc, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not mpi_available(), reason="mpi4py not installed")
def test_mpi_bh_matches_brute(small_plummer_state):
    """Single-rank Barnes–Hut MPI path matches brute force at small N."""
    state = small_plummer_state
    serial = compute_forces_brute(state.pos, state.mass, state.eps)
    mpi_acc = compute_forces_mpi(
        state.pos, state.mass, state.eps, method="bh", theta=0.3
    )
    np.testing.assert_allclose(serial, mpi_acc, rtol=0.15, atol=1e-3)
