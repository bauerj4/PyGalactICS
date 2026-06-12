"""Tests for MPI parallel force computation."""

from __future__ import annotations

import numpy as np
import pytest

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
def test_mpi_bh_matches_brute(small_plummer_state):
    """Single-rank Barnes–Hut MPI path matches brute force at small N."""
    state = small_plummer_state
    serial = compute_forces_brute(state.pos, state.mass, state.eps)
    mpi_acc = compute_forces_mpi(
        state.pos, state.mass, state.eps, method="bh", theta=0.3
    )
    np.testing.assert_allclose(serial, mpi_acc, rtol=0.15, atol=1e-3)
