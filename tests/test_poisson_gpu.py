"""Parity / smoke tests for CuPy polar density harmonic fill."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from galacticsics.models import GalaxyModel, NFWHalo, PotentialGrid, SersicBulge
from galacticsics.physics.backend import PhysicsBackendKind
from galacticsics.potential.poisson import fast
from galacticsics.potential.solver import solve_potential
from galacticsics.io import read_harmonic_potential


def _small_model() -> GalaxyModel:
    return GalaxyModel(
        halo=NFWHalo(r_outer=40.0, v0=2.0, a=6.0, dr_trunc=6.0, enabled=True),
        bulge=SersicBulge(n_sersic=4.0, ppp=0.5, v0=1.5, a=0.4, enabled=True),
        grid=PotentialGrid(dr=0.25, nr=40, lmax=2),
    )


@pytest.mark.skipif(not fast.gpu_available(), reason="CuPy GPU not available")
def test_cupy_solve_matches_python_psi0(tmp_path: Path) -> None:
    """Full solve with GALACTICSICS_POISSON_GPU=1 should match Python polar fill."""
    model = _small_model()
    work_cpu = tmp_path / "cpu"
    work_gpu = tmp_path / "gpu"

    os.environ["GALACTICSICS_POISSON_THREADS"] = "0"
    os.environ["GALACTICSICS_POISSON_GPU"] = "0"
    solve_potential(
        model, work_dir=work_cpu, cleanup=False, backend=PhysicsBackendKind.PYTHON, max_iter=12
    )

    os.environ["GALACTICSICS_POISSON_GPU"] = "1"
    solve_potential(
        model, work_dir=work_gpu, cleanup=False, backend=PhysicsBackendKind.PYTHON, max_iter=12
    )
    os.environ["GALACTICSICS_POISSON_GPU"] = "0"

    pot_c = read_harmonic_potential(work_cpu / "dbh.dat")
    pot_g = read_harmonic_potential(work_gpu / "dbh.dat")
    assert pot_c.psi0 == pytest.approx(pot_g.psi0, rel=1e-6, abs=1e-8)
    assert pot_c.psic == pytest.approx(pot_g.psic, rel=1e-6, abs=1e-8)

    mr_c = np.loadtxt(work_cpu / "mr.dat")
    mr_g = np.loadtxt(work_gpu / "mr.dat")
    assert np.allclose(mr_c, mr_g, rtol=1e-5, atol=1e-8)


def test_gpu_module_reports_availability() -> None:
    """gpu_available is a bool whether or not a device is present."""
    assert isinstance(fast.gpu_available(), bool)
