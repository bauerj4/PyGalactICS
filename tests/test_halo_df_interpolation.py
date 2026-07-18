"""Regression: dfnfw.dat energies are descending; samplers must sort before interp."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from galacticsics.diagnostics.df_validation import _read_log_df_table
from galacticsics.physics.backend import PhysicsBackendKind
from galacticsics.potential.solver import solve_potential
from galacticsics.sampling.openmp.pack import pack_halo_context
from galacticsics.sampling.python.samplers import sample_halo_python
from galacticsics.sampling.sampler import SampleConfig
from ntropy.integrations.galacticsics import nfw_halo_model_fast, particle_state_from_galacticsics
from ntropy.softening import virial_diagnostic


@pytest.fixture(scope="module")
def nfw_work(tmp_path_factory) -> Path:
    work = tmp_path_factory.mktemp("nfw_df_interp")
    model = nfw_halo_model_fast()
    solve_potential(model, work_dir=work, cleanup=False, backend=PhysicsBackendKind.PYTHON)
    return work


def test_dfnfw_file_is_descending_but_reader_sorts(nfw_work: Path) -> None:
    raw = np.loadtxt(nfw_work / "dfnfw.dat")
    assert np.all(np.diff(raw[:, 0]) < 0), "dfnfw.dat should be written descending"

    energies, log_df = _read_log_df_table(nfw_work / "dfnfw.dat")
    assert np.all(np.diff(energies) > 0)
    assert log_df[0] < log_df[-1]  # f grows toward the centre (higher E)


def test_pack_halo_context_sorts_df_energies(nfw_work: Path) -> None:
    pack = pack_halo_context(nfw_work)
    assert np.all(np.diff(pack["df_energy"]) > 0)
    # Interpolation at an interior energy must not clamp to the outer fcut.
    from scipy.interpolate import interp1d

    li = interp1d(
        pack["df_energy"],
        pack["df_log_df"],
        kind="linear",
        assume_sorted=True,
        bounds_error=False,
        fill_value=(float(pack["df_log_df"][0]), float(pack["df_log_df"][-1])),
    )
    e_lo, e_hi = float(pack["df_energy"][10]), float(pack["df_energy"][-10])
    assert float(li(e_hi)) > float(li(e_lo))


@pytest.mark.physics_python
def test_sampled_nfw_halo_near_virial_equilibrium(nfw_work: Path) -> None:
    """Broken DF interpolation produced 2T/|W| ~ 1.9; healthy is ~0.8–1.0."""
    ps = sample_halo_python(
        nfw_work,
        n_particles=3000,
        seed=-42,
        center=True,
        config=SampleConfig(use_openmp=False),
    )
    state = particle_state_from_galacticsics(ps, eps=0.04, tag="halo")
    virial = virial_diagnostic(state.pos, state.vel, state.mass, state.eps, rtol=0.35)
    assert 0.65 <= virial["virial_ratio"] <= 1.15, virial
    assert virial["is_virial_equilibrium"]
