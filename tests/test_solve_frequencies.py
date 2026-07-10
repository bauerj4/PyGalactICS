"""Tests for Poisson solve frequency tabulation and diskdf convergence."""

from __future__ import annotations

import math

import numpy as np
import pytest

from galacticsics.campaign.spec import preview_dbh_model
from galacticsics.distribution.toomre import compute_toomre_q
from galacticsics.io.formats import cordbh_is_valid, read_disk_correction, read_frequency_table
from galacticsics.potential.frequencies_tabulate import tabulate_frequencies
from galacticsics.potential.solver import solve_potential


@pytest.mark.physics_python
def test_tabulated_kappa_positive_at_disk_radius(tmp_path):
    model = preview_dbh_model(base="milky_way_disk_halo", coarse=True)
    solve_potential(model, work_dir=tmp_path, cleanup=False, backend="python")
    tabulate_frequencies(tmp_path)
    freq = read_frequency_table(tmp_path / "freqdbh.dat")
    r = 2.5 * model.disk.scale_length
    assert freq.kappa(r) > 0.05
    assert freq.omega(r) > 0.05
    assert math.isfinite(freq.omega(r))


@pytest.mark.physics_python
@pytest.mark.slow
def test_diskdf_converges_after_freq_fix(tmp_path):
    from galacticsics.physics.python_backend import python_ensure_disk_df

    model = preview_dbh_model(base="milky_way_disk_halo", coarse=True)
    work = tmp_path / "diskdf"
    solve_potential(model, work_dir=work, cleanup=False, backend="python")
    python_ensure_disk_df(model, work, diskdf_backend="python")
    corr = read_disk_correction(work / "cordbh.dat")
    assert float(np.median(corr.f_d[1:])) > 0.1
    assert cordbh_is_valid(work / "cordbh.dat")


@pytest.mark.physics_python
def test_toomre_target_after_freq_fix(reference_artifacts_dir, tmp_path):
    import shutil
    from dataclasses import replace

    from galacticsics.io.legacy_inputs import write_gendenspsi_input
    from galacticsics.models import DiskKinematics
    from galacticsics.physics.python_backend import python_ensure_disk_df

    work = tmp_path / "toomre"
    work.mkdir()
    for name in ("dbh.dat", "h.dat"):
        shutil.copy2(reference_artifacts_dir / name, work / name)
    write_gendenspsi_input(work / "in.gendenspsi")
    model = preview_dbh_model(base="milky_way_disk_halo", coarse=True)
    kin = replace(model.disk_kinematics, toomre_q_target=1.5)
    model = replace(model, disk_kinematics=kin)
    tabulate_frequencies(work)
    scaled = python_ensure_disk_df(model, work)
    corr = read_disk_correction(work / "cordbh.dat")
    q, _ = compute_toomre_q(scaled, work)
    assert math.isfinite(q)
    assert float(np.median(corr.f_d[1:])) > 0.1
    assert cordbh_is_valid(work / "cordbh.dat")
