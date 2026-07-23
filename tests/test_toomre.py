"""Tests for Toomre Q scaling and logging."""

from __future__ import annotations

from dataclasses import replace

import pytest

from galacticsics.campaign.spec import GridSpec, expand_grid
from galacticsics.distribution.toomre import (
    apply_toomre_q_target,
    compute_toomre_q,
    log_toomre_q,
    write_toomre_q,
)
from galacticsics.io.formats import read_toomre_q
from galacticsics.models import DiskKinematics, GalaxyModel


@pytest.fixture
def solved_reference(tmp_path):
    model = GalaxyModel.reference_disk_halo()
    from galacticsics.potential.solver import solve_potential

    solve_potential(model, work_dir=tmp_path, cleanup=False, backend="python")
    return model, tmp_path


def test_expand_grid_disk_kinematics_axes():
    spec = GridSpec(
        base="reference_disk_halo",
        axes={"disk_kinematics.toomre_q_target": [1.2, 1.5]},
        omit_components=[[]],
        coarse_grid=True,
    )
    models = expand_grid(spec)
    assert len(models) == 2
    _, m = models[0]
    assert m.disk_kinematics.toomre_q_target == 1.2


def test_compute_and_write_toomre_q(solved_reference):
    model, work_dir = solved_reference
    q, r = compute_toomre_q(model, work_dir)
    assert q > 0
    assert r == pytest.approx(2.5 * model.disk.scale_length)
    write_toomre_q(work_dir, q)
    assert read_toomre_q(work_dir / "toomre2.5") == pytest.approx(q)


def test_apply_toomre_q_target_scales_sigma_r0(solved_reference):
    model, work_dir = solved_reference
    q0, _ = compute_toomre_q(model, work_dir)
    target = q0 * 1.25
    kin = replace(model.disk_kinematics, toomre_q_target=target)
    model_t = replace(model, disk_kinematics=kin)
    scaled = apply_toomre_q_target(model_t, work_dir)
    q1, _ = compute_toomre_q(scaled, work_dir)
    assert q1 == pytest.approx(target, rel=1e-3)
    assert scaled.disk_kinematics.sigma_r0 == pytest.approx(
        model.disk_kinematics.sigma_r0 * target / q0, rel=1e-6
    )


def test_log_toomre_q_writes_file(solved_reference):
    model, work_dir = solved_reference
    out = log_toomre_q(model, work_dir)
    assert (work_dir / "toomre2.5").is_file()
    assert out["toomre_q"] == pytest.approx(read_toomre_q(work_dir / "toomre2.5"))
