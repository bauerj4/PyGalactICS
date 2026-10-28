"""Tests for Toomre Q scaling and logging."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from galacticsics.campaign.spec import GridSpec, expand_grid
from galacticsics.distribution.diskdf_solve import (
    _sigma_r2_base,
    bind_disk_kinematics,
)
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


def test_disk_df_uses_correction_table_kinematics() -> None:
    """A post-potential Q adjustment must replace the stale dbh header value."""
    from types import SimpleNamespace

    model = GalaxyModel.reference_disk_halo()
    pot = SimpleNamespace(model=model)
    tuned_sigma_r0 = 1.37 * model.disk_kinematics.sigma_r0
    bind_disk_kinematics(
        pot,
        sigma_r0=tuned_sigma_r0,
        sigma_r_scale=model.disk_kinematics.sigma_r_scale,
    )

    radius = 2.5 * model.disk.scale_length
    expected = tuned_sigma_r0**2 * math.exp(
        -radius / model.disk_kinematics.sigma_r_scale
    )
    assert pot.model.disk_kinematics.sigma_r0 == pytest.approx(tuned_sigma_r0)
    assert _sigma_r2_base(radius, pot.model) == pytest.approx(expected)


def test_diskdf_solver_binds_q_adjusted_kinematics(
    solved_reference,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The correction integral itself must see the post-potential Q scaling."""
    from galacticsics.distribution import diskdf_solve

    model, work_dir = solved_reference
    tuned_sigma_r0 = 1.29 * model.disk_kinematics.sigma_r0
    tuned = replace(
        model,
        disk_kinematics=replace(
            model.disk_kinematics,
            sigma_r0=tuned_sigma_r0,
        ),
    )
    seen_sigma_r0: list[float] = []

    def record_integral(vt, pot, *args):
        del vt, args
        seen_sigma_r0.append(pot.model.disk_kinematics.sigma_r0)
        return 1.0

    monkeypatch.setattr(diskdf_solve, "_integrate_vphi", record_integral)
    corr = diskdf_solve.solve_diskdf_python(
        tuned,
        work_dir,
        n_radial_steps=2,
        n_iterations=1,
    )

    assert seen_sigma_r0
    assert seen_sigma_r0 == pytest.approx([tuned_sigma_r0] * len(seen_sigma_r0))
    assert corr.sigma_r0 == pytest.approx(tuned_sigma_r0)


def test_log_toomre_q_writes_file(solved_reference):
    model, work_dir = solved_reference
    out = log_toomre_q(model, work_dir)
    assert (work_dir / "toomre2.5").is_file()
    assert out["toomre_q"] == pytest.approx(read_toomre_q(work_dir / "toomre2.5"))
