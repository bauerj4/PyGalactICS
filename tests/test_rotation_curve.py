"""Tests for rotation curve diagnostics."""

from __future__ import annotations

import json

import numpy as np

from galacticsics.diagnostics.rotation_curve import (
    build_rotation_curve_report,
    write_rotation_curve_diagnostic,
)
from galacticsics.models import GalaxyModel


def test_rotation_curve_report_on_solved_model(tmp_path):
    model = GalaxyModel.reference_disk_halo()
    from galacticsics.potential.solver import solve_potential

    solve_potential(model, work_dir=tmp_path, cleanup=False, backend="python")
    report = build_rotation_curve_report(tmp_path, model=model, label="model")
    assert len(report["r_kpc"]) == 40
    assert len(report["v_circ_potential"]) == 40
    assert "v_circ_frequency" in report
    assert all(v > 0 for v in report["v_circ_potential"][5:15])


def test_write_rotation_curve_diagnostic(tmp_path):
    model = GalaxyModel.reference_disk_halo()
    from galacticsics.potential.solver import solve_potential

    solve_potential(model, work_dir=tmp_path, cleanup=False, backend="python")
    path = write_rotation_curve_diagnostic(tmp_path, model=model, label="ic")
    data = json.loads(path.read_text())
    assert data["label"] == "ic"
    assert "v_circ_potential" in data


def test_rotation_curve_report_per_component(tmp_path):
    model = GalaxyModel.reference_disk_halo()
    from galacticsics.potential.solver import solve_potential

    solve_potential(model, work_dir=tmp_path, cleanup=False, backend="python")

    from ntropy.particle_types import TypeRegistry
    from ntropy.particles import ParticleState

    registry = TypeRegistry.default_galaxy()
    n = 200
    state = ParticleState.from_arrays(
        np.random.default_rng(0).normal(size=(n, 3)),
        np.random.default_rng(1).normal(size=(n, 3)) * 0.1,
        np.ones(n),
        np.full(n, 0.05),
        type_id=np.concatenate(
            [
                np.full(n // 2, registry.id_for("halo"), dtype=np.int32),
                np.full(n - n // 2, registry.id_for("disk"), dtype=np.int32),
            ]
        ),
    )
    report = build_rotation_curve_report(tmp_path, state=state, model=model, label="ic")
    assert "v_circ_disk" in report
    assert "v_circ_halo" in report
