"""Tests for campaign particle file validation."""

from __future__ import annotations

import json

import pytest

from galacticsics.campaign.runner import (
    _commit_evolve_success,
    _validated_evolve_output,
    _validate_particle_files,
)


def test_validate_particle_files_empty(tmp_path):
    (tmp_path / "disk").write_text("")
    (tmp_path / "halo").write_text("")
    with pytest.raises(RuntimeError, match="empty or missing"):
        _validate_particle_files(tmp_path)


def test_validate_particle_files_ok(tmp_path):
    (tmp_path / "disk").write_text("1.0 0 0 0 0 0 0\n" * 3)
    counts = _validate_particle_files(tmp_path)
    assert counts["disk"] == 3


def _write_evolve_artifacts(tmp_path, *, n_particles=3, t_gyr=2.0):
    evolution = tmp_path / "evolution"
    evolution.mkdir()
    (evolution / "final.dat").write_text("1 0 0 0 0 0 0 1\n" * n_particles)
    (evolution / "final.types").write_text("1\n" * n_particles)
    records = [
        {
            "step": 0,
            "t_gyr": 0.0,
            "n_particles": n_particles,
            "dE_over_E0": 0.0,
        },
        {
            "step": 10,
            "t_gyr": t_gyr,
            "n_particles": n_particles,
            "dE_over_E0": 0.001,
        },
    ]
    (evolution / "diagnostics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )


def test_validated_evolve_output_requires_complete_matching_horizon(tmp_path):
    _write_evolve_artifacts(tmp_path)
    evidence = _validated_evolve_output(
        tmp_path, end_time_gyr=2.0, expected_n_particles=3
    )
    assert evidence is not None
    assert evidence["completed_t_gyr"] == 2.0
    assert evidence["n_particles"] == 3
    assert evidence["dE_over_E0"] == pytest.approx(0.001)

    assert (
        _validated_evolve_output(
            tmp_path, end_time_gyr=3.0, expected_n_particles=3
        )
        is None
    )
    (tmp_path / "evolution" / "final.types").write_text("1\n" * 2)
    assert (
        _validated_evolve_output(
            tmp_path, end_time_gyr=2.0, expected_n_particles=3
        )
        is None
    )


def test_commit_evolve_success_writes_result_before_marker(tmp_path):
    _write_evolve_artifacts(tmp_path)
    committed = _commit_evolve_success(
        tmp_path,
        {"evolve_seconds": 12.5},
        end_time_gyr=2.0,
        expected_n_particles=3,
    )
    assert committed["output_validated"] is True
    assert json.loads((tmp_path / "evolve_result.json").read_text()) == committed
    assert (tmp_path / ".done_evolve").read_text() == (
        "verified final output: 3 particles at 2.0 Gyr\n"
    )
