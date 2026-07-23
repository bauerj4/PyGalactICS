"""Tests for campaign particle file validation."""

from __future__ import annotations

import pytest

from galacticsics.campaign.runner import _validate_particle_files


def test_validate_particle_files_empty(tmp_path):
    (tmp_path / "disk").write_text("")
    (tmp_path / "halo").write_text("")
    with pytest.raises(RuntimeError, match="empty or missing"):
        _validate_particle_files(tmp_path)


def test_validate_particle_files_ok(tmp_path):
    (tmp_path / "disk").write_text("1.0 0 0 0 0 0 0\n" * 3)
    counts = _validate_particle_files(tmp_path)
    assert counts["disk"] == 3
