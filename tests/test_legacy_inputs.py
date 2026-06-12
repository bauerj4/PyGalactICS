"""Tests for legacy stdin file generation."""

from __future__ import annotations

from pathlib import Path

from galacticsics.io.legacy_inputs import write_dbh_input, write_gendenspsi_input
from galacticsics.models import GalaxyModel


def test_write_dbh_input_milky_way(tmp_path):
    model = GalaxyModel.milky_way_disk_halo()
    path = tmp_path / "in.dbh"
    write_dbh_input(model, path)
    lines = path.read_text().strip().splitlines()
    assert lines[0] == "y"  # halo
    assert "200.0" in lines[1] or lines[1].startswith("200")
    assert lines[2] == "y"  # disk
    assert lines[-2] == "0.02 20000"
    assert lines[-1] == "6"


def test_write_gendenspsi(tmp_path):
    write_gendenspsi_input(tmp_path / "in.gendenspsi", npsi=500, nint=10)
    assert tmp_path.joinpath("in.gendenspsi").read_text().strip() == "500 10"
