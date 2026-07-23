"""Tests for arbitrary IC component discovery in the campaign runner."""

from __future__ import annotations

from pathlib import Path

from galacticsics.campaign.runner import _ic_component_names, _particle_file_counts


def _write_ic(path: Path, n: int = 3) -> None:
    lines = [
        "  1.00000E+00  0.00000E+00  1.00000E+00  0.00000E+00  0.00000E+00  0.00000E+00  0.00000E+00"
        for _ in range(n)
    ]
    path.write_text("\n".join(lines) + "\n")


def test_ic_component_discovery_includes_arbitrary_labels(tmp_path: Path):
    _write_ic(tmp_path / "disk")
    _write_ic(tmp_path / "halo")
    _write_ic(tmp_path / "gas", n=2)
    (tmp_path / "dbh.dat").write_text("not particles\n")

    names = _ic_component_names(tmp_path)
    assert names == ["disk", "gas", "halo"]
    assert _particle_file_counts(tmp_path) == {"disk": 3, "gas": 2, "halo": 3}
