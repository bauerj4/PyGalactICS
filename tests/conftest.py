"""Shared pytest configuration and tolerances."""

from __future__ import annotations

from pathlib import Path

import pytest

from galacticsics.artifacts.generate import generate_reference_artifacts
from galacticsics.artifacts.paths import default_artifact_dir
from galacticsics.legacy.paths import require_binary

ROOT = Path(__file__).resolve().parents[1]

RTOL = 1e-5
ATOL = 1e-8
RTOL_SAMPLING = 1e-2


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def reference_artifacts_dir() -> Path:
    """
    Generated reference artifacts (dbh.dat, h.dat, disk, halo, …).

    Built on first test session via :func:`generate_reference_artifacts` if
    missing.  Requires compiled ``legacy/bin`` executables.
    """
    try:
        require_binary("dbh")
        require_binary("getfreqs")
        require_binary("diskdf")
        require_binary("gendisk")
        require_binary("genhalo")
    except FileNotFoundError as exc:
        pytest.skip(
            f"{exc}. Run: make legacy-build legacy-samplers && "
            "galacticsics-generate-artifacts generate"
        )

    out = default_artifact_dir()
    if not (out / "manifest.json").is_file():
        generate_reference_artifacts(out, verify=True)
    return out


@pytest.fixture(scope="session")
def milky_way_dir(reference_artifacts_dir) -> Path:
    """Backward-compatible alias for :func:`reference_artifacts_dir`."""
    return reference_artifacts_dir


@pytest.fixture(scope="session")
def dbh_path(reference_artifacts_dir) -> Path:
    return reference_artifacts_dir / "dbh.dat"


@pytest.fixture(scope="session")
def cordbh_path(reference_artifacts_dir) -> Path:
    return reference_artifacts_dir / "cordbh.dat"


@pytest.fixture(scope="session")
def freqdbh_path(reference_artifacts_dir) -> Path:
    return reference_artifacts_dir / "freqdbh.dat"


@pytest.fixture(scope="session")
def mr_path(reference_artifacts_dir) -> Path:
    return reference_artifacts_dir / "mr.dat"


@pytest.fixture(scope="session")
def reference_model():
    from galacticsics.models import GalaxyModel

    return GalaxyModel.reference_disk_halo()
