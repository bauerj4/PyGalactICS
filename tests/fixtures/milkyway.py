"""Reference model helpers (generated artifacts, not legacy files)."""

from __future__ import annotations

from pathlib import Path

from galacticsics.artifacts.paths import default_artifact_dir
from galacticsics.models import GalaxyModel


def reference_model() -> GalaxyModel:
    return GalaxyModel.reference_disk_halo()


def milky_way_model() -> GalaxyModel:
    """Alias for the reference test model."""
    return reference_model()


def reference_artifacts_dir(root: Path | None = None) -> Path:
    """Directory with generated reference artifacts."""
    if root is not None:
        return root / "tests" / "generated" / "reference"
    return default_artifact_dir()


def milky_way_data_dir(root: Path) -> Path:
    """Backward-compatible alias."""
    return reference_artifacts_dir(root)
