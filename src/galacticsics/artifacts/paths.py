"""Filesystem locations for generated reference artifacts."""

from __future__ import annotations

import os
from pathlib import Path

from galacticsics.legacy.paths import _repo_root
from galacticsics.models import GalaxyModel


def default_artifact_dir() -> Path:
    """
    Directory where reference artifacts are stored.

    Resolution order:

    1. ``GALACTICSICS_ARTIFACT_DIR`` environment variable
    2. ``<repo>/tests/generated/reference`` when running from a checkout
    3. ``~/.cache/galacticsics/reference`` for installed packages

    Returns
    -------
    Path
        Target directory (not guaranteed to exist).
    """
    env = os.environ.get("GALACTICSICS_ARTIFACT_DIR")
    if env:
        return Path(env).resolve()
    root = _repo_root()
    checkout = root / "tests" / "generated" / "reference"
    if (root / "tests").is_dir():
        return checkout
    return Path.home() / ".cache" / "galacticsics" / "reference"


def reference_model() -> GalaxyModel:
    """Alias for :meth:`~galacticsics.models.GalaxyModel.reference_disk_halo`."""
    return GalaxyModel.reference_disk_halo()
