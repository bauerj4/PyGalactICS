"""Filesystem paths to the isolated legacy tree."""

from __future__ import annotations

import os
from pathlib import Path


def _repo_root() -> Path:
    """Return repository root (parent of ``src/``)."""
    env = os.environ.get("GALACTICSICS_ROOT")
    if env:
        return Path(env).resolve()
    # src/galacticsics/legacy/paths.py -> repo root is 3 parents up
    return Path(__file__).resolve().parents[3]


def legacy_root() -> Path:
    """
    Path to the legacy code tree.

    Returns
    -------
    Path
        Absolute path to ``legacy/`` containing ``fortran/``, ``bin/``, and
        ``python/`` subdirectories.
    """
    return _repo_root() / "legacy"


def legacy_bin_dir() -> Path:
    """
    Path to compiled legacy executables.

    Returns
    -------
    Path
        Absolute path to ``legacy/bin/``.

    Notes
    -----
    Build binaries with ``make legacy-build`` from the repository root, or
    ``make -C legacy/fortran all install``.
    """
    return legacy_root() / "bin"


def require_binary(name: str) -> Path:
    """
    Resolve a legacy executable, raising if missing.

    Parameters
    ----------
    name : str
        Executable basename (e.g. ``"dbh"``, ``"gendisk"``).

    Returns
    -------
    Path
        Absolute path to the executable.

    Raises
    ------
    FileNotFoundError
        If the executable is not present under ``legacy/bin/``.
    """
    path = legacy_bin_dir() / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Legacy binary '{name}' not found at {path}. "
            "Build with: make legacy-build"
        )
    return path
