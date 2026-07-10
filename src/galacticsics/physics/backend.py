"""Physics backend selection for solve / diskdf / sampling."""

from __future__ import annotations

import os
from enum import Enum


class PhysicsBackendKind(str, Enum):
    """IC-generation numerics backend."""

    PYTHON = "python"
    LEGACY = "legacy"


def default_physics_backend() -> PhysicsBackendKind:
    """Return the default backend (``GALACTICSICS_PHYSICS_BACKEND`` or python)."""
    raw = os.environ.get("GALACTICSICS_PHYSICS_BACKEND", "python").strip().lower()
    try:
        return PhysicsBackendKind(raw)
    except ValueError:
        return PhysicsBackendKind.PYTHON


def resolve_physics_backend(
    backend: PhysicsBackendKind | str | None,
) -> PhysicsBackendKind:
    if backend is None:
        return default_physics_backend()
    if isinstance(backend, PhysicsBackendKind):
        return backend
    return PhysicsBackendKind(str(backend).strip().lower())


def get_physics_backend(kind: PhysicsBackendKind | str | None = None) -> PhysicsBackendKind:
    """Alias for :func:`resolve_physics_backend`."""
    return resolve_physics_backend(kind)


def physics_backend_available(kind: PhysicsBackendKind | str | None = None) -> bool:
    """Return whether the requested backend can run halo / disk / bulge IC workflows."""
    resolved = resolve_physics_backend(kind)
    if resolved == PhysicsBackendKind.LEGACY:
        try:
            from galacticsics.legacy.paths import require_binary

            require_binary("dbh")
            require_binary("getfreqs")
            require_binary("diskdf")
            require_binary("gendisk")
            require_binary("genhalo")
            return True
        except (ImportError, FileNotFoundError):
            return False
    return True
