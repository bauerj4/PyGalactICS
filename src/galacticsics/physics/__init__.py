"""Physics backends for GalactICS IC generation (Python or legacy Fortran)."""

from galacticsics.physics.backend import (
    PhysicsBackendKind,
    default_physics_backend,
    get_physics_backend,
    physics_backend_available,
    resolve_physics_backend,
)

__all__ = [
    "PhysicsBackendKind",
    "default_physics_backend",
    "get_physics_backend",
    "physics_backend_available",
    "resolve_physics_backend",
]
