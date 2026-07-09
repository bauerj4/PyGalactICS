"""Convert between model representations."""

from __future__ import annotations

from galacticsics.models import GalaxyModel
from galacticsics.representations import ModelArtifact, RepresentationKind


def artifact_from_parametric(model: GalaxyModel, *, source_hash: str | None = None) -> ModelArtifact:
    return ModelArtifact(
        kind=RepresentationKind.PARAMETRIC,
        data=model,
        provenance=["GalaxyModel"],
        source_hash=source_hash,
    )


def to_harmonic(artifact: ModelArtifact, work_dir: str) -> ModelArtifact:
    """Solve parametric model → harmonic potential."""
    from galacticsics.potential.solver import solve_potential

    if artifact.kind != RepresentationKind.PARAMETRIC:
        raise TypeError(f"expected PARAMETRIC, got {artifact.kind}")
    result = solve_potential(artifact.data, work_dir=work_dir, cleanup=False)
    return artifact.with_conversion(
        f"solve_potential({work_dir})",
        RepresentationKind.HARMONIC,
        result.potential,
    )
