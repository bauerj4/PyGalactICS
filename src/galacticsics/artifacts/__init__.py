"""Generate and verify reference artifacts for tests and examples."""

from galacticsics.artifacts.generate import generate_reference_artifacts
from galacticsics.artifacts.paths import default_artifact_dir, reference_model
from galacticsics.artifacts.verify import verify_artifact_consistency

__all__ = [
    "generate_reference_artifacts",
    "verify_artifact_consistency",
    "default_artifact_dir",
    "reference_model",
]
