"""Model representation taxonomy and artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RepresentationKind(str, Enum):
    PARAMETRIC = "parametric"
    HARMONIC = "harmonic"
    PARTICLE = "particle"
    TABULATED = "tabulated"
    LEARNED = "learned"


@dataclass
class ModelArtifact:
    """One model instance with provenance metadata."""

    kind: RepresentationKind
    data: Any
    provenance: list[str] = field(default_factory=list)
    source_hash: str | None = None

    def with_conversion(self, step: str, new_kind: RepresentationKind, new_data: Any) -> ModelArtifact:
        return ModelArtifact(
            kind=new_kind,
            data=new_data,
            provenance=[*self.provenance, step],
            source_hash=self.source_hash,
        )
