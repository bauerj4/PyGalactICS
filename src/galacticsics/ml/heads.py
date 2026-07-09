"""Multi-task training objectives for particle encoders (stubs)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class TrainingObjective(str, Enum):
    """Self-supervised and supervised heads for encoder training."""

    CONTRASTIVE = "contrastive"
    QUALITY_REGRESSION = "quality_regression"
    PARAM_INVERSE = "param_inverse"
    FIELD_RECONSTRUCTION = "field_reconstruction"
    POTENTIAL_CORRECTION = "potential_correction"


@dataclass
class MultiTaskHeadConfig:
    """
    Configuration for auxiliary prediction heads on a shared latent.

    See ``docs/ml_encoder_strategy.md`` Phase 1 for target definitions.
    """

    objectives: list[TrainingObjective] = field(
        default_factory=lambda: [
            TrainingObjective.CONTRASTIVE,
            TrainingObjective.QUALITY_REGRESSION,
        ]
    )
    contrastive_temperature: float = 0.07
    quality_targets: tuple[str, ...] = (
        "dE_over_E0",
        "active_fraction",
        "mean_bin",
    )
    param_targets: tuple[str, ...] = (
        "halo.v0",
        "disk.mass",
    )


@dataclass
class TrainingLabels:
    """
    Scalar labels attached to one campaign snapshot.

    Populated from manifest rows and ``evolution/diagnostics.csv``.
    """

    run_hash: str
    label: str
    scalars: dict[str, float] = field(default_factory=dict)
    params: dict[str, float] = field(default_factory=dict)

    def vector(self, keys: tuple[str, ...], *, source: str = "scalars") -> np.ndarray:
        """Stack selected keys into a float vector (missing → NaN)."""
        store = self.scalars if source == "scalars" else self.params
        return np.array([float(store.get(k, np.nan)) for k in keys], dtype=np.float64)


class MultiTaskHeads:
    """
    Untrained stub for multi-task heads.

    Replace ``forward`` with ``nn.Module`` children once a training loop exists.
    """

    def __init__(self, *, latent_dim: int, config: MultiTaskHeadConfig | None = None) -> None:
        self.latent_dim = latent_dim
        self.config = config or MultiTaskHeadConfig()
        self._trained = False

    @property
    def trained(self) -> bool:
        return self._trained

    def predict_quality(self, latent: np.ndarray) -> dict[str, float]:
        """Stub: return zeros for each quality target."""
        return {k: 0.0 for k in self.config.quality_targets}

    def predict_params(self, latent: np.ndarray) -> dict[str, float]:
        """Stub: return zeros for each campaign parameter."""
        return {k: 0.0 for k in self.config.param_targets}

    def contrastive_loss(
        self,
        z_anchor: np.ndarray,
        z_positive: np.ndarray,
        z_negatives: list[np.ndarray] | None = None,
    ) -> float:
        """
        InfoNCE-style contrastive loss stub (numpy cosine similarity).

        Use for same-run augmentations (rotation, subsample) vs different runs.
        """
        if z_negatives is None:
            z_negatives = []
        anchor = z_anchor / (np.linalg.norm(z_anchor) + 1e-8)
        pos = z_positive / (np.linalg.norm(z_positive) + 1e-8)
        pos_sim = float(anchor @ pos)
        neg_sims = [
            float(anchor @ (n / (np.linalg.norm(n) + 1e-8))) for n in z_negatives
        ]
        logits = [pos_sim / self.config.contrastive_temperature, *[
            s / self.config.contrastive_temperature for s in neg_sims
        ]]
        exp_logits = np.exp(np.array(logits))
        return float(-np.log(exp_logits[0] / exp_logits.sum()))

    def training_summary(self) -> dict[str, Any]:
        return {
            "trained": self._trained,
            "objectives": [o.value for o in self.config.objectives],
            "latent_dim": self.latent_dim,
        }
