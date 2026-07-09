"""Canonical preprocessing for particle-set encoders."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from galacticsics.representations.particle_features import ParticleFeatureBatch

# Columns fed to linear / convolutional encoders (type_id uses nn.Embedding instead).
ENCODER_FEATURE_NAMES: tuple[str, ...] = (
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "log_mass",
    "log_eps",
)

# Indices into the full :data:`~galacticsics.representations.particle_features.FEATURE_NAMES`
# matrix when exporting 9-column tensors.
_ENCODER_FEATURE_INDICES: tuple[int, ...] = tuple(range(8))


@dataclass
class PreprocessConfig:
    """
    Frame normalization applied before neural encoders.

    Parameters
    ----------
    center_com : bool
        Subtract mass-weighted centre of position and velocity.
    pca_align : bool
        Rotate positions and velocities into the principal-axis frame of the
        mass distribution (reduces orientation leakage into the latent).
    mass_weighted_com : bool
        Use particle masses when computing the centre of mass.
    """

    center_com: bool = True
    pca_align: bool = False
    mass_weighted_com: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreprocessResult:
    """
    Preprocessed batch plus bookkeeping for inverse transforms / provenance.

    Attributes
    ----------
    batch : ParticleFeatureBatch
        Copy with transformed kinematic columns; ``type_id`` unchanged.
    com_pos, com_vel : ndarray, shape (3,)
        Centres removed when ``center_com`` is enabled.
    rotation : ndarray, shape (3, 3) or None
        PCA rotation applied to positions and velocities.
    config : PreprocessConfig
        Settings used.
    """

    batch: ParticleFeatureBatch
    com_pos: np.ndarray
    com_vel: np.ndarray
    rotation: np.ndarray | None
    config: PreprocessConfig

    def metadata(self) -> dict[str, Any]:
        return {
            "center_com": self.config.center_com,
            "pca_align": self.config.pca_align,
            "mass_weighted_com": self.config.mass_weighted_com,
            "com_pos": self.com_pos.tolist(),
            "com_vel": self.com_vel.tolist(),
            "rotation": self.rotation.tolist() if self.rotation is not None else None,
        }


def encoder_features_from_batch(batch: ParticleFeatureBatch) -> np.ndarray:
    """
    Extract the 8-column encoder feature matrix (no ``type_id`` column).

    Parameters
    ----------
    batch : ParticleFeatureBatch
        Full 9-column export batch.

    Returns
    -------
    features : ndarray, shape (N, 8)
    """
    return batch.features[:, _ENCODER_FEATURE_INDICES].copy()


def preprocess_batch(
    batch: ParticleFeatureBatch,
    *,
    config: PreprocessConfig | None = None,
) -> PreprocessResult:
    """
    Apply COM centring and optional PCA alignment to a feature batch.

    Parameters
    ----------
    batch : ParticleFeatureBatch
        Raw tokens from :func:`~galacticsics.representations.particle_features.particle_state_to_batch`.
    config : PreprocessConfig, optional
        Defaults to COM centring only.

    Returns
    -------
    result : PreprocessResult
        Transformed batch and transform metadata.
    """
    cfg = config or PreprocessConfig()
    features = batch.features.copy()
    mask = batch.mask
    valid = features[mask]
    if valid.size == 0:
        return PreprocessResult(
            batch=batch,
            com_pos=np.zeros(3),
            com_vel=np.zeros(3),
            rotation=None,
            config=cfg,
        )

    masses = 10.0 ** valid[:, 6]
    weights = masses if cfg.mass_weighted_com else np.ones(len(valid))

    com_pos = np.zeros(3)
    com_vel = np.zeros(3)
    if cfg.center_com:
        wsum = weights.sum()
        com_pos = np.average(valid[:, :3], axis=0, weights=weights)
        com_vel = np.average(valid[:, 3:6], axis=0, weights=weights)
        features[mask, :3] -= com_pos
        features[mask, 3:6] -= com_vel

    rotation: np.ndarray | None = None
    if cfg.pca_align:
        pos = features[mask, :3]
        cov = np.cov(pos.T, aweights=weights)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        rotation = evecs[:, order]
        features[mask, :3] = pos @ rotation
        features[mask, 3:6] = features[mask, 3:6] @ rotation

    out_batch = ParticleFeatureBatch(
        features=features,
        type_id=batch.type_id.copy(),
        tags=batch.tags,
        n_particles=batch.n_particles,
        n_features=batch.n_features,
        mask=mask.copy(),
    )
    return PreprocessResult(
        batch=out_batch,
        com_pos=com_pos,
        com_vel=com_vel,
        rotation=rotation,
        config=cfg,
    )
