"""Learned (deep learning / transformer) model representations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np

from galacticsics.representations import ModelArtifact, RepresentationKind
from galacticsics.representations.particle_features import (
    FEATURE_NAMES,
    ParticleFeatureBatch,
    particle_state_to_batch,
)
from galacticsics.representations.preprocess import PreprocessConfig


class EncoderBackend(str, Enum):
    """
    Encoder architecture family.

    See ``docs/ml_encoder_strategy.md`` for when to use each backend.
    """

    MEAN_POOL = "mean_pool"
    TRANSFORMER = "transformer"
    FIELD = "field"
    GRAPH = "graph"
    PERCEIVER = "perceiver"  # reserved — not implemented yet


@dataclass
class EncoderMetadata:
    """
    Provenance for a single encoding call.

    Attributes
    ----------
    trained : bool
        ``False`` for scaffold / randomly initialized weights.
    backend : str
        Value of :class:`EncoderBackend`.
    preprocessing : dict
        Serialized :class:`~galacticsics.representations.preprocess.PreprocessConfig`.
    checkpoint_path : str or None
        Weights file when loaded from disk.
    """

    trained: bool = False
    backend: str = EncoderBackend.MEAN_POOL.value
    preprocessing: dict[str, Any] = field(default_factory=dict)
    checkpoint_path: str | None = None


@dataclass
class LearnedEncoderConfig:
    """
    Hyperparameters for a particle-set encoder.

    Parameters
    ----------
    backend : EncoderBackend
        Architecture family (see :class:`EncoderBackend`).
    model_type : str
        Legacy alias for ``backend.value`` (kept for compatibility).
    d_model : int
        Latent width / embedding dimension.
    n_heads : int
        Attention heads (transformer / perceiver only).
    n_layers : int
        Encoder depth.
    max_particles : int
        Pad/truncate sequences for batched transformer training.
    type_vocab_size : int
        Distinct particle types for learned embeddings.
    dropout : float
        Dropout rate during training.
    preprocess : PreprocessConfig
        Frame normalization before encoding.
    field_n_bins : int
        Cubic grid side for field encoder.
    field_r_max : float
        Field binning half-width [kpc].
    graph_k : int
        kNN degree for graph encoder.
    graph_max_nodes : int
        Subsample cap for graph encoder.
    """

    backend: EncoderBackend = EncoderBackend.MEAN_POOL
    model_type: str = "mean_pool"
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    max_particles: int = 4096
    type_vocab_size: int = 16
    dropout: float = 0.1
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    field_n_bins: int = 32
    field_r_max: float = 30.0
    graph_k: int = 16
    graph_max_nodes: int = 4096

    def __post_init__(self) -> None:
        if isinstance(self.backend, str):
            self.backend = EncoderBackend(self.backend)
        if self.model_type == "mean_pool" and self.backend != EncoderBackend.MEAN_POOL:
            self.model_type = self.backend.value


@dataclass
class LearnedRepresentation:
    """
    Latent encoding of a particle snapshot.

    Attributes
    ----------
    latent : ndarray, shape (d_model,)
        Pooled galaxy-level embedding.
    per_particle : ndarray or None, shape (N, d_model)
        Optional per-token embeddings before pooling.
    config : LearnedEncoderConfig
        Architecture used for this encoding.
    feature_names : tuple of str
        Input feature column names.
    metadata : EncoderMetadata or None
        Training status and preprocessing provenance.
    field : ndarray or None
        Binned field tensor when using field encoder.
    graph_edges : ndarray or None
        Edge index ``(2, E)`` when using graph encoder.
    """

    latent: np.ndarray
    per_particle: np.ndarray | None
    config: LearnedEncoderConfig
    feature_names: tuple[str, ...] = FEATURE_NAMES
    metadata: EncoderMetadata | None = None
    field: np.ndarray | None = None
    graph_edges: np.ndarray | None = None


@runtime_checkable
class ParticleEncoder(Protocol):
    """Protocol for encoders that map particle batches to latent vectors."""

    def encode(self, batch: ParticleFeatureBatch) -> LearnedRepresentation:
        """Encode one particle snapshot."""
        ...


def mean_pool_encoder(
    batch: ParticleFeatureBatch,
    *,
    config: LearnedEncoderConfig | None = None,
) -> LearnedRepresentation:
    """
    Numpy baseline: per-feature mean → latent vector.

    Useful for tests and as a fallback when PyTorch is not installed.
    """
    from galacticsics.representations.preprocess import (
        ENCODER_FEATURE_NAMES,
        encoder_features_from_batch,
        preprocess_batch,
    )

    cfg = config or LearnedEncoderConfig()
    prep = preprocess_batch(batch, config=cfg.preprocess)
    feats = encoder_features_from_batch(prep.batch)
    masked = feats[prep.batch.mask]
    if masked.size == 0:
        latent = np.zeros(cfg.d_model, dtype=np.float64)
    else:
        pooled = masked.mean(axis=0)
        latent = np.zeros(cfg.d_model, dtype=np.float64)
        n = min(len(pooled), cfg.d_model)
        latent[:n] = pooled[:n]
    return LearnedRepresentation(
        latent=latent,
        per_particle=None,
        config=cfg,
        feature_names=ENCODER_FEATURE_NAMES,
        metadata=EncoderMetadata(
            trained=False,
            backend=EncoderBackend.MEAN_POOL.value,
            preprocessing=prep.metadata(),
        ),
    )


class MeanPoolEncoderWrapper:
    """Adapter implementing :class:`ParticleEncoder`."""

    def __init__(self, config: LearnedEncoderConfig | None = None) -> None:
        self._config = config or LearnedEncoderConfig(backend=EncoderBackend.MEAN_POOL)

    def encode(self, batch: ParticleFeatureBatch) -> LearnedRepresentation:
        return mean_pool_encoder(batch, config=self._config)


def particles_to_learned_artifact(
    state,
    *,
    encoder: ParticleEncoder | None = None,
    config: LearnedEncoderConfig | None = None,
) -> ModelArtifact:
    """
    Wrap a particle snapshot as a :class:`~galacticsics.representations.ModelArtifact`.
    """
    batch = particle_state_to_batch(state)
    if encoder is None:
        rep = mean_pool_encoder(batch, config=config)
    else:
        rep = encoder.encode(batch)
    return ModelArtifact(
        kind=RepresentationKind.LEARNED,
        data=rep,
        provenance=["particle_state_to_batch", "encode"],
    )


def torch_available() -> bool:
    """Return True if PyTorch is importable."""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def build_encoder(
    config: LearnedEncoderConfig | None = None,
) -> ParticleEncoder:
    """
    Factory for particle encoders.

    Parameters
    ----------
    config : LearnedEncoderConfig, optional
        Selects backend and hyperparameters.

    Returns
    -------
    encoder : ParticleEncoder
        Callable wrapper with ``encode(batch)`` method.

    Raises
    ------
    ImportError
        If a torch backend is requested without PyTorch installed.
    NotImplementedError
        For reserved backends (e.g. perceiver).
    """
    cfg = config or LearnedEncoderConfig()
    backend = cfg.backend

    if backend == EncoderBackend.MEAN_POOL:
        return MeanPoolEncoderWrapper(cfg)

    if backend == EncoderBackend.TRANSFORMER:
        if not torch_available():
            raise ImportError(
                "PyTorch is required for transformer encoders. "
                "Install with: pip install galacticsics[ml]"
            )
        from galacticsics.integrations.torch_encoder import TorchTransformerWrapper, build_transformer_module

        return TorchTransformerWrapper(build_transformer_module(cfg))

    if backend == EncoderBackend.FIELD:
        from galacticsics.integrations.field_encoder import FieldGridConfig, ParticleFieldEncoder

        grid = FieldGridConfig(
            n_bins=cfg.field_n_bins,
            r_max=cfg.field_r_max,
            type_vocab_size=cfg.type_vocab_size,
        )
        return ParticleFieldEncoder(grid=grid, config=cfg, preprocess=cfg.preprocess)

    if backend == EncoderBackend.GRAPH:
        from galacticsics.integrations.graph_encoder import GraphConfig, ParticleGraphEncoder

        graph = GraphConfig(
            k_neighbors=cfg.graph_k,
            max_nodes=cfg.graph_max_nodes,
        )
        return ParticleGraphEncoder(graph=graph, config=cfg, preprocess=cfg.preprocess)

    if backend == EncoderBackend.PERCEIVER:
        raise NotImplementedError(
            "Perceiver encoder is reserved for future work. "
            "Use EncoderBackend.FIELD or EncoderBackend.GRAPH for scalable stubs."
        )

    raise ValueError(f"Unknown encoder backend: {backend!r}")


def build_transformer_encoder(
    config: LearnedEncoderConfig | None = None,
) -> Any:
    """
    Build a PyTorch transformer encoder module (legacy alias).

    Prefer :func:`build_encoder` with ``backend=EncoderBackend.TRANSFORMER``.
    """
    cfg = config or LearnedEncoderConfig(backend=EncoderBackend.TRANSFORMER)
    if not torch_available():
        raise ImportError(
            "PyTorch is required for transformer encoders. "
            "Install with: pip install galacticsics[ml]"
        )
    from galacticsics.integrations.torch_encoder import build_transformer_module

    return build_transformer_module(cfg)
