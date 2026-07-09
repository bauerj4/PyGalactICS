"""kNN graph encoder stub for particle snapshots (optional torch)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from galacticsics.representations.learned import (
    EncoderBackend,
    EncoderMetadata,
    LearnedEncoderConfig,
    LearnedRepresentation,
)
from galacticsics.representations.particle_features import ParticleFeatureBatch
from galacticsics.representations.preprocess import (
    ENCODER_FEATURE_NAMES,
    PreprocessConfig,
    encoder_features_from_batch,
    preprocess_batch,
)


@dataclass
class GraphConfig:
    """
    kNN graph construction settings.

    Parameters
    ----------
    k_neighbors : int
        Edges per node (symmetrized).
    max_nodes : int
        Subsample particles when ``N > max_nodes`` (training / stub inference).
    seed : int
        RNG seed for subsampling.
    """

    k_neighbors: int = 16
    max_nodes: int = 4096
    seed: int = 0


def build_knn_edges(
    pos: np.ndarray,
    *,
    k: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build undirected kNN edge indices from positions.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
    k : int
        Neighbors per node (excluding self).

    Returns
    -------
    edge_index : ndarray, shape (2, E)
        COO format ``(source, target)`` pairs.
    edge_weight : ndarray, shape (E,)
        Inverse-distance weights.
    """
    n = len(pos)
    if n == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float64)
    k = min(k, n - 1)
    if k < 1:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float64)

    # O(N²) brute force — replace with scipy.spatial.cKDTree for production
    d2 = np.sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    nn_idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]

    sources: list[int] = []
    targets: list[int] = []
    weights: list[float] = []
    for i in range(n):
        for j in nn_idx[i]:
            dist = np.sqrt(d2[i, j])
            w = 1.0 / max(dist, 1e-6)
            sources.extend([i, int(j)])
            targets.extend([int(j), i])
            weights.extend([w, w])

    edge_index = np.array([sources, targets], dtype=np.int64)
    edge_weight = np.asarray(weights, dtype=np.float64)
    return edge_index, edge_weight


def subsample_batch(
    batch: ParticleFeatureBatch,
    *,
    max_nodes: int,
    seed: int = 0,
) -> ParticleFeatureBatch:
    """Random subsample when particle count exceeds ``max_nodes``."""
    if batch.n_particles <= max_nodes:
        return batch
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(batch.n_particles, size=max_nodes, replace=False))
    mask = np.zeros(batch.n_particles, dtype=bool)
    mask[idx] = True
    return ParticleFeatureBatch(
        features=batch.features[idx],
        type_id=batch.type_id[idx],
        tags=batch.tags[idx] if batch.tags is not None else None,
        n_particles=max_nodes,
        n_features=batch.n_features,
        mask=np.ones(max_nodes, dtype=bool),
    )


class ParticleGraphEncoder:
    """
    Untrained message-passing stub.

    Performs one round of neighbor mean aggregation in numpy; swap in e3nn /
    PyG layers for equivariant or deeper models.
    """

    def __init__(
        self,
        *,
        graph: GraphConfig | None = None,
        config: LearnedEncoderConfig | None = None,
        preprocess: PreprocessConfig | None = None,
    ) -> None:
        self.graph = graph or GraphConfig()
        self._preprocess = preprocess or PreprocessConfig()
        self._config = config or LearnedEncoderConfig(backend=EncoderBackend.GRAPH)
        self._metadata = EncoderMetadata(
            trained=False,
            backend=EncoderBackend.GRAPH.value,
            preprocessing=self._preprocess.as_dict(),
        )

    @property
    def config(self) -> LearnedEncoderConfig:
        return self._config

    @property
    def metadata(self) -> EncoderMetadata:
        return self._metadata

    def encode(self, batch: ParticleFeatureBatch) -> LearnedRepresentation:
        prep = preprocess_batch(batch, config=self._preprocess)
        sub = subsample_batch(
            prep.batch,
            max_nodes=self.graph.max_nodes,
            seed=self.graph.seed,
        )
        feats = encoder_features_from_batch(sub)
        pos = feats[:, :3]
        edge_index, edge_weight = build_knn_edges(pos, k=self.graph.k_neighbors)

        node_feat = feats
        n_nodes = len(node_feat)
        if edge_index.shape[1] == 0:
            pooled = node_feat.mean(axis=0)
        else:
            agg = np.zeros_like(node_feat)
            counts = np.zeros(n_nodes, dtype=np.float64)
            src, tgt = edge_index
            for s, t, w in zip(src, tgt, edge_weight, strict=True):
                agg[t] += node_feat[s] * w
                counts[t] += w
            with np.errstate(divide="ignore", invalid="ignore"):
                agg = np.where(counts[:, None] > 0, agg / counts[:, None], node_feat)
            pooled = agg.mean(axis=0)

        latent = np.zeros(self._config.d_model, dtype=np.float64)
        n = min(len(pooled), self._config.d_model)
        latent[:n] = pooled[:n]

        return LearnedRepresentation(
            latent=latent,
            per_particle=node_feat[:, : self._config.d_model]
            if node_feat.shape[1] >= self._config.d_model
            else None,
            config=self._config,
            feature_names=ENCODER_FEATURE_NAMES,
            metadata=self._metadata,
            graph_edges=edge_index,
        )
