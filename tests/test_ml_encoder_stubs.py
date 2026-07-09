"""Tests for ML encoder stubs, preprocessing, and training export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from galacticsics.representations.learned import (
    EncoderBackend,
    LearnedEncoderConfig,
    build_encoder,
    torch_available,
)
from galacticsics.representations.particle_features import (
    ParticleFeatureBatch,
    particles_to_feature_matrix,
)
from galacticsics.representations.preprocess import (
    ENCODER_FEATURE_NAMES,
    PreprocessConfig,
    encoder_features_from_batch,
    preprocess_batch,
)
from galacticsics.integrations.field_encoder import (
    FieldGridConfig,
    particles_to_field_tensor,
)
from galacticsics.integrations.graph_encoder import build_knn_edges
from galacticsics.ml.decoders import ConditionalICDecoder
from galacticsics.ml.heads import MultiTaskHeads, TrainingLabels


def _dummy_batch(n: int = 20) -> ParticleFeatureBatch:
    pos = np.random.randn(n, 3)
    vel = np.random.randn(n, 3)
    mass = np.full(n, 1.0)
    eps = np.full(n, 0.01)
    type_id = np.zeros(n, dtype=np.int32)
    type_id[: n // 2] = 1
    features = particles_to_feature_matrix(pos, vel, mass, eps, type_id)
    return ParticleFeatureBatch(
        features=features,
        type_id=type_id,
        tags=None,
        n_particles=n,
        n_features=features.shape[1],
        mask=np.ones(n, dtype=bool),
    )


def test_encoder_features_exclude_type_id():
    batch = _dummy_batch()
    enc_feats = encoder_features_from_batch(batch)
    assert enc_feats.shape == (batch.n_particles, len(ENCODER_FEATURE_NAMES))
    assert "type_id" not in ENCODER_FEATURE_NAMES


def test_preprocess_com_centers_positions():
    batch = _dummy_batch()
    batch.features[:, :3] += np.array([10.0, 0.0, 0.0])
    result = preprocess_batch(batch, config=PreprocessConfig(center_com=True))
    centered = result.batch.features[result.batch.mask, :3]
    assert np.allclose(centered.mean(axis=0), 0.0, atol=1e-10)


def test_build_encoder_mean_pool():
    enc = build_encoder(LearnedEncoderConfig(backend=EncoderBackend.MEAN_POOL))
    rep = enc.encode(_dummy_batch())
    assert rep.latent.shape == (128,)
    assert rep.metadata is not None
    assert rep.metadata.trained is False
    assert rep.metadata.backend == "mean_pool"


def test_build_encoder_field():
    cfg = LearnedEncoderConfig(backend=EncoderBackend.FIELD, d_model=64, field_n_bins=8)
    enc = build_encoder(cfg)
    rep = enc.encode(_dummy_batch())
    assert rep.latent.shape == (64,)
    assert rep.field is not None
    assert rep.field.ndim == 4


def test_particles_to_field_tensor_shape():
    batch = _dummy_batch()
    grid = FieldGridConfig(n_bins=8, type_vocab_size=4)
    field = particles_to_field_tensor(batch, grid=grid)
    assert field.shape == (grid.type_vocab_size * grid.channels_per_type, 8, 8, 8)


def test_build_encoder_graph():
    enc = build_encoder(LearnedEncoderConfig(backend=EncoderBackend.GRAPH, d_model=32))
    rep = enc.encode(_dummy_batch())
    assert rep.latent.shape == (32,)
    assert rep.graph_edges is not None


def test_build_encoder_perceiver_raises():
    with pytest.raises(NotImplementedError):
        build_encoder(LearnedEncoderConfig(backend=EncoderBackend.PERCEIVER))


def test_knn_edges_symmetric():
    pos = np.random.randn(10, 3)
    edge_index, weights = build_knn_edges(pos, k=3)
    assert edge_index.shape[0] == 2
    assert len(weights) == edge_index.shape[1]


@pytest.mark.skipif(not torch_available(), reason="PyTorch not installed")
def test_transformer_uses_eight_features():
    import torch

    from galacticsics.integrations.torch_encoder import ParticleTransformerEncoder

    enc = ParticleTransformerEncoder(d_model=32, n_heads=4, n_layers=1)
    b, n = 2, 8
    feat = torch.randn(b, n, len(ENCODER_FEATURE_NAMES))
    tid = torch.zeros(b, n, dtype=torch.long)
    latent, hidden = enc(feat, tid)
    assert latent.shape == (b, 32)
    assert hidden.shape == (b, n + 1, 32)


def test_multitask_heads_contrastive_finite():
    heads = MultiTaskHeads(latent_dim=16)
    z = np.random.randn(16)
    loss = heads.contrastive_loss(z, z + 0.01, [np.random.randn(16)])
    assert np.isfinite(loss)


def test_conditional_decoder_stub():
    dec = ConditionalICDecoder()
    coeffs, pert = dec.decode(np.random.randn(128))
    assert "stub_l2_m0" in coeffs.coeffs
    assert pert.triaxiality >= 0.0


def test_training_labels_vector():
    labels = TrainingLabels(
        run_hash="abc",
        label="test",
        scalars={"dE_over_E0": 0.01},
        params={"halo.v0": 220.0},
    )
    v = labels.vector(("dE_over_E0", "missing"))
    assert v[0] == 0.01
    assert np.isnan(v[1])


def test_export_campaign_training_bundle_empty(tmp_path: Path):
    from galacticsics.ml.training_data import export_campaign_training_bundle

    bundle = export_campaign_training_bundle(tmp_path)
    assert bundle.records == []
    manifest = tmp_path / "ml_training" / "training_manifest.json"
    assert manifest.is_file()
    data = json.loads(manifest.read_text())
    assert data["n_records"] == 0
