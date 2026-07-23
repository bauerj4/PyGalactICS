"""Tests for particle feature export and learned representations."""

from __future__ import annotations

import numpy as np
import pytest

from galacticsics.representations import RepresentationKind
from galacticsics.representations.learned import (
    LearnedEncoderConfig,
    mean_pool_encoder,
    particles_to_learned_artifact,
    torch_available,
)
from galacticsics.representations.particle_features import (
    FEATURE_NAMES,
    ParticleFeatureBatch,
    particles_to_feature_matrix,
)


def test_feature_matrix_shape_and_columns():
    n = 5
    pos = np.random.randn(n, 3)
    vel = np.random.randn(n, 3)
    mass = np.full(n, 1.0)
    eps = np.full(n, 0.01)
    type_id = np.arange(n, dtype=np.int32)

    feats = particles_to_feature_matrix(pos, vel, mass, eps, type_id)
    assert feats.shape == (n, len(FEATURE_NAMES))
    np.testing.assert_array_equal(feats[:, :3], pos)
    np.testing.assert_array_equal(feats[:, 3:6], vel)
    assert np.allclose(feats[:, 6], 0.0)  # log10(1)
    assert np.allclose(feats[:, 8], type_id.astype(float))


def test_particle_feature_batch_as_dict():
    n, f = 3, 9
    batch = ParticleFeatureBatch(
        features=np.zeros((n, f)),
        type_id=np.zeros(n, dtype=np.int32),
        tags=None,
        n_particles=n,
        n_features=f,
        mask=np.ones(n, dtype=bool),
    )
    d = batch.as_dict()
    assert set(d) == {"features", "type_id", "mask"}


def test_mean_pool_encoder_latent_dim():
    n = 10
    batch = ParticleFeatureBatch(
        features=np.ones((n, 9)),
        type_id=np.zeros(n, dtype=np.int32),
        tags=None,
        n_particles=n,
        n_features=9,
        mask=np.ones(n, dtype=bool),
    )
    cfg = LearnedEncoderConfig(d_model=32)
    rep = mean_pool_encoder(batch, config=cfg)
    assert rep.latent.shape == (32,)
    assert rep.config.d_model == 32


def test_particles_to_learned_artifact_kind():
    from ntropy.ics.plummer import sample_plummer

    state = sample_plummer(seed=0)
    artifact = particles_to_learned_artifact(state)
    assert artifact.kind == RepresentationKind.LEARNED
    assert artifact.data.latent.ndim == 1


@pytest.mark.skipif(not torch_available(), reason="PyTorch not installed")
def test_transformer_encoder_forward():
    import torch

    from galacticsics.integrations.torch_encoder import ParticleTransformerEncoder
    from galacticsics.representations.preprocess import ENCODER_FEATURE_NAMES

    enc = ParticleTransformerEncoder(d_model=32, n_heads=4, n_layers=1)
    b, n = 2, 8
    feat = torch.randn(b, n, len(ENCODER_FEATURE_NAMES))
    tid = torch.zeros(b, n, dtype=torch.long)
    mask = torch.ones(b, n, dtype=torch.bool)
    latent, hidden = enc(feat, tid, mask)
    assert latent.shape == (b, 32)
    assert hidden.shape == (b, n + 1, 32)
