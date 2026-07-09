"""PyTorch transformer encoder for particle token sequences (optional dependency)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

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


class ParticleTransformerEncoder(nn.Module):
    """
    Transformer encoder over per-particle feature tokens.

    Each particle is a token: linear projection of 8 kinematic/scalar features
    plus a learned type embedding (``type_id`` is **not** in the linear input).
    A learnable CLS token is prepended; its final hidden state is the
    galaxy-level latent.

    .. warning::

        This is an **untrained scaffold**.  Full self-attention is O(N²) and
        does not scale to MW production particle counts without subsampling.
        Prefer :class:`~galacticsics.integrations.field_encoder.ParticleFieldEncoder`
        for campaign-scale work (see ``docs/ml_encoder_strategy.md``).

    Parameters
    ----------
    n_features : int
        Input feature dimension per token (default 8 — no ``type_id`` column).
    type_vocab_size : int
        Number of particle types for ``nn.Embedding``.
    d_model, n_heads, n_layers, dropout : int / float
        Standard transformer hyperparameters.
    """

    def __init__(
        self,
        *,
        n_features: int = len(ENCODER_FEATURE_NAMES),
        type_vocab_size: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        config: LearnedEncoderConfig | None = None,
        preprocess: PreprocessConfig | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.feature_proj = nn.Linear(n_features, d_model)
        self.type_embed = nn.Embedding(type_vocab_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self._preprocess = preprocess or PreprocessConfig()
        self._config = config or LearnedEncoderConfig(
            backend=EncoderBackend.TRANSFORMER,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            type_vocab_size=type_vocab_size,
            dropout=dropout,
            preprocess=self._preprocess,
        )
        self._metadata = EncoderMetadata(
            trained=False,
            backend=EncoderBackend.TRANSFORMER.value,
            preprocessing=self._preprocess.as_dict(),
        )

    def forward(
        self,
        features: torch.Tensor,
        type_id: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of particle sequences.

        Parameters
        ----------
        features : Tensor, shape (B, N, F)
            Eight-column encoder features (no type_id).
        type_id : Tensor, shape (B, N), dtype long
        mask : Tensor, shape (B, N), optional
            True for valid tokens.

        Returns
        -------
        latent : Tensor, shape (B, d_model)
        per_token : Tensor, shape (B, N+1, d_model)
            Includes CLS at index 0.
        """
        x = self.feature_proj(features) + self.type_embed(type_id.clamp(min=0))
        b = x.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)

        if mask is not None:
            cls_mask = torch.ones(b, 1, dtype=torch.bool, device=mask.device)
            full_mask = torch.cat([cls_mask, mask], dim=1)
            src_key_padding_mask = ~full_mask
        else:
            src_key_padding_mask = None

        hidden = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return hidden[:, 0, :], hidden

    @torch.no_grad()
    def encode_batch(self, batch: ParticleFeatureBatch) -> LearnedRepresentation:
        """Encode a single :class:`ParticleFeatureBatch` (batch size 1)."""
        self.eval()
        prep = preprocess_batch(batch, config=self._preprocess)
        feats = encoder_features_from_batch(prep.batch)
        feat = torch.as_tensor(feats, dtype=torch.float32).unsqueeze(0)
        tid = torch.as_tensor(prep.batch.type_id, dtype=torch.long).unsqueeze(0)
        msk = torch.as_tensor(prep.batch.mask, dtype=torch.bool).unsqueeze(0)
        latent_t, hidden_t = self(feat, tid, msk)
        per_particle = hidden_t[0, 1:, :].cpu().numpy()
        meta = EncoderMetadata(
            trained=self._metadata.trained,
            backend=self._metadata.backend,
            preprocessing=prep.metadata(),
            checkpoint_path=self._metadata.checkpoint_path,
        )
        return LearnedRepresentation(
            latent=latent_t[0].cpu().numpy(),
            per_particle=per_particle,
            config=self._config,
            feature_names=ENCODER_FEATURE_NAMES,
            metadata=meta,
        )


class TorchTransformerWrapper:
    """Adapter implementing :class:`~galacticsics.representations.learned.ParticleEncoder`."""

    def __init__(self, module: ParticleTransformerEncoder) -> None:
        self.module = module

    def encode(self, batch: ParticleFeatureBatch) -> LearnedRepresentation:
        return self.module.encode_batch(batch)


def build_transformer_module(
    config: LearnedEncoderConfig | None = None,
) -> ParticleTransformerEncoder:
    """Build :class:`ParticleTransformerEncoder` from config."""
    cfg = config or LearnedEncoderConfig(backend=EncoderBackend.TRANSFORMER)
    return ParticleTransformerEncoder(
        n_features=len(ENCODER_FEATURE_NAMES),
        type_vocab_size=cfg.type_vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        config=cfg,
        preprocess=cfg.preprocess,
    )
