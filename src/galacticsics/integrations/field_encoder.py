"""Field-based particle encoder: bin (rho, v) grids then CNN / MLP (optional torch)."""

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
class FieldGridConfig:
    """
  3D binning settings for field encoders.

    Parameters
    ----------
    n_bins : int
        Cubic grid side length (``n_bins³`` cells).
    r_max : float
        Half-width of the cubical domain [kpc]; particles outside are clipped.
    channels_per_type : int
        Number of stacked channels per particle type (default: rho + |v|).
    type_vocab_size : int
        Distinct type slots in the channel stack.
    """

    n_bins: int = 32
    r_max: float = 30.0
    channels_per_type: int = 2
    type_vocab_size: int = 8


def particles_to_field_tensor(
    batch: ParticleFeatureBatch,
    *,
    grid: FieldGridConfig | None = None,
) -> np.ndarray:
    """
    Bin a particle snapshot into a multi-channel density / speed field.

    This is the recommended scalable representation for MW-scale campaigns
  (see ``docs/ml_encoder_strategy.md``).  Each particle type contributes
    ``channels_per_type`` channels: mass-weighted density and mean speed norm.

    Parameters
    ----------
    batch : ParticleFeatureBatch
        Particle tokens (uses COM-centred positions when preprocessed).
    grid : FieldGridConfig, optional
        Bin geometry.

    Returns
    -------
    field : ndarray, shape (C, G, G, G)
        ``C = type_vocab_size * channels_per_type``.
    """
    gcfg = grid or FieldGridConfig()
    n_bins = gcfg.n_bins
    r_max = gcfg.r_max
    cpt = gcfg.channels_per_type
    n_types = gcfg.type_vocab_size
    n_channels = n_types * cpt
    field = np.zeros((n_channels, n_bins, n_bins, n_bins), dtype=np.float64)

    feats = encoder_features_from_batch(batch)
    mask = batch.mask
    pos = feats[mask, :3]
    vel = feats[mask, 3:6]
    log_mass = feats[mask, 6]
    mass = 10.0 ** log_mass
    speed = np.linalg.norm(vel, axis=1)
    type_id = batch.type_id[mask].astype(int)

    edges = np.linspace(-r_max, r_max, n_bins + 1)
    for i in range(len(pos)):
        t = int(np.clip(type_id[i], 0, n_types - 1))
        ix = np.searchsorted(edges, pos[i, 0], side="right") - 1
        iy = np.searchsorted(edges, pos[i, 1], side="right") - 1
        iz = np.searchsorted(edges, pos[i, 2], side="right") - 1
        if not (0 <= ix < n_bins and 0 <= iy < n_bins and 0 <= iz < n_bins):
            continue
        base = t * cpt
        field[base, ix, iy, iz] += mass[i]
        field[base + 1, ix, iy, iz] += speed[i] * mass[i]

    # Normalize speed channel by local mass deposit
    for t in range(n_types):
        rho = field[t * cpt]
        spd = field[t * cpt + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            field[t * cpt + 1] = np.where(rho > 0, spd / rho, 0.0)

    return field


class ParticleFieldEncoder:
    """
    Untrained field encoder stub: flatten binned fields → linear projection.

    Replace the ``field_mlp`` module with a small 3D CNN or ViT once training
    data from campaigns is available.
    """

    def __init__(
        self,
        *,
        grid: FieldGridConfig | None = None,
        config: LearnedEncoderConfig | None = None,
        preprocess: PreprocessConfig | None = None,
    ) -> None:
        self.grid = grid or FieldGridConfig()
        self._preprocess = preprocess or PreprocessConfig()
        self._config = config or LearnedEncoderConfig(backend=EncoderBackend.FIELD)
        self._metadata = EncoderMetadata(
            trained=False,
            backend=EncoderBackend.FIELD.value,
            preprocessing=self._preprocess.as_dict(),
        )
        self._field_mlp = None
        if _torch_available():
            self._init_torch()

    def _init_torch(self) -> None:
        import torch
        from torch import nn

        n_ch = self.grid.type_vocab_size * self.grid.channels_per_type
        flat = n_ch * self.grid.n_bins**3
        d_model = self._config.d_model
        self._field_mlp = nn.Sequential(
            nn.Linear(flat, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        for p in self._field_mlp.parameters():
            nn.init.zeros_(p) if p.ndim == 1 else nn.init.xavier_uniform_(p)

    @property
    def config(self) -> LearnedEncoderConfig:
        return self._config

    @property
    def metadata(self) -> EncoderMetadata:
        return self._metadata

    def field_tensor(self, batch: ParticleFeatureBatch) -> np.ndarray:
        """Return binned field without encoding."""
        prep = preprocess_batch(batch, config=self._preprocess)
        return particles_to_field_tensor(prep.batch, grid=self.grid)

    def encode(self, batch: ParticleFeatureBatch) -> LearnedRepresentation:
        field = self.field_tensor(batch)
        if self._field_mlp is None:
            latent = _numpy_field_latent(field, d_model=self._config.d_model)
            return LearnedRepresentation(
                latent=latent,
                per_particle=None,
                config=self._config,
                feature_names=ENCODER_FEATURE_NAMES,
                metadata=self._metadata,
                field=field,
            )
        import torch

        self._field_mlp.eval()
        with torch.no_grad():
            flat = torch.as_tensor(field.ravel(), dtype=torch.float32).unsqueeze(0)
            latent_t = self._field_mlp(flat)[0]
        return LearnedRepresentation(
            latent=latent_t.cpu().numpy(),
            per_particle=None,
            config=self._config,
            feature_names=ENCODER_FEATURE_NAMES,
            metadata=self._metadata,
            field=field,
        )


def _numpy_field_latent(field: np.ndarray, *, d_model: int) -> np.ndarray:
    """Deterministic numpy fallback: log-scaled channel means padded to d_model."""
    per_ch = field.reshape(field.shape[0], -1).mean(axis=1)
    per_ch = np.sign(per_ch) * np.log1p(np.abs(per_ch))
    latent = np.zeros(d_model, dtype=np.float64)
    n = min(len(per_ch), d_model)
    latent[:n] = per_ch[:n]
    return latent


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False
