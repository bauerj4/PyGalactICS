"""Particle-set generative baselines (Morton-ordered tokens).

These models operate on subsampled particle sequences, not field maps.

* :class:`SequenceVAE` — conditional set VAE (phase-space / mix baseline).
* :class:`MortonTransformer` — autoregressive Morton-token LM.

For **morphology-first** non-equilibrium IC sampling (bars, spirals), prefer the
field conditional VAE in :mod:`galacticsics.ml.fields.vae` /
``scripts/smoke_field_vae.py``.  Keep this package for particle-level ablations
and phase-space diagnostics documented in ``docs/morton_generative.md``.
"""

from galacticsics.ml.models.sequence_vae import SequenceVAE, SequenceVAEConfig
from galacticsics.ml.models.morton_transformer import MortonTransformer, MortonTransformerConfig

__all__ = [
    "MortonTransformer",
    "MortonTransformerConfig",
    "SequenceVAE",
    "SequenceVAEConfig",
]
