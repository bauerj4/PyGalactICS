"""ML training utilities for GalactICS campaigns.

Package layout (roles):

* **``fields``** — multi-scale slice/voxel maps, U-Net recon AE, **conditional
  field VAE** (recommended for non-equilibrium morphology / IC sampling).
  See ``docs/field_maps.md``, ``scripts/smoke_field_vae.py``.
* **``models``** — particle-set Morton VAE + AR transformer (**baseline** for
  phase-space tokens; weaker morphology than fields).
  See ``docs/morton_generative.md``.
* **``morton``** — snapshot index, tokenization, on-the-fly particle datasets.
* **``conditioning``** — shared structural ``θ`` keys for field + particle VAEs.
* **``profiles``** — soft radial / Fourier / virial auxiliaries (particle path).
* **Field / coefficient stubs** — ``heads``, ``decoders``, ``training_data``
  (see ``docs/ml_encoder_strategy.md``).
"""

from galacticsics.ml.decoders import (
    ConditionalICDecoder,
    DecoderConfig,
    HarmonicCoefficientTarget,
    NonAxisymmetricPerturbation,
)
from galacticsics.ml.heads import MultiTaskHeadConfig, MultiTaskHeads, TrainingLabels, TrainingObjective
from galacticsics.ml.training_data import (
    CampaignTrainingBundle,
    CampaignTrainingRecord,
    export_campaign_training_bundle,
)

__all__ = [
    "CampaignTrainingBundle",
    "CampaignTrainingRecord",
    "ConditionalICDecoder",
    "DecoderConfig",
    "HarmonicCoefficientTarget",
    "MultiTaskHeadConfig",
    "MultiTaskHeads",
    "NonAxisymmetricPerturbation",
    "TrainingLabels",
    "TrainingObjective",
    "export_campaign_training_bundle",
]
