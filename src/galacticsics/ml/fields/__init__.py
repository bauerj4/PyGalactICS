"""Snapshot-as-fields: multi-scale slices/voxels, U-Net AE, and generative latents.

**Recommended generative path** for non-equilibrium morphology: freeze the crisp
multi-tower AE, build an **A₂-stratified** teacher feature library (with bar
rotations), compress pooled bottlenecks to a single ``z`` (PCA), sample with
``amplify_residual`` / ``uniform_knn`` / local-jitter ``z``-retrieve
(``scripts/sample_latent_ic.py``,
:class:`~galacticsics.ml.fields.feature_library.TeacherFeatureLibrary`).

Skip-synth :class:`~galacticsics.ml.fields.latent_code.FrozenAECodeVAE` and the
end-to-end :class:`~galacticsics.ml.fields.vae.MultiTowerSliceVAE` are research /
legacy priors (VAE prior path washes bars).

Shared COM centering is mandatory (:func:`~galacticsics.ml.fields.frame.prepare_shared_frame`).
Docs: ``docs/field_maps.md``, ``docs/ml_findings.md``,
``runs/ml/field_maps/CREATIVE_LATENT_SUMMARY.md``.
"""

from galacticsics.ml.fields.binning import (
    MOMENT_KEYS,
    MOMENT_KEYS_BASE,
    MOMENT_KEYS_FULL,
    ComponentSliceGrid,
    ComponentVoxelGrid,
    MultiScaleSliceConfig,
    MultiScaleVoxelConfig,
    SliceMapConfig,
    VoxelMapConfig,
    bin_multiscale_slice_stacks,
    bin_multiscale_voxel_stacks,
    bin_vertical_slice_stack,
    bin_voxel_stack,
    fuse_slice_maps_to_common,
    resolve_moment_keys,
    scale_summary,
)
from galacticsics.ml.fields.potential import (
    plummer_potential_multiscale,
    plummer_potential_on_xy_slices,
)
from galacticsics.ml.fields.resample import (
    bin_spherical_shell_moments,
    resample_bulge_from_spherical_shells,
    resample_particles_from_multiscale,
    resample_particles_from_slice_stack,
    stitch_retained_components,
)
from galacticsics.ml.fields.vae import FieldVAEConfig, MultiTowerSliceVAE, field_vae_loss
from galacticsics.ml.fields.latent_code import (
    FrozenAECodeVAE,
    LatentCodeConfig,
    latent_code_loss,
    load_frozen_teacher,
)
from galacticsics.ml.fields.feature_library import (
    FeatureLibraryConfig,
    TeacherFeatureLibrary,
    fit_index_basis,
)

__all__ = [
    "MOMENT_KEYS",
    "MOMENT_KEYS_BASE",
    "MOMENT_KEYS_FULL",
    "ComponentSliceGrid",
    "ComponentVoxelGrid",
    "FeatureLibraryConfig",
    "FieldVAEConfig",
    "FrozenAECodeVAE",
    "LatentCodeConfig",
    "MultiScaleSliceConfig",
    "MultiScaleVoxelConfig",
    "MultiTowerSliceVAE",
    "SliceMapConfig",
    "TeacherFeatureLibrary",
    "VoxelMapConfig",
    "bin_multiscale_slice_stacks",
    "bin_multiscale_voxel_stacks",
    "bin_vertical_slice_stack",
    "bin_voxel_stack",
    "field_vae_loss",
    "fit_index_basis",
    "fuse_slice_maps_to_common",
    "latent_code_loss",
    "load_frozen_teacher",
    "resolve_moment_keys",
    "scale_summary",
    "plummer_potential_multiscale",
    "plummer_potential_on_xy_slices",
    "resample_particles_from_multiscale",
    "resample_particles_from_slice_stack",
    "resample_bulge_from_spherical_shells",
    "bin_spherical_shell_moments",
    "stitch_retained_components",
]
