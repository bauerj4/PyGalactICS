"""Optional integrations with sibling packages."""

from ntropy.integrations.galacticsics import (
    GalactICSSampleResult,
    galacticsics_available,
    merge_galacticsics_components,
    nfw_halo_model_fast,
    particle_state_from_galacticsics,
    require_galacticsics,
    sample_galacticsics_galaxy,
    sample_galacticsics_halo,
)

__all__ = [
    "GalactICSSampleResult",
    "galacticsics_available",
    "require_galacticsics",
    "particle_state_from_galacticsics",
    "merge_galacticsics_components",
    "nfw_halo_model_fast",
    "sample_galacticsics_halo",
    "sample_galacticsics_galaxy",
]
