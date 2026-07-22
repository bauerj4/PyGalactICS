"""Campaign and notebook diagnostics."""

from galacticsics.diagnostics.df_validation import (
    DEFAULT_THRESHOLDS,
    validate_bulge_df,
    validate_disk_df,
    validate_halo_df,
    validate_ic_distribution_functions,
)
from galacticsics.diagnostics.profiles import nfw_density_profile, nfw_rho0
from galacticsics.diagnostics.rotation_curve import (
    build_rotation_curve_report,
    frequency_rotation_curve,
    particle_rotation_curve,
    potential_rotation_curve,
    write_rotation_curve_diagnostic,
)
from galacticsics.diagnostics.virial import (
    potential_gradient_cartesian,
    virial_diagnostic_potential,
)

__all__ = [
    "DEFAULT_THRESHOLDS",
    "build_rotation_curve_report",
    "frequency_rotation_curve",
    "nfw_density_profile",
    "nfw_rho0",
    "particle_rotation_curve",
    "potential_gradient_cartesian",
    "potential_rotation_curve",
    "validate_bulge_df",
    "validate_disk_df",
    "validate_halo_df",
    "validate_ic_distribution_functions",
    "virial_diagnostic_potential",
    "write_rotation_curve_diagnostic",
]
