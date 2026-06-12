"""GalactICS: galaxy potential fitting and N-body initial conditions."""

from galacticsics.models import (
    DiskKinematics,
    ExponentialDisk,
    GalaxyModel,
    GasDisk,
    NFWHalo,
    PotentialGrid,
    SersicBulge,
)
from galacticsics.units import DEFAULT_UNITS, UnitSystem
from galacticsics.potential import (
    HarmonicPotential,
    SolveDiagnostics,
    SolveResult,
    evaluate_force,
    evaluate_potential,
    solve_potential,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "GalaxyModel",
    "NFWHalo",
    "ExponentialDisk",
    "GasDisk",
    "SersicBulge",
    "PotentialGrid",
    "DiskKinematics",
    "UnitSystem",
    "DEFAULT_UNITS",
    "HarmonicPotential",
    "evaluate_potential",
    "evaluate_force",
    "solve_potential",
    "SolveResult",
    "SolveDiagnostics",
]
