"""ntropy — minimal N-body simulation for testing initial conditions."""

from ntropy.config import RunConfig, load_config
from ntropy.integrations.galacticsics import (
    galacticsics_available,
    sample_galacticsics_galaxy,
    sample_galacticsics_halo,
)
from ntropy.particles import ParticleState
from ntropy.simulation import Simulation, run_simulation

__version__ = "0.1.0"

__all__ = [
    "RunConfig",
    "load_config",
    "ParticleState",
    "Simulation",
    "run_simulation",
    "galacticsics_available",
    "sample_galacticsics_halo",
    "sample_galacticsics_galaxy",
    "__version__",
]
