"""Gravitational potential representation and evaluation."""

from galacticsics.potential.evaluate import evaluate_force, evaluate_potential
from galacticsics.potential.harmonics import ComponentFlags, HarmonicPotential
from galacticsics.potential.halo_first import (
    BaryonsInFixedHaloResult,
    HaloFirstStepResult,
    run_halo_first_workflow,
    solve_baryons_in_fixed_halo,
    solve_halo_potential,
)
from galacticsics.potential.solver import SolveDiagnostics, SolveResult, solve_potential

__all__ = [
    "HarmonicPotential",
    "ComponentFlags",
    "evaluate_potential",
    "evaluate_force",
    "solve_potential",
    "SolveResult",
    "SolveDiagnostics",
    "solve_halo_potential",
    "solve_baryons_in_fixed_halo",
    "run_halo_first_workflow",
    "HaloFirstStepResult",
    "BaryonsInFixedHaloResult",
]
