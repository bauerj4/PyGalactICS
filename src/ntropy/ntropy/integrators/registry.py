"""Integrator metadata helpers."""

from __future__ import annotations

from typing import Literal

IntegratorType = Literal["leapfrog", "euler", "rk2", "rk3", "rk4"]

SYMPLECTIC_TYPES = frozenset({"leapfrog"})


def integrator_force_evaluations(
    integrator_type: IntegratorType,
    *,
    order: int = 2,
) -> int:
    """
    Return the number of force evaluations performed per timestep.

    Parameters
    ----------
    integrator_type : {'leapfrog', 'euler', 'rk2', 'rk3', 'rk4'}
        Integrator name.
    order : int
        Leapfrog order (1 or 2); ignored for other types.

    Returns
    -------
    n_forces : int
    """
    if integrator_type == "leapfrog":
        return 1 if order == 1 else 2
    if integrator_type == "euler":
        return 1
    if integrator_type == "rk2":
        return 2
    if integrator_type == "rk3":
        return 3
    if integrator_type == "rk4":
        return 4
    raise ValueError(f"Unknown integrator type {integrator_type!r}")


def is_symplectic(integrator_type: IntegratorType) -> bool:
    """Return True when the integrator is symplectic (Hamiltonian-preserving)."""
    return integrator_type in SYMPLECTIC_TYPES
