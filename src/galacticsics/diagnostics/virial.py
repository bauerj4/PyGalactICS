"""Virial-theorem diagnostics against the continuous GalactICS potential."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from galacticsics.io import read_harmonic_potential
from galacticsics.potential.evaluate import evaluate_potential
from galacticsics.potential.harmonics import HarmonicPotential


def potential_gradient_cartesian(
    pot: HarmonicPotential,
    pos: np.ndarray,
    *,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Cartesian ``∇Ψ`` by finite differences of :func:`evaluate_potential`.

    GalactICS stores the relative potential ``Ψ = −Φ`` (up to a constant), so
    the physical acceleration is ``a = ∇Ψ``. Do **not** use
    :func:`~galacticsics.potential.evaluate.evaluate_force` here: its
    tabulated ``fr`` harmonics are not identical to ``∂Ψ/∂R``.
    """
    pos = np.asarray(pos, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("pos must have shape (N, 3)")
    acc = np.zeros_like(pos, dtype=float)
    for i, (x, y, z) in enumerate(pos):
        r_cyl = float(np.hypot(x, y))
        # ∂Ψ/∂R, ∂Ψ/∂z in cylindrical, then project to Cartesian.
        psi_rp = evaluate_potential(pot, r_cyl + eps, z)
        psi_rm = evaluate_potential(pot, max(r_cyl - eps, 0.0), z)
        psi_zp = evaluate_potential(pot, r_cyl, z + eps)
        psi_zm = evaluate_potential(pot, r_cyl, z - eps)
        denom_r = 2.0 * eps if r_cyl > eps else eps
        dpsi_dr = (psi_rp - psi_rm) / denom_r
        dpsi_dz = (psi_zp - psi_zm) / (2.0 * eps)
        if r_cyl > 1e-12:
            acc[i, 0] = dpsi_dr * x / r_cyl
            acc[i, 1] = dpsi_dr * y / r_cyl
        acc[i, 2] = dpsi_dz
    return acc


def virial_diagnostic_potential(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    pot: HarmonicPotential | Path,
    *,
    rtol: float = 0.25,
    eps: float = 1e-4,
) -> dict[str, Any]:
    """
    Softening-free virial check against the fixed ``dbh`` potential.

    For tracers (or the full multi-component IC) in equilibrium in ``Ψ``,

    .. math::

       2K + W \\approx 0, \\qquad
       W = \\sum_i m_i\\,\\mathbf{x}_i\\cdot\\nabla\\Psi(\\mathbf{x}_i), \\qquad
       K = \\tfrac12\\sum_i m_i v_i^2.

    Pairwise N-body ``virial_diagnostic`` is **not** the right IC test: GalactICS
    particles are drawn for the continuous potential, not softened self-gravity.
    """
    if not isinstance(pot, HarmonicPotential):
        pot = read_harmonic_potential(Path(pot))
    pos = np.asarray(pos, dtype=float)
    vel = np.asarray(vel, dtype=float)
    mass = np.asarray(mass, dtype=float).reshape(-1)
    acc = potential_gradient_cartesian(pot, pos, eps=eps)
    kinetic = 0.5 * float(np.sum(mass * np.sum(vel * vel, axis=1)))
    virial_w = float(np.sum(mass * np.sum(pos * acc, axis=1)))
    abs_w = abs(virial_w)
    virial_sum = 2.0 * kinetic + virial_w
    virial_ratio = (2.0 * kinetic / abs_w) if abs_w > 0.0 else 0.0
    residual_rel = abs(virial_sum) / max(abs_w, 1e-30)
    return {
        "kinetic_energy": kinetic,
        "potential_virial": virial_w,
        "virial_sum": virial_sum,
        "virial_ratio": virial_ratio,
        "virial_residual_rel": residual_rel,
        "is_virial_equilibrium": bool(residual_rel <= rtol),
        "rtol": float(rtol),
    }
