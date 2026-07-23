"""gala dynamical model adapter (optional dependency)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from galacticsics.potential.harmonics import HarmonicPotential


def gala_available() -> bool:
    try:
        import gala  # noqa: F401
        return True
    except ImportError:
        return False


def harmonic_to_gala(harmonic: HarmonicPotential, *, n_r: int = 40, n_z: int = 40):
    """
    Build a gala potential from interpolated Psi(R, z).

    Requires ``pip install gala``.
    """
    if not gala_available():
        raise ImportError("gala is required; install with pip install gala")

    from gala.potential import CustomPotential
    from galacticsics.potential.evaluate import evaluate_potential

    r_vals = np.linspace(0.1, harmonic.r_edge, n_r)
    z_vals = np.linspace(-harmonic.r_edge * 0.5, harmonic.r_edge * 0.5, n_z)

    def phi(R, z, _t=0.0):
        R = np.atleast_1d(R)
        z = np.atleast_1d(z)
        out = np.zeros_like(R, dtype=float)
        for i in range(len(R)):
            out[i] = evaluate_potential(harmonic, float(R[i]), float(z[i]))
        return out

    return CustomPotential(parameters={}, units=None, potential_func=phi)
