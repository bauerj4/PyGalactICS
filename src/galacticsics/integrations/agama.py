"""agama dynamical model adapter (optional dependency)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from galacticsics.potential.harmonics import HarmonicPotential


def agama_available() -> bool:
    try:
        import agama  # noqa: F401
        return True
    except ImportError:
        return False


def harmonic_to_agama(harmonic: HarmonicPotential, *, n_r: int = 50, n_z: int = 50):
    """
    Build an agama potential from a harmonic grid via cylindrical interpolation.

  Requires ``pip install agama``.
    """
    if not agama_available():
        raise ImportError("agama is required; install with pip install agama")

    import agama
    from galacticsics.potential.evaluate import evaluate_potential

    r_vals = np.linspace(0.1, harmonic.r_edge, n_r)
    z_vals = np.linspace(-harmonic.r_edge * 0.5, harmonic.r_edge * 0.5, n_z)
    psi = np.zeros((len(z_vals), len(r_vals)))
    for iz, z in enumerate(z_vals):
        for ir, r in enumerate(r_vals):
            psi[iz, ir] = evaluate_potential(harmonic, r, z)

    return agama.Potential(
        type="CylSpline",
        R=r_vals,
        z=z_vals,
        V=psi,
    )
