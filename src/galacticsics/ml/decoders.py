"""Conditional decoders for non-axisymmetric IC generation (stubs)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class NonAxisymmetricPerturbation:
    """
    Low-dimensional description of departure from axisymmetry.

    Intended decoder output before particle sampling (see
    ``docs/ml_encoder_strategy.md`` Phase 4).
    """

    m1_amplitude: float = 0.0
    m2_amplitude: float = 0.0
    bar_angle_rad: float = 0.0
    triaxiality: float = 1.0
    pattern_speed: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "m1_amplitude": self.m1_amplitude,
            "m2_amplitude": self.m2_amplitude,
            "bar_angle_rad": self.bar_angle_rad,
            "triaxiality": self.triaxiality,
            "pattern_speed": self.pattern_speed,
        }


@dataclass
class HarmonicCoefficientTarget:
    """
    DBH / multipole coefficient targets for hybrid IC generation.

    The decoder predicts these; existing GalactICS builders materialize the
    bulk axisymmetric IC, then apply :class:`NonAxisymmetricPerturbation`.
    """

    lmax: int = 6
    coeffs: dict[str, float] = field(default_factory=dict)
    potential_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecoderConfig:
    """Settings for conditional IC decoders."""

    latent_dim: int = 128
    output_mode: str = "coefficients"  # "coefficients" | "field" | "particles"
    validate_with_short_evolve: bool = True
    evolve_time_gyr: float = 0.05


class ConditionalICDecoder:
    """
    Untrained stub: latent + conditions → harmonic coeffs + perturbation.

    Does **not** generate raw 200k-particle coordinates.  Use GalactICS
    sampling after coefficient prediction.
    """

    def __init__(self, config: DecoderConfig | None = None) -> None:
        self.config = config or DecoderConfig()
        self._trained = False

    @property
    def trained(self) -> bool:
        return self._trained

    def decode(
        self,
        latent: np.ndarray,
        *,
        conditions: dict[str, float] | None = None,
    ) -> tuple[HarmonicCoefficientTarget, NonAxisymmetricPerturbation]:
        """
        Map latent vector to coefficient and perturbation stubs.

        Parameters
        ----------
        latent : ndarray, shape (latent_dim,)
        conditions : dict, optional
            Campaign parameters (``halo.v0``, ``disk.mass``, etc.).

        Returns
        -------
        coeffs : HarmonicCoefficientTarget
        perturbation : NonAxisymmetricPerturbation
        """
        _ = conditions
        # Deterministic stub: first latent components scaled to small perturbations
        z = np.asarray(latent, dtype=np.float64).ravel()
        pad = np.zeros(self.config.latent_dim)
        n = min(len(z), len(pad))
        pad[:n] = z[:n]

        perturbation = NonAxisymmetricPerturbation(
            m1_amplitude=float(0.01 * pad[0]) if n > 0 else 0.0,
            m2_amplitude=float(0.01 * pad[1]) if n > 1 else 0.0,
            bar_angle_rad=float(pad[2]) if n > 2 else 0.0,
            triaxiality=float(1.0 + 0.1 * pad[3]) if n > 3 else 1.0,
        )
        coeffs = HarmonicCoefficientTarget(
            lmax=6,
            coeffs={"stub_l2_m0": float(pad[4]) if n > 4 else 0.0},
            potential_metadata={"decoder_trained": self._trained},
        )
        return coeffs, perturbation

    def validate_stub(self) -> dict[str, Any]:
        """Return planned validation pipeline (not executed in stub)."""
        return {
            "short_evolve": self.config.validate_with_short_evolve,
            "evolve_time_gyr": self.config.evolve_time_gyr,
            "quality_checks": ["dE_over_E0", "rho_drift", "bin_occupancy"],
            "note": "Wire to ntropy Simulation + quality head after training",
        }
