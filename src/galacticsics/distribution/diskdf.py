"""Disk distribution function correction tables (cordbh.dat)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from galacticsics.numerics import bounded_linear_interpolator


@dataclass
class DiskCorrectionTable:
    """Radial correction splines f_d(R) and f_sz(R) from diskdf."""

    sigma_r0: float
    sigma_r_scale: float
    radius: np.ndarray
    f_d: np.ndarray
    f_sz: np.ndarray
    _spline_f_d: object | None = None
    _spline_f_sz: object | None = None

    def __post_init__(self) -> None:
        # Corrections are clipped positive at their knots.  A natural cubic
        # can overshoot below zero between them, driving the DF normalization
        # or vertical correction toward zero and creating unphysical spikes.
        # Native sampling is linear.
        self._spline_f_d = bounded_linear_interpolator(self.radius, self.f_d)
        self._spline_f_sz = bounded_linear_interpolator(self.radius, self.f_sz)

    def f_d_at(self, r: float) -> float:
        return float(self._spline_f_d(r))

    def f_sz_at(self, r: float) -> float:
        return float(self._spline_f_sz(r))

    def sigma_r_squared(self, r: float, rdisk: float) -> float:
        """sigma_R^2(R) = sigma_r0^2 exp(-R / sigma_r_scale) per sigr2.f.

        The ``f_d`` correction changes the DF normalization through
        ``fnamidden``; it does not modify the prescribed radial dispersion.
        """
        del rdisk
        return self.sigma_r0**2 * math.exp(-r / self.sigma_r_scale)
