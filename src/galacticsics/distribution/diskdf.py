"""Disk distribution function correction tables (cordbh.dat)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from galacticsics.numerics import natural_cubic_spline


@dataclass
class DiskCorrectionTable:
    """Radial correction splines f_d(R) and f_sz(R) from diskdf."""

    sigma_r0: float
    sigma_r_scale: float
    radius: np.ndarray
    f_d: np.ndarray
    f_sz: np.ndarray
    _spline_f_d: CubicSpline | None = None
    _spline_f_sz: CubicSpline | None = None

    def __post_init__(self) -> None:
        self._spline_f_d = natural_cubic_spline(self.radius, self.f_d)
        self._spline_f_sz = natural_cubic_spline(self.radius, self.f_sz)

    def f_d_at(self, r: float) -> float:
        return float(self._spline_f_d(r))

    def f_sz_at(self, r: float) -> float:
        return float(self._spline_f_sz(r))

    def sigma_r_squared(self, r: float, rdisk: float) -> float:
        """sigma_R^2(R) = sigma_r0^2 exp(-R / sigma_r_scale) per sigr2.f."""
        return self.sigma_r0**2 * math.exp(-r / self.sigma_r_scale) * self.f_d_at(r)
