"""Epicycle frequency tables from getfreqs (freqdbh.dat)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from galacticsics.numerics import natural_cubic_spline


@dataclass
class FrequencyTable:
    """Epicycle and rotation frequencies tabulated vs radius."""

    radius: np.ndarray
    omega_h: np.ndarray
    nu_h: np.ndarray
    sigma_d: np.ndarray
    v_circ_total: np.ndarray
    v_circ_bulge: np.ndarray
    nu_b: np.ndarray
    psi_midplane: np.ndarray
    d2psi_dr2: np.ndarray
    _omega_spline: CubicSpline | None = None
    _kappa_spline: CubicSpline | None = None

    def __post_init__(self) -> None:
        self._omega_spline = natural_cubic_spline(self.radius, self.omega_h)
        # kappa^2 = d^2 Psi/dR^2 + 3 Omega^2 (omekap.f)
        kappa_sq = self.d2psi_dr2 + 3.0 * self.omega_h**2
        kappa = np.sqrt(np.maximum(kappa_sq, 0.0))
        self._kappa_spline = natural_cubic_spline(self.radius, kappa)

    def omega(self, r: float) -> float:
        return float(self._omega_spline(r))

    def kappa(self, r: float) -> float:
        return float(self._kappa_spline(r))

    def toomre_q(self, r: float, sigma_r: float, sigma_surface: float) -> float:
        """Toomre Q = sigma_R / sigma_crit with sigma_crit = 3.36 sigma_surface / kappa."""
        kap = self.kappa(r)
        if kap <= 0:
            return float("inf")
        sigma_crit = 3.36 * sigma_surface / kap
        return sigma_r / sigma_crit
