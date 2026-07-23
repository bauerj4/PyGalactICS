"""Epicycle frequency tables from getfreqs (freqdbh.dat)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from galacticsics.numerics import natural_cubic_spline

PathLike = str | Path


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
        self._omega_halo_spline = natural_cubic_spline(self.radius, self.omega_h)
        omega_total = np.zeros_like(self.radius, dtype=float)
        if self.radius.size > 1:
            omega_total[1:] = self.v_circ_total[1:] / self.radius[1:]
            omega_total[0] = omega_total[1]
        self._omega_spline = natural_cubic_spline(self.radius, omega_total)
        # Epicycle frequency from the **total** midplane potential (diskdf / gendisk).
        kappa_sq = self.d2psi_dr2 + 3.0 * omega_total**2
        kappa = np.sqrt(np.maximum(kappa_sq, 0.0))
        self._kappa_spline = natural_cubic_spline(self.radius, kappa)

    def omega(self, r: float) -> float:
        """Circular frequency ``v_circ / R`` from the total midplane potential."""
        return float(self._omega_spline(r))

    def omega_halo(self, r: float) -> float:
        """Halo-only circular frequency (``getfreqs`` column ``OMEGA_H``)."""
        return float(self._omega_halo_spline(r))

    def kappa(self, r: float) -> float:
        return float(self._kappa_spline(r))

    @classmethod
    def from_omekap_file(cls, path: PathLike) -> FrequencyTable:
        """
        Load ``freqdbh.dat`` the way legacy ``omekap.f`` does (subsampled rows).

        ``diskdf`` / ``gendisk`` call ``omekap`` rather than using the raw table
        directly; matching that layout is required for convergent ``cordbh.dat``.
        """
        rows: list[list[float]] = []
        for line in Path(path).read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 9:
                rows.append([float(x) for x in parts[:9]])
        data = np.asarray(rows, dtype=float)
        if data.shape[0] < 4:
            raise ValueError(f"freqdbh too short for omekap layout: {path}")
        sub = data[1::2]
        rr = sub[:, 0]
        vc = sub[:, 4]
        psi = sub[:, 7]
        psirr = sub[:, 8]
        n = sub.shape[0] + 1
        radius = np.zeros(n, dtype=float)
        omega_h = np.zeros(n, dtype=float)
        v_circ = np.zeros(n, dtype=float)
        psi_mid = np.zeros(n, dtype=float)
        d2psi = np.zeros(n, dtype=float)
        radius[1:] = rr
        omega_h[1:] = vc / np.maximum(rr, 1e-30)
        v_circ[1:] = vc
        psi_mid[1:] = psi
        d2psi[1:] = psirr
        if n >= 3:
            omega_h[0] = 2.0 * omega_h[1] - omega_h[2]
            psi_mid[0] = (4.0 * psi_mid[1] - psi_mid[2]) / 3.0
            d2psi[0] = (4.0 * d2psi[1] - d2psi[2]) / 3.0
        zeros = np.zeros(n, dtype=float)
        return cls(
            radius=radius,
            omega_h=omega_h,
            nu_h=zeros,
            sigma_d=zeros,
            v_circ_total=v_circ,
            v_circ_bulge=zeros,
            nu_b=zeros,
            psi_midplane=psi_mid,
            d2psi_dr2=d2psi,
        )

    def toomre_q(self, r: float, sigma_r: float, sigma_surface: float) -> float:
        """Toomre Q = sigma_R / sigma_crit with sigma_crit = 3.36 sigma_surface / kappa."""
        kap = self.kappa(r)
        if kap <= 0:
            return float("inf")
        sigma_crit = 3.36 * sigma_surface / kap
        return sigma_r / sigma_crit
