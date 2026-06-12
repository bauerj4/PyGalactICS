"""Rotation curve fitting (legacy fitvc.f logic in Python)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

from galacticsics.potential.evaluate import evaluate_force
from galacticsics.potential.harmonics import HarmonicPotential
from galacticsics.units import MASS_UNIT_MSUN, velocity_to_kms

PathLike = Union[str, Path]


@dataclass
class RotationCurveFit:
    """Result of scaling model circular velocity to observations."""

    rms: float
    v_scale: float
    disk_mass_msun: float
    bulge_mass_msun: float
    halo_mass_msun: float
    halo_radius_kpc: float


def read_rotation_curve(path: PathLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read new_rotation.dat style file: R[kpc], V[km/s], sigma."""
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    data = np.asarray(rows, dtype=float)
    return data[:, 0], data[:, 1], data[:, 2]


def fit_rotation_curve(
    potential: HarmonicPotential,
    radii_kpc: np.ndarray,
    v_obs_kms: np.ndarray,
    sigma_kms: np.ndarray,
    component_masses: dict[str, tuple[float, float]],
    *,
    r_unit_kpc: float = 4.465,
) -> RotationCurveFit:
    """Fit velocity scale factor to match observed rotation curve.

    Parameters
    ----------
    potential
        Solved harmonic potential.
    radii_kpc
        Observed radii in kpc (converted to GalactICS units internally).
    v_obs_kms, sigma_kms
        Observed velocities and uncertainties in km/s.
    component_masses
        Disk, bulge, halo (mass, radius) from mr.dat in GalactICS units.
    r_unit_kpc
        Legacy scaling used in fitvc.f (rf converted by /4.465).
    """
    r_gal = radii_kpc / r_unit_kpc
    v_model = np.zeros_like(r_gal)
    for i, r in enumerate(r_gal):
        fr, _, _ = evaluate_force(potential, r, 0.0)
        v_model[i] = np.sqrt(max(0.0, -fr * r))

    weights = v_model
    v_scale = float(np.sum(weights * v_obs_kms / 100.0) / np.sum(weights * v_model))
    residuals = (v_model * v_scale - v_obs_kms / 100.0) / (sigma_kms / 100.0)
    rms = float(np.sqrt(np.mean(residuals**2)))

    dm, _ = component_masses.get("disk", (0.0, 0.0))
    bm, _ = component_masses.get("bulge", (0.0, 0.0))
    hm, hr = component_masses.get("halo", (0.0, 0.0))
    scale_msun = v_scale**2 / 4.3e-6  # legacy fitvc conversion

    return RotationCurveFit(
        rms=rms,
        v_scale=v_scale,
        disk_mass_msun=dm * scale_msun,
        bulge_mass_msun=bm * scale_msun,
        halo_mass_msun=hm * scale_msun,
        halo_radius_kpc=hr,
    )
