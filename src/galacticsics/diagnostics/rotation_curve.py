"""Rotation curve diagnostics before/after N-body evolution."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from galacticsics.io import read_harmonic_potential
from galacticsics.io.formats import read_frequency_table
from galacticsics.models import GalaxyModel
from galacticsics.potential.evaluate import evaluate_potential


def potential_rotation_curve(
    pot,
    r_vals: np.ndarray,
    *,
    z: float = 0.0,
    dr: float = 0.01,
) -> np.ndarray:
    """Circular speed from ``V_c^2 = R dPsi/dR`` via finite difference [100 km/s]."""
    vc = np.zeros_like(r_vals, dtype=float)
    for i, r in enumerate(r_vals):
        if r <= 0:
            continue
        psi_r = evaluate_potential(pot, float(r), z)
        psi_dr = evaluate_potential(pot, float(r) + dr, z)
        dpsi_dr = (psi_dr - psi_r) / dr
        vc[i] = math.sqrt(max(0.0, r * abs(dpsi_dr)))
    return vc


def frequency_rotation_curve(freq, r_vals: np.ndarray) -> np.ndarray:
    """``V_circ`` from tabulated epicycle table [100 km/s]."""
    return np.array([float(r) * freq.omega(float(r)) for r in r_vals])


def particle_rotation_curve(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    r_vals: np.ndarray,
    *,
    min_count: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Binned mean azimuthal speed ``<v_phi>(R)`` [100 km/s].

    Returns ``(v_phi_mean, counts)`` aligned with ``r_vals`` bin centers.
    """
    x, y = pos[:, 0], pos[:, 1]
    vx, vy = vel[:, 0], vel[:, 1]
    r = np.sqrt(x * x + y * y)
    v_phi = np.zeros_like(r)
    ok = r > 1e-8
    # Prograde disk: ê_φ = (-sin φ, cos φ) ⇒ v_φ = (-y vx + x vy) / R
    # (the opposite sign was writing a retrograde rotation curve for healthy ICs).
    v_phi[ok] = (-y[ok] * vx[ok] + x[ok] * vy[ok]) / r[ok]

    edges = np.empty(len(r_vals) + 1)
    if len(r_vals) > 1:
        ratio = r_vals[1] / r_vals[0]
        half = np.sqrt(ratio)
        edges[1:-1] = np.sqrt(r_vals[:-1] * r_vals[1:])
        edges[0] = r_vals[0] / half
        edges[-1] = r_vals[-1] * half
    else:
        edges[0] = 0.0
        edges[1] = r_vals[0] * 1.1

    v_mean = np.full(len(r_vals), np.nan, dtype=float)
    counts = np.zeros(len(r_vals), dtype=int)
    for i in range(len(r_vals)):
        mask = (r >= edges[i]) & (r < edges[i + 1])
        counts[i] = int(mask.sum())
        if counts[i] >= min_count:
            v_mean[i] = float(np.mean(v_phi[mask]))
    return v_mean, counts


def build_rotation_curve_report(
    work_dir: Path | str,
    *,
    state=None,
    model: GalaxyModel | None = None,
    r_max: float | None = None,
    n_r: int = 40,
    label: str = "ic",
) -> dict[str, Any]:
    """
    Assemble rotation curve data for JSON export.

    Includes harmonic potential, frequency table (if present), and optional
    particle-based curves split by component when ``state.type_id`` is set.
    """
    work_dir = Path(work_dir)
    pot = read_harmonic_potential(work_dir / "dbh.dat")
    if model is None and (work_dir / "model.json").is_file():
        from galacticsics.campaign.serialize import model_from_dict

        model = model_from_dict(json.loads((work_dir / "model.json").read_text()))

    if r_max is None:
        if model is not None and model.disk and model.disk.enabled:
            r_max = model.disk.outer_radius * 1.2
        else:
            r_max = 25.0
    r_vals = np.logspace(np.log10(max(0.3, r_max / (n_r * 4))), np.log10(r_max), n_r)

    report: dict[str, Any] = {
        "label": label,
        "r_kpc": r_vals.tolist(),
        "v_circ_potential": potential_rotation_curve(pot, r_vals).tolist(),
    }

    freq_path = work_dir / "freqdbh.dat"
    if freq_path.is_file():
        freq = read_frequency_table(freq_path)
        report["v_circ_frequency"] = frequency_rotation_curve(freq, r_vals).tolist()

    if state is not None:
        v_all, c_all = particle_rotation_curve(state.pos, state.vel, state.mass, r_vals)
        report["v_circ_particles"] = np.where(np.isnan(v_all), None, v_all).tolist()
        report["particle_counts"] = c_all.tolist()
        if state.type_id is not None:
            from ntropy.particle_types import TypeRegistry

            registry = TypeRegistry.default_galaxy()
            for name in ("disk", "halo", "bulge"):
                if name not in registry.types:
                    continue
                tid = registry.id_for(name)
                sel = state.type_id == tid
                if not np.any(sel):
                    continue
                v_c, c_c = particle_rotation_curve(
                    state.pos[sel], state.vel[sel], state.mass[sel], r_vals
                )
                report[f"v_circ_{name}"] = np.where(np.isnan(v_c), None, v_c).tolist()
                report[f"particle_counts_{name}"] = c_c.tolist()

    return report


def write_rotation_curve_diagnostic(
    work_dir: Path | str,
    *,
    state=None,
    model: GalaxyModel | None = None,
    label: str = "ic",
) -> Path:
    """Write ``rotation_curve_{label}.json`` under ``work_dir``."""
    work_dir = Path(work_dir)
    report = build_rotation_curve_report(work_dir, state=state, model=model, label=label)
    path = work_dir / f"rotation_curve_{label}.json"
    path.write_text(json.dumps(report, indent=2))
    return path
