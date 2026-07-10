"""Toomre Q diagnostics and target scaling (legacy ``diskdf.f`` parity)."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from galacticsics.io import read_harmonic_potential
from galacticsics.io.formats import read_frequency_table
from galacticsics.models import GalaxyModel

TOOMRE_RADIUS_FACTOR = 2.5


def toomre_reference_radius(model: GalaxyModel) -> float:
    """Radius used for Q reporting: ``2.5 R_d`` (``diskdf.f``)."""
    disk = model.disk
    if disk is None or not disk.enabled:
        raise ValueError("disk component required for Toomre Q")
    return TOOMRE_RADIUS_FACTOR * disk.scale_length


def sigma_r_at_r(model: GalaxyModel, r: float, *, f_d: float = 1.0) -> float:
    kin = model.disk_kinematics
    return math.sqrt(
        kin.sigma_r0**2 * math.exp(-r / kin.sigma_r_scale) * max(f_d, 1e-6)
    )


def sigma_surface_at_r(pot, r: float) -> float:
    """Surface density estimate ``2 z_d rho(R, 0)`` matching legacy ``diskdf.f``."""
    from galacticsics.distribution.diskdf_solve import _disk_midplane_density

    disk = pot.model.disk
    assert disk is not None
    rho_mid = _disk_midplane_density(r, 0.0, pot)
    return rho_mid * 2.0 * disk.scale_height


def compute_toomre_q(
    model: GalaxyModel,
    work_dir: Path | str,
    *,
    r: float | None = None,
) -> tuple[float, float]:
    """
    Compute Toomre Q at reference radius.

    Returns
    -------
    q, r_kpc
        Toomre Q and the radius at which it was evaluated [kpc].
    """
    work_dir = Path(work_dir)
    pot = read_harmonic_potential(work_dir / "dbh.dat")
    freq = read_frequency_table(work_dir / "freqdbh.dat")
    r_eval = toomre_reference_radius(model) if r is None else r
    sigma_r = sigma_r_at_r(model, r_eval)
    sigma_surface = sigma_surface_at_r(pot, r_eval)
    q = freq.toomre_q(r_eval, sigma_r, sigma_surface)
    return q, r_eval


def write_toomre_q(work_dir: Path | str, q: float) -> Path:
    """Write ``toomre2.5`` (single-line float, legacy format)."""
    path = Path(work_dir) / "toomre2.5"
    path.write_text(f"{q}\n")
    return path


def apply_toomre_q_target(model: GalaxyModel, work_dir: Path | str) -> GalaxyModel:
    """
    Scale ``sigma_r0`` so Q matches ``disk_kinematics.toomre_q_target``.

    Leaves ``toomre_q_target`` unchanged on the model (grid input); only
    ``sigma_r0`` is updated for sampling / ``diskdf``.
    """
    target = model.disk_kinematics.toomre_q_target
    if target is None:
        return model
    if target <= 0:
        raise ValueError(f"toomre_q_target must be positive, got {target}")
    q_meas, _ = compute_toomre_q(model, work_dir)
    if not math.isfinite(q_meas) or q_meas <= 0:
        raise ValueError(f"cannot scale Toomre Q from measured Q={q_meas}")
    kin = model.disk_kinematics
    new_sigma_r0 = kin.sigma_r0 * (target / q_meas)
    return replace(model, disk_kinematics=replace(kin, sigma_r0=new_sigma_r0))


def log_toomre_q(
    model: GalaxyModel,
    work_dir: Path | str,
    *,
    log: Callable[[str], None] | None = None,
) -> dict[str, float]:
    """Compute Q, write ``toomre2.5``, optionally log a one-line summary."""
    q, r = compute_toomre_q(model, work_dir)
    write_toomre_q(work_dir, q)
    kin = model.disk_kinematics
    msg = (
        f"Toomre Q = {q:.4f} at R = {r:.3f} kpc "
        f"(sigma_r0 = {kin.sigma_r0:.4f}, sigma_r_scale = {kin.sigma_r_scale:.3f} kpc)"
    )
    if model.disk_kinematics.toomre_q_target is not None:
        msg += f"; target Q = {model.disk_kinematics.toomre_q_target:.4f}"
    if log is not None:
        log(msg)
    return {
        "toomre_q": q,
        "toomre_r_kpc": r,
        "sigma_r0": kin.sigma_r0,
        "sigma_r_scale": kin.sigma_r_scale,
        "toomre_q_target": model.disk_kinematics.toomre_q_target,
    }
