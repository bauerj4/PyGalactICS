"""Shared phase-space frame for multi-component field maps.

**Critical rule — one center for the whole snapshot**

Disk, bulge, and halo must **not** be recentered on their own COMs before
binning.  Per-component centering misaligns components relative to each other
(e.g. a bar offset from the bulge, or a lopsided halo).  Always:

1. Compute the **mass-weighted global COM** of *all* particles once.
2. Subtract that COM (and VCOM) from the full state.
3. Optionally apply **one** common in-plane rotation to the full state.
4. Bin each component (and Φ) on that same origin / FOV frame.

Binning functions themselves never recenter — callers must pass already-
aligned coordinates via :func:`prepare_shared_frame`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from galacticsics.ml.morton.tokenize import (
    COMPONENT_IDS,
    center_phase_space,
    rotate_about_z,
)


def mass_weighted_com(
    pos: np.ndarray,
    mass: np.ndarray | None = None,
) -> np.ndarray:
    """Mass-weighted center of mass (uniform mean if ``mass`` is omitted)."""
    pos = np.asarray(pos, dtype=np.float64)
    if mass is None:
        return pos.mean(axis=0)
    m = np.asarray(mass, dtype=np.float64).reshape(-1)
    w = m / max(float(m.sum()), 1e-30)
    return (pos * w[:, None]).sum(axis=0)


def component_com(
    pos: np.ndarray,
    mass: np.ndarray,
    component_id: np.ndarray,
    name_or_id: str | int,
) -> np.ndarray:
    """Mass-weighted COM of one component in the current frame (no recentering)."""
    cid = int(COMPONENT_IDS[name_or_id]) if isinstance(name_or_id, str) else int(name_or_id)
    mask = np.asarray(component_id, dtype=np.int64).reshape(-1) == cid
    if not np.any(mask):
        return np.full(3, np.nan)
    return mass_weighted_com(pos[mask], mass[mask])


def prepare_shared_frame(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray | None = None,
    *,
    center: bool = True,
    rotate: bool = False,
    phi: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Align the **full** snapshot to one shared origin (and optional rotation).

    Parameters
    ----------
    pos, vel :
        Full-snapshot phase space ``(N, 3)``.
    mass :
        Masses for COM weighting; uniform if omitted.
    center :
        If True, subtract the global mass-weighted COM / VCOM once.
    rotate :
        If True, apply one in-plane rotation to **all** particles after centering.
    phi :
        Fixed rotation angle (radians).  When ``rotate`` and ``phi is None``,
        draw ``φ ∼ U[0, 2π)`` from ``rng``.
    rng :
        RNG for random rotation (required if ``rotate`` and ``phi is None``).

    Returns
    -------
    pos_c, vel_c, meta
        Transformed arrays (copies) and metadata including ``com``, ``vcom``,
        ``phi``, and ``shared_center=True``.
    """
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    meta: dict[str, Any] = {
        "shared_center": True,
        "centered": bool(center),
        "rotated": False,
        "phi": 0.0,
        "com": np.zeros(3, dtype=np.float64),
        "vcom": np.zeros(3, dtype=np.float64),
    }

    if center:
        if mass is None:
            com = pos.mean(axis=0)
            vcom = vel.mean(axis=0)
            pos_c = pos - com
            vel_c = vel - vcom
        else:
            m = np.asarray(mass, dtype=np.float64).reshape(-1)
            w = m / max(float(m.sum()), 1e-30)
            com = (pos * w[:, None]).sum(axis=0)
            vcom = (vel * w[:, None]).sum(axis=0)
            # Same transform as Morton track (global COM once).
            pos_c, vel_c = center_phase_space(pos, vel, mass)
        meta["com"] = np.asarray(com, dtype=np.float64)
        meta["vcom"] = np.asarray(vcom, dtype=np.float64)
    else:
        pos_c = pos.copy()
        vel_c = vel.copy()

    if rotate:
        if phi is None:
            if rng is None:
                raise ValueError("prepare_shared_frame(rotate=True) needs rng or phi")
            phi = float(rng.uniform(0.0, 2.0 * np.pi))
        pos_c, vel_c = rotate_about_z(pos_c, vel_c, float(phi))
        meta["phi"] = float(phi)
        meta["rotated"] = True

    return pos_c, vel_c, meta
