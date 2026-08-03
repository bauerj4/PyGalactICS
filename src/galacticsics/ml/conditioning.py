"""Shared structural conditioning ``θ`` for field and particle generative models.

Both the multi-scale field VAE (:mod:`galacticsics.ml.fields.vae`) and the
particle-set Morton VAE (:mod:`galacticsics.ml.models.sequence_vae`) condition
on the same GalactICS structural keys harvested from ``model.json`` plus
``t_gyr``.  Keep key order stable so checkpoints stay compatible.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

DEFAULT_THETA_KEYS: list[str] = [
    "disk.mass",
    "disk.scale_length",
    "disk.scale_height",
    "disk_kinematics.toomre_q_target",
    "disk_kinematics.sigma_r0",
    "halo.v0",
    "halo.a",
    "bulge.v0",
    "bulge.a",
    "t_gyr",
]


def theta_vector(
    theta: Mapping[str, float],
    keys: list[str] | None = None,
) -> np.ndarray:
    """Stack conditioning keys; non-finite values (e.g. missing ``t_gyr``) → 0."""
    key_list = list(keys or DEFAULT_THETA_KEYS)
    arr = np.asarray([float(theta.get(k, 0.0)) for k in key_list], dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def theta_from_record(
    theta: Mapping[str, float],
    *,
    t_gyr: float | None = None,
    keys: list[str] | None = None,
) -> np.ndarray:
    """Build ``θ`` from a snapshot record dict, injecting ``t_gyr`` when absent."""
    merged = dict(theta)
    if t_gyr is not None and np.isfinite(t_gyr):
        # Prefer the resolved dump time — older manifests store nan in θ.
        merged["t_gyr"] = float(t_gyr)
    elif "t_gyr" in merged and not np.isfinite(float(merged["t_gyr"])):
        merged["t_gyr"] = 0.0
    return theta_vector(merged, keys=keys)
