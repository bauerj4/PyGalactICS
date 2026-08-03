"""3D Morton keys (same bit-interleave as ``ntropy.parallel.domains.peano_keys``).

Kept local so importing the ML package does not pull CUDA via ``ntropy.parallel``.
"""

from __future__ import annotations

import numpy as np


def _expand_bits(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.uint64)
    v = (v | (v << 16)) & 0x0000FFFF0000FFFF
    v = (v | (v << 8)) & 0x00FF00FF00FF00FF
    v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0F
    v = (v | (v << 2)) & 0x3333333333333333
    v = (v | (v << 1)) & 0x5555555555555555
    return v


def morton_keys(
    pos: np.ndarray,
    box_min: np.ndarray,
    box_size: float,
    bits: int = 10,
) -> np.ndarray:
    """
    Compute 3-D Morton (Z-order) keys for particle positions.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Cartesian positions [kpc].
    box_min : ndarray, shape (3,)
        Lower corner of the axis-aligned bounding box.
    box_size : float
        Edge length of the cubic quantisation domain (usually the max extent
        of the AABB).  Non-positive values are replaced by ``1.0``.
    bits : int
        Bits per axis.  Keys use bit-interleave of three ``bits``-wide integers
        (key space size ``(2^bits)^3``).

    Returns
    -------
    keys : ndarray, shape (N,), dtype uint64
        Morton keys suitable for stable sorting / AR ordering features.
    """
    if box_size <= 0:
        box_size = 1.0
    scaled = (pos - box_min) / box_size
    scaled = np.clip(scaled, 0.0, 1.0 - 1e-12)
    max_val = (1 << bits) - 1
    ix = (scaled[:, 0] * max_val).astype(np.uint64)
    iy = (scaled[:, 1] * max_val).astype(np.uint64)
    iz = (scaled[:, 2] * max_val).astype(np.uint64)
    return _expand_bits(ix) | (_expand_bits(iy) << 1) | (_expand_bits(iz) << 2)


def sort_by_morton(pos: np.ndarray, *, bits: int = 10) -> np.ndarray:
    """
    Indices that sort particles by Morton key over their AABB.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Positions [kpc].
    bits : int
        Per-axis Morton resolution.

    Returns
    -------
    order : ndarray, shape (N,), dtype intp
        Stable argsort indices.
    """
    box_min = pos.min(axis=0)
    box_size = float((pos.max(axis=0) - box_min).max())
    if box_size == 0:
        box_size = 1.0
    keys = morton_keys(pos, box_min, box_size, bits=bits)
    return np.argsort(keys, kind="stable")
