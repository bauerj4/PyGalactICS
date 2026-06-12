"""Peano-Hilbert / Morton domain decomposition (Gadget-2 style)."""

from __future__ import annotations

import numpy as np


def _expand_bits(v: np.ndarray) -> np.ndarray:
    """Spread 10-bit integers into 30 bits for 3D Morton interleaving."""
    v = v.astype(np.uint64)
    v = (v | (v << 16)) & 0x0000FFFF0000FFFF
    v = (v | (v << 8)) & 0x00FF00FF00FF00FF
    v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0F
    v = (v | (v << 2)) & 0x3333333333333333
    v = (v | (v << 1)) & 0x5555555555555555
    return v


def peano_keys(
    pos: np.ndarray,
    box_min: np.ndarray,
    box_size: float,
    bits: int = 10,
) -> np.ndarray:
    """
    Compute 3D Morton (Z-order) spatial hash keys.

    Gadget-2 uses Peano–Hilbert curves; Morton keys provide equivalent
    spatial locality with a simpler bit-interleaving scheme.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Particle positions.
    box_min : ndarray, shape (3,)
        Minimum corner of the bounding box.
    box_size : float
        Side length of the bounding cube.
    bits : int
        Bits per dimension for key quantization.

    Returns
    -------
    keys : ndarray, shape (N,), dtype uint64
        Morton keys for sorting.
    """
    if box_size <= 0:
        box_size = 1.0
    scaled = (pos - box_min) / box_size
    scaled = np.clip(scaled, 0.0, 1.0 - 1e-12)
    max_val = (1 << bits) - 1
    ix = (scaled[:, 0] * max_val).astype(np.uint64)
    iy = (scaled[:, 1] * max_val).astype(np.uint64)
    iz = (scaled[:, 2] * max_val).astype(np.uint64)
    return (_expand_bits(ix) | (_expand_bits(iy) << 1) | (_expand_bits(iz) << 2))


def sort_by_peano(pos: np.ndarray) -> np.ndarray:
    """
    Return indices that sort particles by Morton spatial key.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Particle positions.

    Returns
    -------
    indices : ndarray, shape (N,)
        Permutation that sorts particles by spatial locality.
    """
    box_min = pos.min(axis=0)
    box_max = pos.max(axis=0)
    box_size = float((box_max - box_min).max())
    if box_size == 0:
        box_size = 1.0
    keys = peano_keys(pos, box_min, box_size)
    return np.argsort(keys, kind="stable")


def domain_slices(n_particles: int, n_domains: int) -> list[slice]:
    """
    Split particle count into contiguous domain slices.

    Parameters
    ----------
    n_particles : int
        Total number of particles.
    n_domains : int
        Number of MPI domains or workers.

    Returns
    -------
    slices : list of slice
        One slice per domain; lengths differ by at most one particle.
    """
    if n_domains < 1:
        raise ValueError("n_domains must be >= 1")
    base, rem = divmod(n_particles, n_domains)
    slices: list[slice] = []
    start = 0
    for d in range(n_domains):
        count = base + (1 if d < rem else 0)
        end = start + count
        slices.append(slice(start, end))
        start = end
    return slices
