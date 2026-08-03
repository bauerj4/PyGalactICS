"""Gravitational potential on slice / voxel grids (subsample Plummer).

Uses a stratified particle subsample and Plummer softening
``Φ(x) = -Σ m_j / sqrt(|x-x_j|² + ε²)``.  This is intentionally cheap for
CPU smoke; production can swap in BH/`bh_c` potential evaluators when available.

Potential maps follow the **same per-component FOV** as density slices so the
bulge Φ is not undersampled on a halo-sized canvas.

**Shared center:** evaluate Φ on the same global-COM frame used for density
binning (:func:`~galacticsics.ml.fields.frame.prepare_shared_frame`).  Do not
recenter sources or targets per component — grids are already aligned to one
origin; only the FOV / ``n_pix`` differ.
"""

from __future__ import annotations

import numpy as np

from galacticsics.ml.fields.binning import (
    ComponentSliceGrid,
    MultiScaleSliceConfig,
    SliceMapConfig,
    z_edges_for_grid,
)
from galacticsics.ml.morton.tokenize import subsample_stratified


def plummer_potential_on_points(
    targets: np.ndarray,
    pos: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray | float,
) -> np.ndarray:
    """
    Softened Plummer potential at ``targets`` from source particles.

    Parameters
    ----------
    targets : ndarray, shape (M, 3)
        Evaluation positions [kpc].
    pos, mass : ndarray
        Source particles.
    eps : float or ndarray, shape (N,)
        Softening length(s).

    Returns
    -------
    phi : ndarray, shape (M,)
        Potential (negative for attractive masses; G=1 code units).
    """
    targets = np.asarray(targets, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64).reshape(-1)
    if np.isscalar(eps):
        eps_arr = np.full(pos.shape[0], float(eps), dtype=np.float64)
    else:
        eps_arr = np.asarray(eps, dtype=np.float64).reshape(-1)

    m = targets.shape[0]
    out = np.zeros(m, dtype=np.float64)
    chunk = max(1, min(m, 4096 * 4096 // max(pos.shape[0], 1)))
    for i0 in range(0, m, chunk):
        i1 = min(m, i0 + chunk)
        dr = targets[i0:i1, None, :] - pos[None, :, :]
        r2 = np.sum(dr * dr, axis=-1)
        h2 = eps_arr[None, :] ** 2
        out[i0:i1] = -np.sum(mass[None, :] / np.sqrt(r2 + h2), axis=1)
    return out


def plummer_potential_on_component_slices(
    pos: np.ndarray,
    mass: np.ndarray,
    component_id: np.ndarray,
    grid: ComponentSliceGrid,
    *,
    eps: float = 0.05,
    n_sub: int = 2048,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Evaluate Plummer Φ on midplanes of one component's ``z`` slabs.

    Returns
    -------
    phi : ndarray, shape (n_z, n_pix, n_pix), float32
    """
    rng = rng or np.random.default_rng(0)
    pos = np.asarray(pos, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64).reshape(-1)
    cid = np.asarray(component_id, dtype=np.int64).reshape(-1)

    n_take = min(int(n_sub), pos.shape[0])
    idx = subsample_stratified(cid, n_take, rng=rng)
    src_pos = pos[idx]
    src_mass = mass[idx]

    n_pix = int(grid.n_pix)
    n_z = int(grid.n_z)
    xy = np.linspace(-float(grid.r_max), float(grid.r_max), n_pix, endpoint=False)
    xy = xy + 0.5 * (2.0 * float(grid.r_max) / n_pix)
    z_edges = z_edges_for_grid(grid)
    z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])

    xx, yy = np.meshgrid(xy, xy, indexing="ij")
    out = np.zeros((n_z, n_pix, n_pix), dtype=np.float64)
    for iz, z0 in enumerate(z_mid):
        targets = np.stack(
            [xx.ravel(), yy.ravel(), np.full(xx.size, float(z0))], axis=1
        )
        phi = plummer_potential_on_points(targets, src_pos, src_mass, eps)
        out[iz] = phi.reshape(n_pix, n_pix)
    return out.astype(np.float32)


def plummer_potential_multiscale(
    pos: np.ndarray,
    mass: np.ndarray,
    component_id: np.ndarray,
    *,
    cfg: MultiScaleSliceConfig,
    eps: float = 0.05,
    n_sub: int = 2048,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Φ maps on each component's native slice grid (shared particle sources)."""
    rng = rng or np.random.default_rng(0)
    return {
        g.name: plummer_potential_on_component_slices(
            pos, mass, component_id, g, eps=eps, n_sub=n_sub, rng=rng
        )
        for g in cfg.grids
    }


def plummer_potential_on_xy_slices(
    pos: np.ndarray,
    mass: np.ndarray,
    component_id: np.ndarray,
    *,
    cfg: SliceMapConfig,
    eps: float = 0.05,
    n_sub: int = 2048,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Evaluate Plummer Φ on the midplane of each ``z`` slab (shared geometry).

    Returns
    -------
    phi : ndarray, shape (n_z, n_pix, n_pix), float32
    """
    grid = ComponentSliceGrid(
        name="shared",
        n_pix=cfg.n_pix,
        n_z=cfg.n_z,
        r_max=cfg.r_max,
        z_max=cfg.z_max,
        z_spacing="uniform",
    )
    return plummer_potential_on_component_slices(
        pos, mass, component_id, grid, eps=eps, n_sub=n_sub, rng=rng
    )
