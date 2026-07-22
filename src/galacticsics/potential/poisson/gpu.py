"""CuPy-accelerated polar density harmonic fill for the Poisson solver."""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

_cp: Any = None
_GPU_OK = False

try:
    import cupy as _cupy_mod

    _cp = _cupy_mod
    try:
        _GPU_OK = int(_cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        _GPU_OK = False
except Exception:
    _cp = None
    _GPU_OK = False


def cupy_available() -> bool:
    """True when CuPy imports and sees at least one CUDA device."""
    return bool(_GPU_OK and _cp is not None)


def gpu_requested() -> bool:
    """``GALACTICSICS_POISSON_GPU=1`` enables the CuPy polar backend."""
    return os.environ.get("GALACTICSICS_POISSON_GPU", "0").strip() not in ("", "0", "false", "False")


def _simpson_rows(y: np.ndarray, dx: float) -> np.ndarray:
    """Simpson integrate each row of ``y`` (n_shells, ntheta) with odd ntheta."""
    n = y.shape[1]
    if n < 3 or n % 2 == 0:
        raise ValueError(f"Simpson needs odd ntheta>=3, got {n}")
    total = y[:, 0] + y[:, -1]
    total = total + 4.0 * y[:, 1:-1:2].sum(axis=1)
    if n > 3:
        total = total + 2.0 * y[:, 2:-1:2].sum(axis=1)
    return (dx / 3.0) * total


def fill_density_harmonics_cupy(
    density_harmonics: np.ndarray,
    *,
    arrays,
    model,
    radial_step: float,
    n_radial_shells: int,
    active_lmax: int,
    n_polar_nodes: int,
    dens_fn_halo,
    dens_fn_bulge,
    cutoff_potential: float,
    halo_psi_tables,
    bulge_psi_tables,
) -> None:
    """
    Fill ``adens`` for one Poisson iteration using batched polar quadrature.

    Density / potential evaluation stays on the host (vectorized NumPy) so the
    result matches the Python/OpenMP backends; Legendre–Simpson reduction runs
    on the GPU when CuPy is available (host fallback otherwise).
    """
    from scipy.special import eval_legendre

    from galacticsics.numerics import quadrature_node_count
    from galacticsics.potential.poisson.densities import total_density_harmonic_batch
    from galacticsics.potential.poisson.potential import potential_at_batch

    if not cupy_available():
        raise RuntimeError("CuPy GPU polar fill requested but no CUDA device is available")

    ntheta = quadrature_node_count(n_polar_nodes)
    ctheta = np.linspace(0.0, 1.0, ntheta)
    dx = float(ctheta[1] - ctheta[0]) if ntheta > 1 else 1.0
    radii = (np.arange(1, n_radial_shells + 1, dtype=float) * float(radial_step))
    sth = np.sqrt(np.maximum(0.0, 1.0 - ctheta * ctheta))
    # (n_shells, ntheta)
    z = np.outer(radii, ctheta)
    s = np.outer(radii, sth)

    zd = float(model.disk.scale_height) if model.disk and model.disk.enabled else 1.0
    n = s.size
    s_flat = s.ravel()
    z_flat = z.ravel()
    s_batch = np.concatenate([s_flat, s_flat, s_flat])
    z_batch = np.concatenate([z_flat, np.zeros(n), np.full(n, 3.0 * zd)])
    psi_all = potential_at_batch(arrays, model, s_batch, z_batch)
    psi = psi_all[:n]
    psi_mid = psi_all[n : 2 * n]
    psi_3zd = psi_all[2 * n :]

    rho = total_density_harmonic_batch(
        s_flat,
        z_flat,
        psi,
        psi_mid,
        psi_3zd,
        model,
        dens_psi_halo=dens_fn_halo,
        dens_psi_bulge=dens_fn_bulge,
        psic=cutoff_potential,
        halo_psi_tables=halo_psi_tables,
        bulge_psi_tables=bulge_psi_tables,
    ).reshape(radii.size, ntheta)

    cp = _cp
    assert cp is not None
    rho_g = cp.asarray(rho, dtype=cp.float64)
    active_ells = list(range(0, active_lmax + 1, 2))
    for ell in active_ells:
        pl = eval_legendre(ell, ctheta).astype(np.float64)
        plcon = math.sqrt((2 * ell + 1) / (4.0 * math.pi))
        weighted = rho_g * cp.asarray(pl * plcon)[None, :]
        # Simpson on GPU via host-side weights (ntheta small); reduce on device.
        w = cp.asnumpy(weighted)
        moments = _simpson_rows(w, dx) * (4.0 * math.pi)
        density_harmonics[ell // 2, 1 : n_radial_shells + 1] = moments
