"""Accelerated polar shell integration (OpenMP C and optional CuPy)."""

from __future__ import annotations

import importlib
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from galacticsics.potential.poisson.integrals import integrate_polar_harmonics_at_shell

try:
    from galacticsics.potential.poisson import _poisson_c
except ImportError:
    _poisson_c = None  # type: ignore[assignment]

try:
    from galacticsics.potential.poisson import gpu as _poisson_gpu
except ImportError:
    _poisson_gpu = None  # type: ignore[assignment]


def extension_available() -> bool:
    return _poisson_c is not None


def openmp_enabled() -> bool:
    if _poisson_c is None:
        return False
    return bool(_poisson_c.extension_available())


def gpu_available() -> bool:
    return _poisson_gpu is not None and bool(_poisson_gpu.cupy_available())


def reload_extension() -> bool:
    global _poisson_c
    try:
        mod = importlib.import_module("galacticsics.potential.poisson._poisson_c")
        importlib.reload(mod)
        _poisson_c = mod
        return True
    except ImportError:
        _poisson_c = None  # type: ignore[assignment]
        return False


def _resolve_poisson_threads(n_threads: int | None) -> int:
    """
    Resolve OpenMP thread count for the C polar integrator.

    Default (env unset): use all CPUs when the OpenMP extension is available,
    so the solver is not silently single-threaded. Set
    ``GALACTICSICS_POISSON_THREADS=0`` to force the Python path.
    """
    if n_threads is not None:
        return max(0, int(n_threads))
    env = os.environ.get("GALACTICSICS_POISSON_THREADS")
    if env is not None and env.strip() != "":
        return max(0, int(env))
    if openmp_enabled():
        return int(os.cpu_count() or 1)
    return 0


def _fill_density_harmonics_python(
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
    n_workers: int,
    halo_psi_tables,
    bulge_psi_tables,
) -> None:
    active_ells = list(range(0, active_lmax + 1, 2))
    shell_indices = list(range(1, n_radial_shells + 1))

    def _integrate_shell(shell_index: int) -> tuple[int, dict[int, float]]:
        shell_radius = shell_index * radial_step
        moments = integrate_polar_harmonics_at_shell(
            arrays,
            model,
            shell_radius,
            active_ells,
            n_polar_nodes,
            dens_psi_halo=dens_fn_halo,
            dens_psi_bulge=dens_fn_bulge,
            psic=cutoff_potential,
            halo_psi_tables=halo_psi_tables,
            bulge_psi_tables=bulge_psi_tables,
        )
        return shell_index, moments

    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for shell_index, moments in pool.map(_integrate_shell, shell_indices):
                for ell in active_ells:
                    density_harmonics[ell // 2, shell_index] = moments[ell]
    else:
        for shell_index in shell_indices:
            _, moments = _integrate_shell(shell_index)
            for ell in active_ells:
                density_harmonics[ell // 2, shell_index] = moments[ell]


def _pack_poisson_context(
    *,
    arrays,
    model,
    cutoff_potential: float,
    halo_psi_tables,
    bulge_psi_tables,
) -> dict:
    disk = model.disk
    pack: dict = {
        "apot": np.ascontiguousarray(arrays.apot, dtype=np.float64),
        "nr": int(model.grid.nr),
        "lmax_active": int(arrays.lmax_active),
        "dr": float(model.grid.dr),
        "psic": float(cutoff_potential),
        "has_disk": int(arrays.flags.disk and disk is not None and disk.enabled),
        "has_halo": int(model.halo is not None and model.halo.enabled),
        "has_bulge": int(model.bulge is not None and model.bulge.enabled),
    }
    if pack["has_disk"] and disk is not None:
        pack.update(
            {
                "disk_const": float(disk.disk_const),
                "disk_rd": float(disk.scale_length),
                "disk_zd": float(disk.scale_height),
                "disk_rtrunc": float(disk.outer_radius),
                "disk_trunc_width": float(disk.trunc_width),
                "disk_hole_radius": float(disk.hole_radius),
                "disk_core_radius": float(disk.core_radius),
            }
        )
    if halo_psi_tables is not None:
        energies, dens_tab, psi0 = halo_psi_tables
        order = np.argsort(energies)
        pack.update(
            {
                "halo_energies": np.ascontiguousarray(energies[order], dtype=np.float64),
                "halo_dens_psi": np.ascontiguousarray(dens_tab[order], dtype=np.float64),
                "psi0": float(psi0),
                "halo_dens_at_psi0": float(dens_tab[0]),
            }
        )
    if bulge_psi_tables is not None:
        energies, dens_tab, psi0, psid = bulge_psi_tables
        pack.update(
            {
                "bulge_energies": np.ascontiguousarray(energies, dtype=np.float64),
                "bulge_dens_psi": np.ascontiguousarray(dens_tab, dtype=np.float64),
                "bulge_psi0": float(psi0),
                "bulge_psid": float(psid),
                "bulge_dens_at_psi0": float(dens_tab[0]),
            }
        )
    return pack


def fill_density_harmonics(
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
    n_workers: int,
    halo_psi_tables,
    bulge_psi_tables,
    n_threads: int | None = None,
) -> None:
    """
    Fill ``adens`` for one Poisson iteration using the fastest available backend.

    Dispatch order:
    1. CuPy batched polar fill when ``GALACTICSICS_POISSON_GPU=1`` and a device exists
    2. OpenMP C when ``GALACTICSICS_POISSON_THREADS>0``
    3. Python (± thread pool)
    """
    use_gpu = (
        _poisson_gpu is not None
        and _poisson_gpu.gpu_requested()
        and _poisson_gpu.cupy_available()
        and halo_psi_tables is not None
    )
    if use_gpu:
        _poisson_gpu.fill_density_harmonics_cupy(
            density_harmonics,
            arrays=arrays,
            model=model,
            radial_step=float(radial_step),
            n_radial_shells=int(n_radial_shells),
            active_lmax=int(active_lmax),
            n_polar_nodes=int(n_polar_nodes),
            dens_fn_halo=dens_fn_halo,
            dens_fn_bulge=dens_fn_bulge,
            cutoff_potential=float(cutoff_potential),
            halo_psi_tables=halo_psi_tables,
            bulge_psi_tables=bulge_psi_tables,
        )
        return

    threads = _resolve_poisson_threads(n_threads)
    use_c = (
        _poisson_c is not None
        and threads != 0
        and n_workers <= 1
        and halo_psi_tables is not None
    )
    if use_c:
        pack = _pack_poisson_context(
            arrays=arrays,
            model=model,
            cutoff_potential=cutoff_potential,
            halo_psi_tables=halo_psi_tables,
            bulge_psi_tables=bulge_psi_tables,
        )
        _poisson_c.fill_polar_density_harmonics(
            density_harmonics,
            pack,
            radial_step=float(radial_step),
            active_lmax=int(active_lmax),
            n_polar_nodes=int(n_polar_nodes),
            n_threads=threads,
        )
        return

    _fill_density_harmonics_python(
        density_harmonics,
        arrays=arrays,
        model=model,
        radial_step=radial_step,
        n_radial_shells=n_radial_shells,
        active_lmax=active_lmax,
        n_polar_nodes=n_polar_nodes,
        dens_fn_halo=dens_fn_halo,
        dens_fn_bulge=dens_fn_bulge,
        cutoff_potential=cutoff_potential,
        n_workers=n_workers,
        halo_psi_tables=halo_psi_tables,
        bulge_psi_tables=bulge_psi_tables,
    )
