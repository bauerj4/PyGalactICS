"""
Force computation backends.

Brute force
    :func:`compute_forces_brute` — vectorized $O(N^2)$ Plummer sum (CPU).
    :func:`compute_forces_gpu` — GPU-accelerated $O(N^2)$ via CuPy.

Barnes–Hut (Python)
    :class:`BarnesHutTree`, :func:`compute_forces_bh` — reference octree in pure Python.

Barnes–Hut (C)
    :class:`BarnesHutTreeC`, :func:`compute_forces_bh_c` — compiled tree in ``forces/c/``;
    use ``ForceConfig.method = "bh_c"`` or see ``forces/c/PARALLEL.md`` for MPI.

Barnes–Hut (GPU)
    :class:`GpuBhState`, :func:`compute_forces_gpu_bh` — hybrid CPU-build / GPU-walk Barnes-Hut;
    use ``ForceConfig.method = "gpu_bh"``.

Availability helpers
    :func:`extension_available` — C extension built?
    :func:`gpu_available` — CuPy + CUDA device available?
    :func:`gpu_bh_available` — same as ``gpu_available`` (alias).
"""

from ntropy.forces.brute import compute_forces_brute
from ntropy.forces.bhtree import BarnesHutTree, compute_forces_bh
from ntropy.forces.bhtree_c import (
    BarnesHutTreeC,
    compute_forces_bh_c,
    extension_available,
)
from ntropy.forces.gpu_direct import (
    GpuDirectState,
    compute_forces_gpu,
    compute_forces_gpu_from_state,
    gpu_available,
    _gpu_available_backend,
)
from ntropy.forces.gpu_bh import (
    GpuBhState,
    compute_forces_gpu_bh,
    gpu_bh_available,
)

__all__ = [
    "compute_forces_brute",
    "BarnesHutTree",
    "compute_forces_bh",
    "BarnesHutTreeC",
    "compute_forces_bh_c",
    "extension_available",
    # GPU backends
    "GpuDirectState",
    "compute_forces_gpu",
    "compute_forces_gpu_from_state",
    "gpu_available",
    "_gpu_available_backend",
    "GpuBhState",
    "compute_forces_gpu_bh",
    "gpu_bh_available",
]
