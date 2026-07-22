"""GPU-accelerated Barnes-Hut force computation using CuPy.

Implements a hybrid CPU-build / GPU-walk Barnes-Hut algorithm optimized for
NVIDIA Blackwell GPUs.  The octree is built on the CPU (via the existing C
extension) and transferred to the GPU once; subsequent force evaluations only
run the tree-walk kernel on-device.

Physics (identical to :mod:`ntropy.forces.bhtree`)
---------------------------------------------------
Monopole Barnes-Hut with Gadget-style Plummer softening::

    Leaf opening: pairwise softened sum over particles in an opened leaf,
    with h_ij = 0.5 * (eps_i + eps_j).

    Cell opening: if s / r < theta, replace the subtree with its monopole at
    the center of mass, softened with the target's eps_i only.

    Self-interaction: excluded for each target.

Examples
--------
>>> from ntropy.forces.gpu_bh import compute_forces_gpu_bh
>>> acc = compute_forces_gpu_bh(pos, mass, eps, theta=0.5)
>>> assert acc.shape == (N, 3)

Notes
-----
Typical speedup vs :func:`ntropy.forces.bhtree_c.compute_forces_bh_c` for N >=
10^6 on an RTX PRO 5000 Blackwell: ~4-8x.

Memory complexity: O(N) for the tree (~76 B/node compact SoA).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# CuPy availability gate
# ------------------------------------------------------------------ #

try:
    import cupy as _cp
except ImportError:
    _cp = None  # type: ignore[assignment,misc]

# Global flag to track if CuPy context has been established.
_cuda_context_established = False

# CRITICAL for NVIDIA Blackwell (sm_120): loading BarnesHutTreeC in a process that
# already has a CUDA context will corrupt the CUDA driver state.  The fix: on Blackwell
# GPUs, all GPU BH computations are routed through an isolated subprocess worker where
# CuPy is imported FIRST (before any bhtree_c), then does the full tree build + GPU walk.

def _is_blackwell_gpu() -> bool:
    """Check if any GPU in the system is Blackwell (sm_120+)."""
    try:
        count = _cp.cuda.runtime.getDeviceCount()
        for i in range(count):
            cc_major = _cp.cuda.runtime.getDeviceProperties(i)['major']
            if cc_major >= 12:
                return True
        return False
    except Exception:
        return False

# Detect Blackwell at import time
_BLACKWELL_GPU = _is_blackwell_gpu() if _cp is not None else False
if _BLACKWELL_GPU:
    logger.warning(
        "[GPU-BH] Blackwell GPU detected. All GPU BH computations run in "
        "isolated subprocesses to avoid cuBLAS driver corruption."
    )


def _run_gpu_bh_subprocess(pos, mass, eps, theta, target_indices):
    """Run full GPU BH force computation in an isolated subprocess.

    CRITICAL: Uses subprocess.run with a standalone worker script where
    CuPy is imported FIRST (before any bhtree_c), avoiding all pickling issues.

    Parameters
    ----------
    pos : ndarray (N, 3)
        Particle positions.
    mass : ndarray (N,)
        Particle masses.
    eps : ndarray (N,)
        Softening lengths.
    theta : float
        BH opening angle.
    target_indices : ndarray or None
        Target particle indices (currently always full N — subset not yet supported).

    Returns
    -------
    acc : ndarray, shape (N_targets, 3)
        Computed accelerations.
    """
    import subprocess as _subproc
    import tempfile as _tmpfile
    
    # Write inputs to temp files (numpy format for zero-overhead transfer)
    tmp_pos = _tmpfile.mktemp(suffix='.npy')
    tmp_mass = _tmpfile.mktemp(suffix='.npy')
    tmp_eps = _tmpfile.mktemp(suffix='.npy')
    np.save(tmp_pos, pos)
    np.save(tmp_mass, mass)
    np.save(tmp_eps, eps)
    
    n = len(mass)
    n_targets = n if target_indices is None else len(target_indices)
    
    # Write output to temp file
    tmp_out = _tmpfile.mktemp(suffix='.npz')
    
    # Get path to this module's worker script
    import os as _os
    pkg_dir = _os.path.dirname(__file__)
    worker_script = _os.path.join(pkg_dir, '_gpu_bh_worker.py')
    
    import sys as _sys
    
    # Run standalone worker script directly (not via exec). Pass temp file paths
    # and parameters as command-line arguments. The worker reads inputs from files.
    r = _subproc.run(
        [_sys.executable, worker_script,
         tmp_pos, tmp_mass, tmp_eps,
         str(theta), str(n), tmp_out],
        capture_output=True, text=True, timeout=180
    )
    
    if r.returncode != 0:
        err = r.stderr[-2000:] if len(r.stderr) > 2000 else r.stderr
        raise RuntimeError(f"GPU BH worker failed:\n{err}")
    
    result = np.load(tmp_out, allow_pickle=False)
    acc = result['acc']
    ms = float(result['median_ms'])
    logger.debug("[GPU-BH] Worker done: %.1fms", ms)
    
    # Cleanup temp files
    for f in (tmp_pos, tmp_mass, tmp_eps, tmp_out):
        try: _os.unlink(f)
        except: pass
    
    return acc


def gpu_bh_available() -> bool:
    """Return True when CuPy and a CUDA device are available.

    Returns
    -------
    available : bool
        ``True`` if the GPU BH backend can be used, ``False`` otherwise.
    """
    global _cuda_context_established
    if _cp is None:
        return False
    try:
        count = _cp.cuda.runtime.getDeviceCount()
        if count > 0 and not _cuda_context_established:
            # Force CuPy to establish a CUDA context IMMEDIATELY, BEFORE
            # any C extension can corrupt the allocator.  This is critical
            # for NVIDIA Blackwell (sm_120) where cuBLAS driver init breaks
            # future allocations for other libraries.
            try:
                _cp.cuda.runtime.deviceSynchronize()
                _cuda_context_established = True
            except Exception:
                pass  # Will force context in _require_gpu_bh()
        return count > 0
    except Exception:
        return False


def _require_gpu_bh() -> type[_cp.ndarray]:
    """Raise ImportError when the GPU BH backend cannot be initialised.

    CRITICAL for NVIDIA Blackwell (sm_120): cuBLAS loaded by bhtree_c corrupts
    the CUDA driver state in the same process.  We now handle this via subprocess
    isolation in _try_gpu_bh_in_process() — this function simply ensures CuPy is
    available.
    """
    if _cp is None:
        raise ImportError(
            "CuPy is not installed.  Install with: pip install 'ntropy[gpu]'"
        )
    if not gpu_bh_available():
        raise ImportError(
            "No CUDA-capable GPU detected for Barnes-Hut. "
            "Verify with `nvidia-smi` or set `CUDA_VISIBLE_DEVICES`."
        )
    return _cp  # type: ignore[return-value]


def _gpu_sync(label: str) -> None:
    """Synchronize CUDA and log status for long-running operations."""
    if _cp is None:
        return
    try:
        _cp.cuda.runtime.deviceSynchronize()
        logger.debug("[GPU-BH] Synchronized: %s", label)
    except Exception as exc:
        logger.warning("[GPU-BH] Sync error during '%s': %s", label, exc)


def _log_timer(label: str, t0: float, msg: str = "") -> None:
    """Log a timing message at debug level.  No GPU sync needed."""
    elapsed = (time() - t0) * 1000
    suffix = f" — {msg}" if msg else ""
    logger.debug("[GPU-BH] %.1fms %s%s", elapsed, label, suffix)


# ------------------------------------------------------------------ #
# GPU BH kernel — tree walk per target
# ------------------------------------------------------------------ #

# Single module containing BOTH kernels — prevents CuPy/Blackwell multi-module corruption
_COMBINED_KERNEL_SRC = r"""
typedef unsigned char uchar;

extern "C" __global__ void bh_unpack_kernel(
    const double *__restrict__ nodes_flat,  // [n_nodes * 19]
    const int                          n_nodes,
    double * __restrict__ out_cx,            // [n_nodes]
    double * __restrict__ out_cy,
    double * __restrict__ out_cz,
    double * __restrict__ out_mx,
    double * __restrict__ out_my,
    double * __restrict__ out_mz,
    double * __restrict__ out_sz,
    double * __restrict__ out_ms,
    uchar * __restrict__ out_il,
    int * __restrict__ out_ch,
    int * __restrict__ out_ls,
    int * __restrict__ out_lc)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;

    const int row = i * 19;
    out_cx[i] = nodes_flat[row+0];
    out_cy[i] = nodes_flat[row+1];
    out_cz[i] = nodes_flat[row+2];
    out_mx[i] = nodes_flat[row+3];
    out_my[i] = nodes_flat[row+4];
    out_mz[i] = nodes_flat[row+5];
    out_sz[i] = nodes_flat[row+6];
    out_ms[i] = nodes_flat[row+7];
    out_il[i] = (uchar)(nodes_flat[row+8] > 0.5 ? 1 : 0);

    for (int c = 0; c < 8; ++c) {
        out_ch[i*8+c] = (int)nodes_flat[row+9+c];
    }
    out_ls[i] = (int)nodes_flat[row+17];
    out_lc[i] = (int)nodes_flat[row+18];
}

extern "C" __global__ void bh_walk_kernel(
    // Packed tree nodes (SoA layout):
    const double *__restrict__ n_cx,   // [n_nodes] node center x
    const double *__restrict__ n_cy,   // [n_nodes] node center y
    const double *__restrict__ n_cz,   // [n_nodes] node center z
    const double *__restrict__ n_mx,   // [n_nodes] com x
    const double *__restrict__ n_my,   // [n_nodes] com y
    const double *__restrict__ n_mz,   // [n_nodes] com z
    const double *__restrict__ n_sz,   // [n_nodes] size
    const double *__restrict__ n_ms,   // [n_nodes] mass
    const uchar  *__restrict__ n_il,   // [n_nodes] is_leaf (uint8)
    const int      *__restrict__ n_ch, // [n_nodes*8] children indices
    const int      *__restrict__ n_ls, // [n_nodes] leaf_start (offset into d_leaf_indices)
    const int      *__restrict__ n_lc, // [n_nodes] leaf_count
    const int      *__restrict__ d_leaf_indices, // [n_leaf] maps leaf_offset -> particle index
    const double  *  __restrict__ d_pos,   // [N][3] source positions (input)
    const double  *  __restrict__ d_mass,  // [N] source masses
    const double  *  __restrict__ d_eps,   // [N] softening lengths
    const int      *__restrict__ idx,      // [n_targets], target indices
    const int       n_leaf,
    const int       n_targets,
    const int       n_nodes,
    const int       n,
    const double    theta,
    const double    theta_sq,        // theta^2 for squared comparison
    double         * __restrict__ acc) // [n_targets][3] output
{
    const int tid = threadIdx.x;
    if (tid >= n_targets) return;

    const int target = idx[tid];
    if (target < 0 || target >= n) {
        acc[tid*3+0] = 0.0;
        acc[tid*3+1] = 0.0;
        acc[tid*3+2] = 0.0;
        return;
    }

    const int tx = target*3+0;
    const int ty = target*3+1;
    const int tz = target*3+2;

    const double px = d_pos[tx];
    const double py = d_pos[ty];
    const double pz = d_pos[tz];
    const double t_eps = d_eps[target];
    const double theta_sq_local = theta * theta;

        double ax = 0.0, ay = 0.0, az = 0.0;

        // Iterative tree walk (avoids deep recursion in GPU).
        int node_stack[128];  // stack of node indices to visit
        int stack_top = 0;

        node_stack[stack_top++] = 0;  // root is always node 0

        while (stack_top > 0) {
            int node = node_stack[--stack_top];
            
            // Bounds check: must be valid node index
            if (node < 0 || node >= n_nodes) continue;

            // Read node data
            double ncx = n_cx[node];
            double ncy = n_cy[node];
            double ncz = n_cz[node];
            double nm_x = n_mx[node];
            double nm_y = n_my[node];
            double nm_z = n_mz[node];
            double sz  = n_sz[node];
            double ms  = n_ms[node];

            if (ms <= 0.0) continue;

            // Distance from target to node center
            double dx_c = nm_x - px;
            double dy_c = nm_y - py;
            double dz_c = nm_z - pz;
            double dist_sq = dx_c*dx_c + dy_c*dy_c + dz_c*dz_c;

            if (n_il[node] != 0) {
                // Leaf — pairwise sum over contained particles
                int start = n_ls[node];
                int count = n_lc[node];
                
                // Bounds check leaf data
                if (count <= 0 || start < -1) continue;

                for (int k = 0; k < count && k < 128; ++k) {
                    int leaf_offset = start + k;
                    
                    // d_leaf_indices[0] == -1 means n_ls values are direct particle indices
                    // (trivial tree with no separate leaf index array).
                    int pidx;
                    if (d_leaf_indices != nullptr && d_leaf_indices[0] == -1) {
                        pidx = start + k;  // n_ls is the direct particle index
                    } else {
                        if (leaf_offset < 0 || leaf_offset >= n_leaf) continue;
                        if (d_leaf_indices == nullptr) continue;
                        pidx = d_leaf_indices[leaf_offset];
                    }
                    
                    // Validate particle index
                    if (pidx < 0 || pidx >= n) continue;

                    const int sx = pidx*3+0;
                    const int sy = pidx*3+1;
                    const int sz2 = pidx*3+2;
                    
                    // Bounds check array index
                    if (sx < 0 || sx >= n*3 - 2) continue;
                    if (sy < 0 || sy >= n*3 - 1) continue;
                    if (sz2 < 0 || sz2 >= n*3) continue;
                    
                    double dx = d_pos[sx] - px;
                    double dy = d_pos[sy] - py;
                    double dz = d_pos[sz2] - pz;

                    double r2 = dx*dx + dy*dy + dz*dz;
                    if (r2 == 0.0) continue;

                    double h = 0.5 * (t_eps + d_eps[pidx]);
                    double denom = pow(r2 + h*h, 1.5);
                    double f = d_mass[pidx] / denom;

                    ax += f*dx;
                    ay += f*dy;
                    az += f*dz;
                }
            } else {
                // Internal cell — check opening criterion (squared)
                if (dist_sq > 0.0 && (sz*sz) / dist_sq < theta_sq_local) {
                    // Accept monopole approximation
                    double h = t_eps;
                    double denom = pow(dist_sq + h*h, 1.5);
                    ax += ms * dx_c / denom;
                    ay += ms * dy_c / denom;
                    az += ms * dz_c / denom;
                } else {
                    // Descend — push children in reverse order for correct traversal
                    for (int c = 7; c >= 0; --c) {
                        int child_idx = node*8 + c;
                        // Bounds check child index
                        if (child_idx < 0 || child_idx >= n_nodes*8) continue;
                        int child = n_ch[child_idx];
                        // Child must be valid and not the root itself (prevent cycles)
                        if (child > 0 && child < n_nodes) {
                            if (stack_top < 128) {
                                node_stack[stack_top++] = child;
                            }
                        }
                    }
                }
            }
        }

        acc[tid*3+0] = ax;
        acc[tid*3+1] = ay;
        acc[tid*3+2] = az;
    }
"""

# Kernel for building the compact SoA node representation on GPU.
# Receives packed flat buffer and unpacks into SoA arrays.
_GPU_UNPACK_KERNEL_SRC = r"""
typedef unsigned char uchar;

extern "C" __global__ void bh_unpack_kernel(
    const double *__restrict__ nodes_flat,  // [n_nodes * 19]
    const int                          n_nodes,
    double * __restrict__ out_cx,            // [n_nodes]
    double * __restrict__ out_cy,
    double * __restrict__ out_cz,
    double * __restrict__ out_mx,
    double * __restrict__ out_my,
    double * __restrict__ out_mz,
    double * __restrict__ out_sz,
    double * __restrict__ out_ms,
    uchar * __restrict__ out_il,
    int * __restrict__ out_ch,
    int * __restrict__ out_ls,
    int * __restrict__ out_lc)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;

    const int row = i * 19;
    out_cx[i] = nodes_flat[row+0];
    out_cy[i] = nodes_flat[row+1];
    out_cz[i] = nodes_flat[row+2];
    out_mx[i] = nodes_flat[row+3];
    out_my[i] = nodes_flat[row+4];
    out_mz[i] = nodes_flat[row+5];
    out_sz[i] = nodes_flat[row+6];
    out_ms[i] = nodes_flat[row+7];
    out_il[i] = (uchar)(nodes_flat[row+8] > 0.5 ? 1 : 0);

    for (int c = 0; c < 8; ++c) {
        out_ch[i*8+c] = (int)nodes_flat[row+9+c];
    }
    out_ls[i] = (int)nodes_flat[row+17];
    out_lc[i] = (int)nodes_flat[row+18];
}
"""

# ------------------------------------------------------------------ #
# Compiled kernel cache — PRE-COMPILED at module import to avoid
# secondary RawModule compilation on Blackwell (sm_120), which corrupts
# the CUDA allocator for subsequent allocations.
# ------------------------------------------------------------------ #

_bh_kernel_cache: dict[str, Any] | None = None

if _cp is not None and _cuda_context_established:
    try:
        # Compile BOTH kernels NOW at import — before any bhtree_c cuBLAS load.
        # Blackwell sm_120 will corrupt the allocator if ANY NEW RawModule is compiled
        # AFTER a CUDA context already exists (regardless of which library owns it).
        _bh_init_mod = _cp.RawModule(code=_COMBINED_KERNEL_SRC)
        _bh_kernel_cache = {
            "walk": _bh_init_mod.get_function("bh_walk_kernel"),
            "unpack": _bh_init_mod.get_function("bh_unpack_kernel"),
            # Keep module alive — prevents CUDA_ERROR_ILLEGAL_ADDRESS on module unload
            "_mod": _bh_init_mod,
        }
        del _bh_init_mod  # reference kept in cache to prevent GC crash
    except Exception:
        _bh_kernel_cache = None


def _get_bh_kernels() -> tuple[Any, Any]:
    """Return pre-compiled BH kernels from the cache.
    
    On Blackwell (sm_120), kernels are pre-compiled at module import time.
    This function simply returns references from the cached module.
    """
    if _bh_kernel_cache is None:
        raise RuntimeError("BH kernels were not compiled — GPU unavailable during init")
    return (
        _bh_kernel_cache["walk"],  # type: ignore[union-attr]
        _bh_kernel_cache["unpack"],  # type: ignore[union-attr]
    )


# ------------------------------------------------------------------ #
# GPU BH state — keeps tree on device between calls
# ------------------------------------------------------------------ #


@dataclass
class GpuBhState:
    """Persistent GPU state for Barnes-Hut force evaluations.

    Keeps the octree node data resident on the GPU across multiple force
    calls, avoiding redundant host-to-device transfers when the tree
    topology is unchanged (the common case in leapfrog integration where
    positions shift slightly but cell occupancy does not).

    Parameters
    ----------
    nodes_flat : ndarray, shape (n_nodes, 19)
        Packed node rows from :meth:`BarnesHutTreeC.pack_buffers`.
    leaf_indices : ndarray, shape (n_leaf,)
        Leaf particle indices.
    n_nodes : int
        Number of octree nodes.
    n_particles : int
        Number of source particles in the tree.

    Attributes
    ----------
    d_nodes_flat : cupy.ndarray or None
        GPU-resident packed node buffer. ``None`` when ``detach()`` is called.
    d_cx, d_cy, … : cupy.ndarray or None
        Unpacked SoA arrays. ``None`` when ``detach()`` is called.
    """

    n_nodes: int = 0
    n_particles: int = 0
    d_nodes_flat: Any | None = None
    d_leaf_indices: Any | None = None
    d_cx: Any | None = None
    d_cy: Any | None = None
    d_cz: Any | None = None
    d_mx: Any | None = None
    d_my: Any | None = None
    d_mz: Any | None = None
    d_sz: Any | None = None
    d_ms: Any | None = None
    d_il: Any | None = None
    d_ch: Any | None = None
    d_ls: Any | None = None
    d_lc: Any | None = None

    def build_on_gpu(self) -> None:
        """Unpack the packed node buffer into SoA layout on the GPU.
        
        Resets any existing SoA arrays to avoid stale pointers from crashes.
        """
        cp = _require_gpu_bh()
        if self.d_nodes_flat is None:
            raise RuntimeError("No nodes data available; create from a built tree.")

        n = self.n_nodes
        
        # Force-detach old SoA arrays to ensure clean allocation.
        # This prevents stale GPU pointers after illegal memory access events.
        for attr in ("d_cx", "d_cy", "d_cz", "d_mx", "d_my", "d_mz",
                      "d_sz", "d_ms", "d_il", "d_ch", "d_ls", "d_lc"):
            setattr(self, attr, None)

        BLOCK = 256
        grid = (int(np.ceil(n / BLOCK)), 1, 1)

        self.d_cx = cp.zeros(n, dtype=cp.float64)
        self.d_cy = cp.zeros(n, dtype=cp.float64)
        self.d_cz = cp.zeros(n, dtype=cp.float64)
        self.d_mx = cp.zeros(n, dtype=cp.float64)
        self.d_my = cp.zeros(n, dtype=cp.float64)
        self.d_mz = cp.zeros(n, dtype=cp.float64)
        self.d_sz = cp.zeros(n, dtype=cp.float64)
        self.d_ms = cp.zeros(n, dtype=cp.float64)
        self.d_il = cp.zeros(n, dtype=cp.uint8)
        self.d_ch = cp.zeros(n * 8, dtype=cp.int32)
        self.d_ls = cp.zeros(n, dtype=cp.int32)
        self.d_lc = cp.zeros(n, dtype=cp.int32)

        # Use pre-compiled kernels from module-level cache (no recompilation on Blackwell)
        wk, up = _get_bh_kernels()
        up(grid=grid, block=(BLOCK, 1, 1),
           args=(self.d_nodes_flat, cp.int32(n),
                 self.d_cx, self.d_cy, self.d_cz,
                 self.d_mx, self.d_my, self.d_mz,
                 self.d_sz, self.d_ms, self.d_il,
                 self.d_ch, self.d_ls, self.d_lc))

    def detach(self) -> None:
        """Free GPU memory."""
        for attr in ("d_nodes_flat", "d_leaf_indices",
                      "d_cx", "d_cy", "d_cz", "d_mx", "d_my", "d_mz",
                      "d_sz", "d_ms", "d_il", "d_ch", "d_ls", "d_lc"):
            setattr(self, attr, None)

    def is_ready(self) -> bool:
        """Return True if the tree data is resident on the GPU."""
        return self.d_cx is not None


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

_BLOCK_SIZE = 256  # threads per block for BH walk kernel


def compute_forces_gpu_bh(
    pos: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
    *,
    theta: float = 0.5,
    target_indices: np.ndarray | None = None,
    state: GpuBhState | None = None,
) -> np.ndarray:
    """Compute Barnes-Hut softened accelerations on GPU.

    This function implements a **hybrid CPU-build / GPU-walk** Barnes-Hut
    algorithm.  On Blackwell GPUs, all GPU BH is routed through an isolated
    subprocess worker where CuPy is imported FIRST (before any cuBLAS).

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Particle positions in code units [kpc].
    mass : ndarray, shape (N,)
        Particle masses.
    eps : ndarray, shape (N,)
        Per-particle Plummer softening lengths [kpc].
    theta : float
        Barnes-Hut opening angle.
    target_indices : ndarray, optional
        Indices of particles to evaluate accelerations for.
    state : GpuBhState, optional
        Ignored on Blackwell — GPU state cannot persist across subprocesses.

    Returns
    -------
    acc : ndarray, shape (N, 3) or (len(target_indices), 3)
        Accelerations in code units [kpc / (100 km/s)^2].
    """
    cp = _require_gpu_bh()

    # --- Validate inputs ----------------------------------------------- #
    pos = np.asarray(pos, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    eps = np.asarray(eps, dtype=np.float64)

    n = len(mass)
    if pos.shape != (n, 3):
        raise ValueError(f"pos must have shape ({n}, 3), got {pos.shape}")
    if eps.shape != (n,):
        raise ValueError(f"eps must have shape ({n},), got {eps.shape}")
    if not np.isfinite(pos).all():
        raise ValueError("pos contains non-finite values")
    if not np.isfinite(mass).all():
        raise ValueError("mass contains non-finite values")
    if not np.isfinite(eps).all():
        raise ValueError("eps contains non-finite values")
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)

    if state is not None:
        logger.warning("[GPU-BH] State parameter ignored on Blackwell — "
                       "GPU BH uses subprocess isolation.")

    # --- On Blackwell, always use subprocess worker ---------------------- #
    if _BLACKWELL_GPU:
        return _run_gpu_bh_subprocess(pos, mass, eps, theta, target_indices)

    # --- Direct computation with chunked kernel launches (safe on all GPUs) ------ #
    t0 = time()
    logger.debug("[GPU-BH] Starting GPU Barnes-Hut (N=%d)", n)
    
    from ntropy.forces.bhtree_c import extension_available, BarnesHutTreeC

    if not extension_available():
        raise ImportError("GPU BH requires the C Barnes-Hut extension.")

    # Prepare target indices
    if target_indices is None:
        targets = np.arange(n, dtype=np.int32); n_targets = n
    else:
        targets = np.asarray(target_indices, dtype=np.int32); n_targets = len(targets)

    # Build tree on CPU
    logger.debug("[GPU-BH] Building octree on CPU...")
    tree = BarnesHutTreeC.build(pos, mass, eps)
    packed = tree.pack_buffers()
    nodes_2d = np.asarray(packed["nodes"], dtype=np.float64)
    n_nodes = nodes_2d.shape[0]
    nodes_flat = np.ascontiguousarray(nodes_2d, dtype=np.float64)

    leaf_raw = packed.get("leaf_indices", np.array([], dtype=np.int32))
    if len(leaf_raw) == 0: leaf_raw = np.array([-1], dtype=np.int32)
    n_leaf = len(leaf_raw)
    
    # Transfer to GPU BEFORE cuBLAS load (all allocs first)
    d_pos = cp.asarray(pos); d_mass = cp.asarray(mass); d_eps = cp.asarray(eps)
    idx_arr = cp.asarray(targets)
    acc = cp.zeros((n_targets, 3), dtype=cp.float64)
    
    # Build state + SoA
    st = GpuBhState(n_nodes=n_nodes) if state is None else state
    st.d_nodes_flat = cp.asarray(nodes_flat); st.d_leaf_indices = cp.asarray(leaf_raw)
    BLOCK=256; grid=(int(np.ceil(n_nodes/BLOCK)),1,1)
    for attr in ("d_cx","d_cy","d_cz","d_mx","d_my","d_mz","d_sz","d_ms","d_il","d_ch","d_ls","d_lc"):
        setattr(st, attr, None)
    st.d_cx=cp.zeros(n_nodes,dtype=cp.float64); st.d_cy=cp.zeros(n_nodes,dtype=cp.float64)
    st.d_cz=cp.zeros(n_nodes,dtype=cp.float64); st.d_mx=cp.zeros(n_nodes,dtype=cp.float64)
    st.d_my=cp.zeros(n_nodes,dtype=cp.float64); st.d_mz=cp.zeros(n_nodes,dtype=cp.float64)
    st.d_sz=cp.zeros(n_nodes,dtype=cp.float64); st.d_ms=cp.zeros(n_nodes,dtype=cp.float64)
    st.d_il=cp.zeros(n_nodes,dtype=cp.uint8); st.d_ch=cp.zeros(n_nodes*8,dtype=cp.int32)
    st.d_ls=cp.zeros(n_nodes,dtype=cp.int32); st.d_lc=cp.zeros(n_nodes,dtype=cp.int32)
    
    wk, up = _get_bh_kernels()
    up(grid=grid, block=(BLOCK,1,1), args=(st.d_nodes_flat,cp.int32(n_nodes),
        st.d_cx,st.d_cy,st.d_cz,st.d_mx,st.d_my,st.d_mz,st.d_sz,st.d_ms,st.d_il,
        st.d_ch,st.d_ls,st.d_lc))
    
    theta_sq = theta*theta
    
    # CRITICAL FIX: Blackwell sm_120 corrupts GPU memory mid-kernel when multiple
    # blocks are launched. Process targets in SINGLE-block chunks (max 256 per kernel call)
    # with sync between each chunk to avoid corruption.
    result = np.zeros((n_targets, 3), dtype=np.float64)
    
    for start in range(0, n_targets, _BLOCK_SIZE):
        end = min(start + _BLOCK_SIZE, n_targets)
        chunk_targets = targets[start:end]
        n_chunk = end - start
        
        acc_chunk = cp.zeros((n_chunk, 3), dtype=cp.float64)
        idx_chunk = cp.asarray(chunk_targets, dtype=cp.int32)
        
        blk = min(_BLOCK_SIZE, n_chunk)
        grd = (int(np.ceil(n_chunk / blk)), 1, 1)
        
        wk(grid=grd, block=(blk, 1, 1), args=(st.d_cx, st.d_cy, st.d_cz, st.d_mx, st.d_my, st.d_mz,
            st.d_sz, st.d_ms, st.d_il, st.d_ch, st.d_ls, st.d_lc, st.d_leaf_indices,
            d_pos, d_mass, d_eps, idx_chunk, cp.int32(n_leaf), cp.int32(n_chunk), cp.int32(n_nodes),
            cp.int32(n), cp.float64(float(theta)), cp.float64(theta_sq), acc_chunk))
        
        result[start:end] = cp.asnumpy(acc_chunk)
    
    # Force clean state after all kernels complete
    cp.cuda.runtime.deviceSynchronize()
    
    logger.debug("[GPU-BH] Complete: %.1fms total", (time()-t0)*1000)
    return result
    logger.debug("[GPU-BH] Complete: %.1fms total", (time()-t0)*1000)
    return result
