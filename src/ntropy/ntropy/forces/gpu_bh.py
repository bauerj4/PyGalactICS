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
from dataclasses import dataclass
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
    logger.info(
        "[GPU-BH] Blackwell GPU detected. Import CuPy before bhtree_c for "
        "in-process walks; set NTROPY_GPU_BH_SUBPROCESS=1 to force the worker."
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
        Target particle indices. ``None`` evaluates all particles.

    Returns
    -------
    acc : ndarray, shape (N_targets, 3)
        Computed accelerations.
    """
    import subprocess as _subproc
    import tempfile as _tmpfile
    import os as _os
    import sys as _sys

    # Write inputs to temp files (numpy format for zero-overhead transfer)
    tmp_pos = _tmpfile.mktemp(suffix='.npy')
    tmp_mass = _tmpfile.mktemp(suffix='.npy')
    tmp_eps = _tmpfile.mktemp(suffix='.npy')
    np.save(tmp_pos, pos)
    np.save(tmp_mass, mass)
    np.save(tmp_eps, eps)

    n = len(mass)
    tmp_targets = None
    if target_indices is not None:
        tmp_targets = _tmpfile.mktemp(suffix='.npy')
        np.save(tmp_targets, np.asarray(target_indices, dtype=np.int32))

    # Write output to temp file
    tmp_out = _tmpfile.mktemp(suffix='.npz')

    # Get path to this module's worker script
    pkg_dir = _os.path.dirname(__file__)
    worker_script = _os.path.join(pkg_dir, '_gpu_bh_worker.py')

    # Scale timeout with N: O(N log N) walk + optional CPU reference check.
    timeout_s = max(180, int(60 + n / 5_000))

    cmd = [_sys.executable, worker_script,
           tmp_pos, tmp_mass, tmp_eps,
           str(theta), str(n), tmp_out]
    if tmp_targets is not None:
        cmd.append(tmp_targets)

    r = _subproc.run(cmd, capture_output=True, text=True, timeout=timeout_s)

    if r.returncode != 0:
        err = r.stderr[-2000:] if len(r.stderr) > 2000 else r.stderr
        raise RuntimeError(f"GPU BH worker failed:\n{err}")

    result = np.load(tmp_out, allow_pickle=False)
    acc = result['acc']
    ms = float(result['median_ms'])
    logger.debug("[GPU-BH] Worker done: %.1fms", ms)

    cleanup = [tmp_pos, tmp_mass, tmp_eps, tmp_out]
    if tmp_targets is not None:
        cleanup.append(tmp_targets)
    for f in cleanup:
        try:
            _os.unlink(f)
        except OSError:
            pass

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

# Single module containing BOTH kernels — prevents CuPy/Blackwell multi-module corruption.
#
# CRITICAL: tid = blockIdx.x * blockDim.x + threadIdx.x  (global thread index).
# An earlier version used only threadIdx.x, which silently ignored all blocks
# beyond the first and forced a catastrophic 256-target chunking workaround.
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

    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        out_ch[i*8+c] = (int)nodes_flat[row+9+c];
    }
    out_ls[i] = (int)nodes_flat[row+17];
    out_lc[i] = (int)nodes_flat[row+18];
}

extern "C" __global__ void bh_walk_kernel(
    // Packed tree nodes (SoA layout) — geometric centers unused in walk:
    const double *__restrict__ n_mx,   // [n_nodes] com x
    const double *__restrict__ n_my,   // [n_nodes] com y
    const double *__restrict__ n_mz,   // [n_nodes] com z
    const double *__restrict__ n_sz,   // [n_nodes] size
    const double *__restrict__ n_ms,   // [n_nodes] mass
    const uchar  *__restrict__ n_il,   // [n_nodes] is_leaf (uint8)
    const int      *__restrict__ n_ch, // [n_nodes*8] children indices
    const int      *__restrict__ n_ls, // [n_nodes] leaf_start
    const int      *__restrict__ n_lc, // [n_nodes] leaf_count
    const int      *__restrict__ d_leaf_indices, // [n_leaf]
    const double  *  __restrict__ d_pos,   // [N*3]
    const double  *  __restrict__ d_mass,  // [N]
    const double  *  __restrict__ d_eps,   // [N]
    const int      *__restrict__ idx,      // [n_targets]
    const int       n_leaf,
    const int       n_targets,
    const int       n_nodes,
    const int       n,
    const int       leaf_direct,     // 1 => n_ls is a direct particle index
    const double    theta_sq,
    double         * __restrict__ acc) // [n_targets*3]
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_targets) return;

    const int target = idx[tid];
    if ((unsigned)target >= (unsigned)n) {
        acc[tid*3+0] = 0.0;
        acc[tid*3+1] = 0.0;
        acc[tid*3+2] = 0.0;
        return;
    }

    const double px = d_pos[target*3+0];
    const double py = d_pos[target*3+1];
    const double pz = d_pos[target*3+2];
    const double t_eps = d_eps[target];
    const double t_eps2 = t_eps * t_eps;

    double ax = 0.0, ay = 0.0, az = 0.0;

    // Iterative tree walk (avoids deep recursion on GPU).
    // Depth of a balanced octree for 1e8 particles is ~9; 64 covers
    // severely unbalanced cases without blowing local memory.
    int node_stack[64];
    int stack_top = 0;
    node_stack[stack_top++] = 0;  // root

    while (stack_top > 0) {
        const int node = node_stack[--stack_top];
        if ((unsigned)node >= (unsigned)n_nodes) continue;

        const double ms = n_ms[node];
        if (ms <= 0.0) continue;

        const double nm_x = n_mx[node];
        const double nm_y = n_my[node];
        const double nm_z = n_mz[node];
        const double dx_c = nm_x - px;
        const double dy_c = nm_y - py;
        const double dz_c = nm_z - pz;
        const double dist_sq = dx_c*dx_c + dy_c*dy_c + dz_c*dz_c;

        if (n_il[node] != 0) {
            // Leaf — pairwise sum over contained particles
            const int start = n_ls[node];
            const int count = n_lc[node];
            if (count <= 0) continue;

            for (int k = 0; k < count; ++k) {
                int pidx;
                if (leaf_direct) {
                    pidx = start + k;
                } else {
                    const int leaf_offset = start + k;
                    if ((unsigned)leaf_offset >= (unsigned)n_leaf) continue;
                    pidx = d_leaf_indices[leaf_offset];
                }
                if ((unsigned)pidx >= (unsigned)n) continue;
                if (pidx == target) continue;

                const double dx = d_pos[pidx*3+0] - px;
                const double dy = d_pos[pidx*3+1] - py;
                const double dz = d_pos[pidx*3+2] - pz;
                const double r2 = dx*dx + dy*dy + dz*dz;

                const double h = 0.5 * (t_eps + d_eps[pidx]);
                const double r2h = r2 + h*h;
                // inv_r3 = 1 / (r2+h2)^{1.5}  via rsqrt (matches C fast_inv_r3)
                const double inv_r = rsqrt(r2h);
                const double inv_r3 = inv_r * inv_r * inv_r;
                const double f = d_mass[pidx] * inv_r3;

                ax += f * dx;
                ay += f * dy;
                az += f * dz;
            }
        } else {
            // Internal cell — opening criterion on squared quantities
            const double sz = n_sz[node];
            if (dist_sq > 0.0 && (sz * sz) < theta_sq * dist_sq) {
                const double r2h = dist_sq + t_eps2;
                const double inv_r = rsqrt(r2h);
                const double inv_r3 = inv_r * inv_r * inv_r;
                const double f = ms * inv_r3;
                ax += f * dx_c;
                ay += f * dy_c;
                az += f * dz_c;
            } else {
                // Descend — push children reverse order (stable stack order)
                const int base = node * 8;
                #pragma unroll
                for (int c = 7; c >= 0; --c) {
                    const int child = n_ch[base + c];
                    if (child > 0 && child < n_nodes && stack_top < 64) {
                        node_stack[stack_top++] = child;
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

# ------------------------------------------------------------------ #
# Compiled kernel cache — PRE-COMPILED at module import to avoid
# secondary RawModule compilation on Blackwell (sm_120), which corrupts
# the CUDA allocator for subsequent allocations.
# ------------------------------------------------------------------ #

_bh_kernel_cache: dict[str, Any] | None = None

if _cp is not None and _cuda_context_established:
    try:
        # Compile BOTH kernels NOW at import — before any bhtree_c cuBLAS load.
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


def _ensure_bh_kernels(cp: Any) -> tuple[Any, Any]:
    """Compile (once) and return (walk, unpack) kernels."""
    global _bh_kernel_cache
    if _bh_kernel_cache is None:
        mod = cp.RawModule(code=_COMBINED_KERNEL_SRC)
        _bh_kernel_cache = {
            "walk": mod.get_function("bh_walk_kernel"),
            "unpack": mod.get_function("bh_unpack_kernel"),
            "_mod": mod,
        }
    return _bh_kernel_cache["walk"], _bh_kernel_cache["unpack"]


def _get_bh_kernels() -> tuple[Any, Any]:
    """Return BH kernels, compiling lazily if needed."""
    cp = _require_gpu_bh()
    return _ensure_bh_kernels(cp)


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
    leaf_direct: int = 0

    def build_on_gpu(self) -> None:
        """Unpack the packed node buffer into SoA layout on the GPU.

        Resets any existing SoA arrays to avoid stale pointers from crashes.
        """
        cp = _require_gpu_bh()
        if self.d_nodes_flat is None:
            raise RuntimeError("No nodes data available; create from a built tree.")

        n = self.n_nodes

        for attr in ("d_cx", "d_cy", "d_cz", "d_mx", "d_my", "d_mz",
                      "d_sz", "d_ms", "d_il", "d_ch", "d_ls", "d_lc"):
            setattr(self, attr, None)

        BLOCK = 256
        grid = (int((n + BLOCK - 1) // BLOCK), 1, 1)

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

        _, up = _get_bh_kernels()
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
        return self.d_mx is not None


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

_BLOCK_SIZE = 256  # threads per block for BH walk kernel


def _launch_bh_walk(
    cp: Any,
    walk_kernel: Any,
    state: GpuBhState,
    d_pos: Any,
    d_mass: Any,
    d_eps: Any,
    d_idx: Any,
    n_leaf: int,
    n_targets: int,
    n_nodes: int,
    n: int,
    theta: float,
    d_acc: Any,
) -> None:
    """Launch a single multi-block BH tree-walk covering all targets."""
    theta_sq = float(theta) * float(theta)
    block = min(_BLOCK_SIZE, max(n_targets, 1))
    grid = (int((n_targets + block - 1) // block), 1, 1)
    leaf_direct = int(state.leaf_direct)

    # Ensure a non-null leaf pointer when leaf_direct (sentinel unused).
    d_leaf = state.d_leaf_indices
    if d_leaf is None:
        d_leaf = cp.zeros(1, dtype=cp.int32)

    walk_kernel(
        grid=grid,
        block=(block, 1, 1),
        args=(
            state.d_mx, state.d_my, state.d_mz,
            state.d_sz, state.d_ms, state.d_il,
            state.d_ch, state.d_ls, state.d_lc,
            d_leaf,
            d_pos, d_mass, d_eps,
            d_idx,
            cp.int32(n_leaf),
            cp.int32(n_targets),
            cp.int32(n_nodes),
            cp.int32(n),
            cp.int32(leaf_direct),
            cp.float64(theta_sq),
            d_acc,
        ),
    )


def _prepare_tree_state(
    cp: Any,
    pos: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
    state: GpuBhState | None,
    bh_opts: Any | None = None,
) -> tuple[GpuBhState, int, int]:
    """Build CPU tree, upload, unpack SoA. Returns (state, n_nodes, n_leaf)."""
    from dataclasses import replace

    from ntropy.config import BhOptimizationsConfig
    from ntropy.forces.bhtree_c import (
        PACK_FORMAT_LEGACY,
        BarnesHutTreeC,
        extension_available,
    )

    if not extension_available():
        raise ImportError("GPU BH requires the C Barnes-Hut extension.")

    # GPU unpack kernel only understands legacy 19-float rows. The optimized
    # preset sets native_pack=True (raw BHNode bytes for MPI), which leaves
    # packed["nodes"] empty and produced all-zero accelerations.
    opts = bh_opts if bh_opts is not None else BhOptimizationsConfig()
    opts = replace(opts, native_pack=False)

    tree = BarnesHutTreeC.build(pos, mass, eps, bh_opts=opts)
    packed = tree.pack_buffers()
    pack_format = int(packed.get("pack_format", PACK_FORMAT_LEGACY))
    nodes_2d = np.asarray(packed["nodes"], dtype=np.float64)
    if pack_format != PACK_FORMAT_LEGACY or nodes_2d.size == 0:
        raise RuntimeError(
            "GPU BH requires legacy float pack buffers; "
            f"got pack_format={pack_format}, nodes.shape={nodes_2d.shape}. "
            "This is an internal bug — native_pack must be forced off for GPU."
        )
    n_nodes = int(nodes_2d.shape[0])
    nodes_flat = np.ascontiguousarray(nodes_2d.ravel(), dtype=np.float64)

    leaf_raw = np.asarray(
        packed.get("leaf_indices", np.array([], dtype=np.int32)),
        dtype=np.int32,
    )
    leaf_direct = 0
    if leaf_raw.size == 0:
        leaf_raw = np.array([-1], dtype=np.int32)
        leaf_direct = 1
    elif int(leaf_raw[0]) == -1:
        leaf_direct = 1
    n_leaf = int(leaf_raw.size)

    st = GpuBhState(n_nodes=n_nodes, n_particles=len(mass)) if state is None else state
    st.n_nodes = n_nodes
    st.n_particles = len(mass)
    st.leaf_direct = leaf_direct
    st.d_nodes_flat = cp.asarray(nodes_flat)
    st.d_leaf_indices = cp.asarray(leaf_raw)

    BLOCK = 256
    grid = (int((n_nodes + BLOCK - 1) // BLOCK), 1, 1)
    st.d_cx = cp.empty(n_nodes, dtype=cp.float64)
    st.d_cy = cp.empty(n_nodes, dtype=cp.float64)
    st.d_cz = cp.empty(n_nodes, dtype=cp.float64)
    st.d_mx = cp.empty(n_nodes, dtype=cp.float64)
    st.d_my = cp.empty(n_nodes, dtype=cp.float64)
    st.d_mz = cp.empty(n_nodes, dtype=cp.float64)
    st.d_sz = cp.empty(n_nodes, dtype=cp.float64)
    st.d_ms = cp.empty(n_nodes, dtype=cp.float64)
    st.d_il = cp.empty(n_nodes, dtype=cp.uint8)
    st.d_ch = cp.empty(n_nodes * 8, dtype=cp.int32)
    st.d_ls = cp.empty(n_nodes, dtype=cp.int32)
    st.d_lc = cp.empty(n_nodes, dtype=cp.int32)

    _, up = _ensure_bh_kernels(cp)
    up(
        grid=grid,
        block=(BLOCK, 1, 1),
        args=(
            st.d_nodes_flat, cp.int32(n_nodes),
            st.d_cx, st.d_cy, st.d_cz,
            st.d_mx, st.d_my, st.d_mz,
            st.d_sz, st.d_ms, st.d_il,
            st.d_ch, st.d_ls, st.d_lc,
        ),
    )
    return st, n_nodes, n_leaf


def compute_forces_gpu_bh(
    pos: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
    *,
    theta: float = 0.5,
    target_indices: np.ndarray | None = None,
    state: GpuBhState | None = None,
    bh_opts: Any | None = None,
) -> np.ndarray:
    """Compute Barnes-Hut softened accelerations on GPU.

    This function implements a **hybrid CPU-build / GPU-walk** Barnes-Hut
    algorithm.  Import CuPy before any cuBLAS-using extension so the in-process
    path works on Blackwell.  Set ``NTROPY_GPU_BH_SUBPROCESS=1`` to force the
    isolated worker instead.

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
        Optional device buffer holder. Ignored when
        ``NTROPY_GPU_BH_SUBPROCESS=1``.
    bh_opts : BhOptimizationsConfig, optional
        C tree-build optimizations (Morton ordering, fast paths, …).

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

    # Subprocess isolation is optional. Prefer in-process when CuPy owns the
    # CUDA context (import cupy before bhtree_c). Force with:
    #   NTROPY_GPU_BH_SUBPROCESS=1
    import os as _os

    force_subprocess = _os.environ.get("NTROPY_GPU_BH_SUBPROCESS", "").strip() in (
        "1", "true", "True", "yes",
    )
    if _BLACKWELL_GPU and force_subprocess:
        if state is not None:
            logger.warning(
                "[GPU-BH] State ignored under NTROPY_GPU_BH_SUBPROCESS=1."
            )
        return _run_gpu_bh_subprocess(pos, mass, eps, theta, target_indices)

    t0 = time()
    logger.debug("[GPU-BH] Starting GPU Barnes-Hut (N=%d)", n)

    if target_indices is None:
        targets = np.arange(n, dtype=np.int32)
        n_targets = n
    else:
        targets = np.asarray(target_indices, dtype=np.int32)
        n_targets = len(targets)

    st, n_nodes, n_leaf = _prepare_tree_state(
        cp, pos, mass, eps, state, bh_opts=bh_opts
    )

    d_pos = cp.asarray(pos)
    d_mass = cp.asarray(mass)
    d_eps = cp.asarray(eps)
    d_idx = cp.asarray(targets)
    d_acc = cp.zeros((n_targets, 3), dtype=cp.float64)

    wk, _ = _ensure_bh_kernels(cp)
    _launch_bh_walk(
        cp, wk, st, d_pos, d_mass, d_eps, d_idx,
        n_leaf, n_targets, n_nodes, n, theta, d_acc,
    )
    cp.cuda.runtime.deviceSynchronize()

    result = cp.asnumpy(d_acc)
    logger.debug("[GPU-BH] Complete: %.1fms total", (time() - t0) * 1000)
    return result
