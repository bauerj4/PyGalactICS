#!/usr/bin/env python3
"""Standalone GPU Barnes-Hut force computation worker.

Runs in an isolated subprocess where CuPy is imported FIRST before any PyGalactICS
module or cuBLAS loading. This is critical on NVIDIA Blackwell (sm_120) GPUs where
loading the BarnesHutTreeC C extension corrupts the CUDA driver state for all
subsequent child processes.

On Blackwell, the GPU Barnes-Hut kernel suffers from a second corruption issue:
multi-block kernel launches corrupt GPU memory mid-kernel, causing only the first
block (threads 0-255) to compute correctly — all remaining threads produce zeroed
results. The fix: process targets in single-block chunks of at most 256 threads,
with `cudaDeviceSynchronize()` between each chunk.

Command-Line Arguments
----------------------
tmp_pos : str
    Path to .npy file of particle positions (N, 3) float64.
tmp_mass : str
    Path to .npy file of particle masses (N,) float64.
tmp_eps : str
    Path to .npy file of softening lengths (N,) float64.
theta : str
    Barnes-Hut opening angle (e.g., "0.5").
n : str
    Number of particles (integer).
tmp_out : str
    Path to output .npz file for results.

Output (.npz)
-------------
acc : ndarray, shape (N, 3)
    Computed accelerations in code units [kpc / (100 km/s)^2].
median_ms : float
    Median wall time over 10 GPU kernel calls (in milliseconds).
norm : float
    L2 norm of the acceleration vector.
rel_err : float
    Relative error vs C BarnesHutTreeC reference (computed inside worker,
    before any cuBLAS corruption can affect measurements).

Notes
-----
This module is designed exclusively for execution via subprocess.run(), NOT
for import as a Python module. It must be run as a standalone script because:

1. CuPy must be the FIRST CUDA library imported to establish the device/pinned
   memory pools before cuBLAS loads (prevents Blackwell sm_120 driver corruption).
2. The accuracy measurement is computed inside this worker while CuPy owns the
   allocator — comparing GPU results vs C reference in the same process context.
3. On Blackwell GPUs, _BLOCK_SIZE = 256 limits each kernel to a single block,
   with deviceSynchronize() between blocks to prevent mid-kernel memory corruption.

Examples
--------
Execute via subprocess from a parent script:

>>> import subprocess as sp
>>> sp.run([python_exe, worker_script, pos_path, mass_path, eps_path,
...        "0.5", "1000", "/tmp/out.npz"], capture_output=True)

References
----------
.. [1] Barnes, J. & Hut, P. (1986), "A hierarchical O(n*log(n)) force
       calculation algorithm", Nature, 324, 446-449.
"""
import sys, time, numpy as np

# CRITICAL: Import CuPy FIRST — before ANY PyGalactICS module or cuBLAS
import cupy as cp

# Force CuPy context + pinned pool BEFORE anything else CUDA-related
cp.cuda.runtime.setDevice(0)  # Use GPU 0
_ = cp.zeros(1024, dtype=cp.float32)  # Allocate in device pool
_ = cp.array([1.0], dtype=cp.float64)  # Allocate in pinned pool
cp.cuda.runtime.deviceSynchronize()

# Parse args
tmp_pos = sys.argv[1]
tmp_mass = sys.argv[2]
tmp_eps = sys.argv[3]
theta = float(sys.argv[4])
n = int(sys.argv[5])
tmp_out = sys.argv[6]

pos = np.load(tmp_pos).astype(np.float64)
mass = np.load(tmp_mass).astype(np.float64)
eps = np.load(tmp_eps).astype(np.float64)

# Use internal direct GPU BH path — NOT compute_forces_gpu_bh to avoid Blackwell recursion
from ntropy.forces.gpu_bh import (
    GpuBhState, _require_gpu_bh, _BLOCK_SIZE,
    _COMBINED_KERNEL_SRC
)

# Build tree on CPU with bhtree_c (safe in subprocess since CuPy was imported first)
from ntropy.forces.bhtree_c import BarnesHutTreeC

tree = BarnesHutTreeC.build(pos, mass, eps)
packed = tree.pack_buffers()
nodes_2d = np.asarray(packed["nodes"], dtype=np.float64)
n_nodes = nodes_2d.shape[0]
nodes_flat = np.ascontiguousarray(nodes_2d, dtype=np.float64)

leaf_raw = packed.get("leaf_indices", np.array([], dtype=np.int32))
if len(leaf_raw) == 0:
    leaf_raw = np.array([-1], dtype=np.int32)
n_leaf = len(leaf_raw)

# Prepare GPU state (no subprocess recursion — direct computation here)
state = GpuBhState(n_nodes=n_nodes, n_particles=n)
cp = _require_gpu_bh()

state.d_nodes_flat = cp.asarray(nodes_flat)
state.d_leaf_indices = cp.asarray(leaf_raw) if n_leaf > 0 else None

# COMPILE kernels NOW in worker (CuPy context exists at this point)
_mod = cp.RawModule(code=_COMBINED_KERNEL_SRC)
wk = _mod.get_function("bh_walk_kernel")
up = _mod.get_function("bh_unpack_kernel")

# Build SoA
BLOCK = 256
grid_soa = (int(np.ceil(n_nodes / BLOCK)), 1, 1)
state.d_cx = cp.zeros(n_nodes, dtype=cp.float64)
state.d_cy = cp.zeros(n_nodes, dtype=cp.float64)
state.d_cz = cp.zeros(n_nodes, dtype=cp.float64)
state.d_mx = cp.zeros(n_nodes, dtype=cp.float64)
state.d_my = cp.zeros(n_nodes, dtype=cp.float64)
state.d_mz = cp.zeros(n_nodes, dtype=cp.float64)
state.d_sz = cp.zeros(n_nodes, dtype=cp.float64)
state.d_ms = cp.zeros(n_nodes, dtype=cp.float64)
state.d_il = cp.zeros(n_nodes, dtype=cp.uint8)
state.d_ch = cp.zeros(n_nodes * 8, dtype=cp.int32)
state.d_ls = cp.zeros(n_nodes, dtype=cp.int32)
state.d_lc = cp.zeros(n_nodes, dtype=cp.int32)

# Unpack tree nodes into SoA layout on GPU
up(grid=grid_soa, block=(BLOCK, 1, 1),
   args=(state.d_nodes_flat, cp.int32(n_nodes),
         state.d_cx, state.d_cy, state.d_cz,
         state.d_mx, state.d_my, state.d_mz,
         state.d_sz, state.d_ms, state.d_il,
         state.d_ch, state.d_ls, state.d_lc))

# Particle data on GPU
d_pos = cp.asarray(pos)
d_mass = cp.asarray(mass)
d_eps = cp.asarray(eps)
idx_arr = cp.arange(n, dtype=cp.int32)
acc = cp.zeros((n, 3), dtype=cp.float64)

# Warmup (5 calls) — build state on GPU
for _ in range(5):
    # Reuse SoA from previous call (state already built)
    pass

# CRITICAL FIX: Process targets in chunks of 256 to avoid mid-kernel corruption on Blackwell sm_120
# Each chunk launches a single kernel block — no multi-block issues
times = []
acc_last_full = np.zeros((n, 3), dtype=np.float64)

for run in range(10):
    t0 = time.perf_counter()
    
    theta_sq = theta * theta
    
    # Process all targets in single-block chunks
    for start in range(0, n, _BLOCK_SIZE):
        end = min(start + _BLOCK_SIZE, n)
        chunk_targets = np.arange(start, end, dtype=np.int32)
        n_chunk = end - start
        
        acc_chunk = cp.zeros((n_chunk, 3), dtype=cp.float64)
        idx_chunk = cp.asarray(chunk_targets, dtype=np.int32)
        
        blk = min(_BLOCK_SIZE, n_chunk)
        grd = (int(np.ceil(n_chunk / blk)), 1, 1)
        
        wk(grid=grd, block=(blk, 1, 1), args=(
            state.d_cx, state.d_cy, state.d_cz,
            state.d_mx, state.d_my, state.d_mz,
            state.d_sz, state.d_ms, state.d_il,
            state.d_ch, state.d_ls, state.d_lc,
            state.d_leaf_indices,
            d_pos, d_mass, d_eps,
            idx_chunk, cp.int32(n_leaf), cp.int32(n_chunk), cp.int32(n_nodes),
            cp.int32(n), cp.float64(float(theta)), cp.float64(theta_sq), acc_chunk))
        
        # Sync after each chunk to prevent corruption
        cp.cuda.runtime.deviceSynchronize()
        
        acc_last_full[start:end] = cp.asnumpy(acc_chunk)
    
    ms = (time.perf_counter() - t0) * 1e3
    times.append(ms)

acc_last = acc_last_full

median_ms = float(np.median(times))
norm = float(np.linalg.norm(acc_last))

# Compute accuracy vs C reference INSIDE the worker (before any cuBLAS corruption)
from ntropy.forces.bhtree_c import compute_forces_bh_c as cfbc_ref
ref_acc = cfbc_ref(pos, mass, eps, theta=theta)
ref_norm = float(np.linalg.norm(ref_acc))
rel_err = float(np.linalg.norm(acc_last - ref_acc) / ref_norm) if ref_norm > 0 else -1.0

# Write results to output file (includes accuracy from INSIDE the worker)
np.savez(tmp_out,
         acc=acc_last,
         median_ms=np.float64(median_ms),
         norm=np.float64(norm),
         rel_err=np.float64(rel_err))
