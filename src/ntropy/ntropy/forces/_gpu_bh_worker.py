#!/usr/bin/env python3
"""Standalone GPU Barnes-Hut force computation worker.

Runs in an isolated subprocess where CuPy is imported FIRST before any PyGalactICS
module or cuBLAS loading. This is critical on NVIDIA Blackwell (sm_120) GPUs where
loading the BarnesHutTreeC C extension corrupts the CUDA driver state for all
subsequent child processes.

The tree-walk kernel uses a proper global thread index
(``blockIdx.x * blockDim.x + threadIdx.x``) so a single multi-block launch covers
all targets — essential for scaling into the millions of particles.

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
tmp_targets : str, optional
    Path to .npy file of int32 target indices. If omitted, all N particles
    are evaluated.

Output (.npz)
-------------
acc : ndarray, shape (N_targets, 3)
    Computed accelerations in code units [kpc / (100 km/s)^2].
median_ms : float
    Median wall time over timed GPU kernel calls (in milliseconds).
norm : float
    L2 norm of the acceleration vector.
rel_err : float
    Relative error vs C BarnesHutTreeC reference (``-1`` when skipped for large N).
"""
import sys
import time

import numpy as np

# CRITICAL: Import CuPy FIRST — before ANY PyGalactICS module or cuBLAS
import cupy as cp

# Force CuPy context + pinned pool BEFORE anything else CUDA-related
cp.cuda.runtime.setDevice(0)
_ = cp.zeros(1024, dtype=cp.float32)
_ = cp.array([1.0], dtype=cp.float64)
cp.cuda.runtime.deviceSynchronize()

# Parse args
tmp_pos = sys.argv[1]
tmp_mass = sys.argv[2]
tmp_eps = sys.argv[3]
theta = float(sys.argv[4])
n = int(sys.argv[5])
tmp_out = sys.argv[6]
tmp_targets = sys.argv[7] if len(sys.argv) > 7 else None

pos = np.load(tmp_pos).astype(np.float64)
mass = np.load(tmp_mass).astype(np.float64)
eps = np.load(tmp_eps).astype(np.float64)

if tmp_targets is not None:
    targets = np.load(tmp_targets).astype(np.int32)
else:
    targets = np.arange(n, dtype=np.int32)
n_targets = int(targets.size)

from ntropy.forces.gpu_bh import (  # noqa: E402
    _ensure_bh_kernels,
    _launch_bh_walk,
    _prepare_tree_state,
    _require_gpu_bh,
)

cp = _require_gpu_bh()
_ensure_bh_kernels(cp)  # compile once up front

# Build tree on CPU (safe here: CuPy imported first), upload + unpack SoA
state, n_nodes, n_leaf = _prepare_tree_state(cp, pos, mass, eps, state=None)

d_pos = cp.asarray(pos)
d_mass = cp.asarray(mass)
d_eps = cp.asarray(eps)
d_idx = cp.asarray(targets)
d_acc = cp.zeros((n_targets, 3), dtype=cp.float64)

wk, _ = _ensure_bh_kernels(cp)

# Warmup — one full multi-block walk
_launch_bh_walk(
    cp, wk, state, d_pos, d_mass, d_eps, d_idx,
    n_leaf, n_targets, n_nodes, n, theta, d_acc,
)
cp.cuda.runtime.deviceSynchronize()

# Adaptive timing budget: fewer repeats at large N (kernel dominates)
if n >= 1_000_000:
    n_runs = 3
elif n >= 200_000:
    n_runs = 5
else:
    n_runs = 10

times = []
for _ in range(n_runs):
    t0 = time.perf_counter()
    _launch_bh_walk(
        cp, wk, state, d_pos, d_mass, d_eps, d_idx,
        n_leaf, n_targets, n_nodes, n, theta, d_acc,
    )
    cp.cuda.runtime.deviceSynchronize()
    times.append((time.perf_counter() - t0) * 1e3)

acc_last = cp.asnumpy(d_acc)
median_ms = float(np.median(times))
norm = float(np.linalg.norm(acc_last))

# Full C reference is O(N log N) on CPU and dominates wall time at large N.
# Keep machine-epsilon checks for the validated small/medium regime only.
rel_err = -1.0
if n <= 100_000 and n_targets == n:
    from ntropy.forces.bhtree_c import compute_forces_bh_c as cfbc_ref

    ref_acc = cfbc_ref(pos, mass, eps, theta=theta)
    ref_norm = float(np.linalg.norm(ref_acc))
    if ref_norm > 0:
        rel_err = float(np.linalg.norm(acc_last - ref_acc) / ref_norm)

np.savez(
    tmp_out,
    acc=acc_last,
    median_ms=np.float64(median_ms),
    norm=np.float64(norm),
    rel_err=np.float64(rel_err),
)
