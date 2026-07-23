#!/usr/bin/env python3
"""Benchmark worker for GPU BH — uses compute_forces_gpu_bh directly.

Since CuPy is imported FIRST in this fresh subprocess, bhtree_c loads cuBLAS
safely without corrupting the CUDA allocator.  On Blackwell GPUs, this runs
in a separate process from the parent, so even though compute_forces_gpu_bh()
detects Blackwell and tries to spawn ANOTHER subprocess — that inner subprocess
works because it also imports CuPy first (via sys.executable call from parent).

However: since compute_forces_gpu_bh DOES already route through _gpu_bh_worker.py
on Blackwell, we simply USE THAT WORKER directly with its proper args format.

Simpler approach: just call compute_forces_gpu_bh and let it handle everything.
The key fix: compute_forces_gpu_bh works in the WORKER subprocess because
the worker imports CuPy first before any cuBLAS loading.

Args: npz_path theta n tmp_out
"""
import sys, time, numpy as np

# MUST import CuPy FIRST before anything CUDA-related — prevents Blackwell corruption
import cupy as cp
cp.cuda.runtime.setDevice(0)
_ = cp.zeros(1024, dtype=cp.float32)
_ = cp.array([1.0], dtype=cp.float64)
cp.cuda.runtime.deviceSynchronize()

# Parse args
npz_path = sys.argv[1]
theta_val = float(sys.argv[2])
n_val = int(sys.argv[3])
tmp_out = sys.argv[4]

data = np.load(npz_path)
pos = data['pos'].astype(np.float64)
mass = data['mass'].astype(np.float64)
eps = data['eps'].astype(np.float64)

# Import AFTER CuPy context established (safe on Blackwell)
from ntropy.forces.gpu_bh import compute_forces_gpu_bh, GpuBhState
from ntropy.forces.bhtree_c import compute_forces_bh_c as cfbc

state = GpuBhState()
targets = np.arange(n_val, dtype=np.int32)

# Warmup (5 calls)
for _ in range(5):
    compute_forces_gpu_bh(pos, mass, eps, theta=theta_val, target_indices=targets, state=state)

# Measure (10 calls)
times = []; acc_last = None
for _ in range(10):
    t0 = time.perf_counter()
    acc_last = compute_forces_gpu_bh(pos, mass, eps, theta=theta_val, target_indices=targets, state=state)
    times.append((time.perf_counter()-t0)*1e3)

median_ms = float(np.median(times))

# Accuracy vs C reference (safe — CuPy context already established)
ref_acc = cfbc(pos, mass, eps, theta=theta_val)
ref_norm = np.linalg.norm(ref_acc)
rel_err = float(np.linalg.norm(acc_last - ref_acc) / ref_norm) if ref_norm > 0 else -1.0

np.savez(tmp_out, acc=acc_last, median_ms=np.float64(median_ms), norm=np.float64(float(np.linalg.norm(acc_last))))