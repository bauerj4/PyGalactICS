#!/usr/bin/env python3
"""GPU Barnes-Hut vs CPU benchmarks on NVIDIA Blackwell.

CRITICAL FIX: On Blackwell (sm_120), ANY CUDA library (like bhtree_c's cuBLAS)
corrupts CuPy's allocator if no CUDA context exists yet.  We fix this by ensuring
CuPy is imported FIRST in EVERY subprocess, and the benchmark parent process stays
free of ALL CUDA imports.

Usage:
    python scripts/gpu_bh_benchmark.py --max-n 100000 -o results.json
"""
import argparse
import json
import os
import subprocess as sp
import sys
import tempfile
from pathlib import Path


def _has_gpu():
    try:
        r = sp.run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                     capture_output=True, text=True, timeout=5)
        return r.returncode == 0 and len(r.stdout.strip()) > 0
    except Exception:
        return False


def run_benchmark(max_n=1_000_000, theta=0.5, output=None):
    n_values = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
    if max_n >= 200_000: n_values += [200000, 500000, 1000000]
    n_values = [n for n in n_values if n <= max_n]

    print("=" * 80)
    print("GPU Barnes-Hut Benchmark (Blackwell-safe)")
    print("=" * 80)
    print(f"max_n={max_n}, theta={theta}")
    print(f"GPU available: {_has_gpu()}")
    print("-" * 80)

    results = []
    gpu_results = {}
    c_times = {}

    # ===== GPU BH (per-N subprocess with cupy FIRST) =====
    print("\n[STEP 1] GPU BH Benchmark:")
    for n in n_values:
        pos_path = tempfile.mktemp(suffix='.npz')
        import numpy as np
        rng = np.random.default_rng(42)
        pos = rng.uniform(-10, 10, size=(n, 3))
        mass = rng.uniform(0.5, 1.5, size=n)
        eps = np.full(n, 0.005)
        np.savez(pos_path, pos=pos, mass=mass, eps=eps)

        # CRITICAL: Cupy FIRST, then gpu_bh, then bhtree_c for accuracy
        script = f"""
import cupy as cp
import time
_ = cp.zeros(1024, dtype=cp.float32)  # force CUDA context + pinned mem

import numpy as np
data = np.load(r'{pos_path}')
n = len(data['mass'])
from ntropy.forces.gpu_bh import compute_forces_gpu_bh, GpuBhState

state = GpuBhState()
t0 = time.perf_counter()
acc = compute_forces_gpu_bh(data['pos'], data['mass'], data['eps'], theta=0.5, state=state)
gpu_ms = (time.perf_counter()-t0)*1e3

# accuracy vs C after GPU
from ntropy.forces.bhtree_c import compute_forces_bh_c as cfbc
acc_c = cfbc(data['pos'], data['mass'], data['eps'], theta=0.5)
err = float(np.linalg.norm(acc - acc_c)/np.linalg.norm(acc_c))

print("gpu,%d,%.1f,%.2e" % (n, gpu_ms, err))
"""
        r = sp.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            for line in r.stdout.strip().split('\n'):
                p = line.split(',')
                if len(p) >= 4 and p[0] == 'gpu':
                    nv, ms, er = int(p[1]), float(p[2]), float(p[3])
                    gpu_results[nv] = (ms, er)
                    print(f"N={nv:>10,}  GPU BH: {ms:>10.1f} ms  err={er:.2e}")
                    results.append({"n": nv, "backend": "gpu_bh", "ms": ms, "rel_err": er})
        else:
            print(f"N={n:>10,}  GPU ERROR: {r.stderr[-300:] if r.stderr else '?'}")

    # ===== C BH benchmark (separate process) =====
    print("\n[STEP 2] C BH Benchmark:")
    for n in n_values:
        pos_path = tempfile.mktemp(suffix='.npz')
        import numpy as np
        rng = np.random.default_rng(42)
        pos = rng.uniform(-10, 10, size=(n, 3))
        mass = rng.uniform(0.5, 1.5, size=n)
        eps = np.full(n, 0.005)
        np.savez(pos_path, pos=pos, mass=mass, eps=eps)

        script = f"""
import time, numpy as np
data = np.load(r'{pos_path}')
from ntropy.forces.bhtree_c import compute_forces_bh_c as cfbc
for _ in range(3): cfbc(data['pos'], data['mass'], data['eps'])
t0=time.perf_counter()
for _ in range(10): cfbc(data['pos'], data['mass'], data['eps'])
print("cbh,%d,%.2f" % (len(data['mass']), (time.perf_counter()-t0)*1e3/10))
"""
        r = sp.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            for line in r.stdout.strip().split('\n'):
                p = line.split(',')
                if len(p) >= 3 and p[0] == 'cbh':
                    nv, mc = int(p[1]), float(p[2])
                    c_times[nv] = mc
                    print(f"N={nv:>10,}  C BH: {mc:>10.2f} ms")
                    results.append({"n": nv, "backend": "bh_c", "ms": mc})

    # Save + print table
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {output}")

    print("\n" + "=" * 80)
    print("GPU vs C BH TIMING")
    print("=" * 80)
    print(f"{'N':>10s}  {'C BH (ms)':>12s}  {'GPU BH (ms)':>12s}  {'Speedup':>10s}  {'Accuracy':>12s}")
    print("-" * 75)
    for nv in n_values:
        mc = c_times.get(nv, 0)
        mg = gpu_results.get(nv, (0,-1))[0]
        sp = mc/mg if mc > 0 and mg > 0 else 0
        er = gpu_results.get(nv, (-1,-1))[1]
        print(f"{nv:>10,}  {mc:>12.2f}  {mg:>12.1f}  {sp:>10.2f}x  {'%.2e'%er if er >= 0 else 'N/A':>12s}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-n", type=int, default=1_000_000)
    p.add_argument("-o", "--output")
    args = p.parse_args()
    run_benchmark(args.max_n, output=args.output)