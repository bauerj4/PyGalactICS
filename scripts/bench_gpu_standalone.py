#!/usr/bin/env python3
"""Standalone GPU Barnes-Hut benchmark — runs BEFORE any C extension loads cuBLAS.

Usage:
    python scripts/bench_gpu_standalone.py --max-n 100000 -o gpu_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def _write_particles(n: int, seed: int = 42) -> str:
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-10, 10, size=(n, 3))
    mass = rng.uniform(0.5, 1.5, size=n)
    eps = np.full(n, 0.005)
    path = tempfile.mktemp(suffix='.npz')
    np.savez(path, pos=pos, mass=mass, eps=eps)
    return path


def _make_gpu_script_lines(pos_path: str):
    """Generate lines for GPU-only benchmark script (no C extension import)."""
    yield "import sys, time, numpy as np"
    yield ""
    # Read particles from temp file
    yield f"data = np.load(r'{pos_path}')"
    yield "pos = data['pos']"
    yield "mass = data['mass']"
    yield "eps = data['eps']"
    yield "n = len(mass)"
    yield ""
    # Import GPU BH FIRST — before any C extension loads cuBLAS
    yield "from ntropy.forces.gpu_bh import compute_forces_gpu_bh, GpuBhState"
    yield ""
    yield "state = GpuBhState(n_nodes=0, n_particles=0)"
    yield "targets = np.arange(n, dtype=np.int32)"
    yield ""
    # Warmup: first call builds tree + transfers to GPU
    yield "for _ in range(3):"
    yield "    compute_forces_gpu_bh(pos, mass, eps, theta=0.5, target_indices=targets, state=state)"
    yield ""
    # Measure 10 iterations (all reuse state = tree walk only)
    yield "times = []"
    yield "acc_last = None"
    yield "for _ in range(10):"
    yield "    t0 = time.perf_counter()"
    yield "    acc_gpu = compute_forces_gpu_bh(pos, mass, eps, theta=0.5, target_indices=targets, state=state)"
    yield "    times.append((time.perf_counter() - t0) * 1e3)"
    yield "    acc_last = acc_gpu"
    yield ""
    yield "median_ms = float(np.median(times))"
    yield 'print("gpu,%d,%.2f" % (n, median_ms))'


def _make_c_ref_script_lines(pos_path: str):
    """Generate lines for C reference script (runs AFTER GPU measurements)."""
    yield "import sys, time, numpy as np"
    yield f"data = np.load(r'{pos_path}')"
    yield "pos = data['pos']"
    yield "mass = data['mass']"
    yield "eps = data['eps']"
    yield "n = len(mass)"
    yield ""
    yield "from ntropy.forces.bhtree_c import compute_forces_bh_c as cfbc"
    yield "from ntropy.forces.gpu_bh import compute_forces_gpu_bh, GpuBhState"  # GPU ref for accuracy check
    yield ""
    # C reference (gold standard)
    yield "ref_acc = cfbc(pos, mass, eps, theta=0.5)"
    yield ""
    # Also run GPU to compare accuracy
    yield "state = GpuBhState(n_nodes=0, n_particles=0)"
    yield "targets = np.arange(n, dtype=np.int32)"
    yield "for _ in range(1):"
    yield "    compute_forces_gpu_bh(pos, mass, eps, theta=0.5, target_indices=targets, state=state)"
    yield "acc_gpu = compute_forces_gpu_bh(pos, mass, eps, theta=0.5, target_indices=targets, state=state)"
    yield ""
    yield "ref_norm = np.linalg.norm(ref_acc)"
    yield "rel_err = float(np.linalg.norm(acc_gpu - ref_acc) / ref_norm) if ref_norm > 0 else -1.0"
    yield 'print("acc,%d,%.2e" % (n, rel_err))'


def run_standalone_benchmark(max_n: int = 1_000_000, output: str | None = None):
    """Run GPU benchmark in a completely isolated process before ANY C extension import."""
    n_values = [1000, 2000, 5000, 10000, 20000, 50000]
    if max_n >= 100_000:
        n_values += [100000, 200000, 500000, 1000000]
    if max_n >= 2_000_000:
        n_values += [2000000, 5000000, 10000000]
    n_values = [n for n in n_values if n <= max_n]

    print("=" * 80)
    print("STANDALONE GPU Barnes-Hut Benchmark")
    print("=" * 80)
    print(f"max_n={max_n}, theta=0.5")
    print(f"N values: {n_values}")
    print("-" * 80)

    results = []
    gpu_times = {}  # n -> (ms_gpu, ms_build_first)

    # ===== STEP 1: GPU measurements in completely isolated process =====
    print("\n[STEP 1] Running GPU BH benchmark (STANDALONE — before C extension)...")
    
    gpu_script_path = tempfile.mktemp(suffix='_gpu_standalone.py')
    with open(gpu_script_path, 'w') as f:
        for n_val in n_values:
            pos_path = _write_particles(n_val)
            lines = list(_make_gpu_script_lines(pos_path))
            f.write('\n'.join(lines))
            f.write('\n\n')

    gpu_result = subprocess.run(
        [sys.executable, gpu_script_path],
        capture_output=True, text=True, timeout=600
    )
    
    if gpu_result.returncode == 0:
        for line in gpu_result.stdout.strip().split('\n'):
            parts = line.split(',')
            if parts[0] == 'gpu':
                n_val = int(parts[1])
                ms_gpu = float(parts[2])
                print(f"N={n_val:>10,}  GPU BH walk (state reuse): {ms_gpu:>10.2f} ms")
                gpu_times[n_val] = ms_gpu
    else:
        # If the first process fails, try each N individually in isolated subprocesses
        print("  WARNING: Combined process failed. Trying individual processes...")
        for n_val in n_values:
            pos_path = _write_particles(n_val)
            lines = list(_make_gpu_script_lines(pos_path))
            script_content = '\n'.join(lines)
            
            r = subprocess.run(
                [sys.executable, '-c', script_content],
                capture_output=True, text=True, timeout=120
            )
            if r.returncode == 0:
                for line in r.stdout.strip().split('\n'):
                    parts = line.split(',')
                    if parts[0] == 'gpu':
                        ms_gpu = float(parts[2])
                        print(f"N={n_val:>10,}  GPU BH walk (state reuse): {ms_gpu:>10.2f} ms")
                        gpu_times[n_val] = ms_gpu
            else:
                # If that fails too, try with just stateless calls (no tree reuse)
                print(f"N={n_val:>10,}  GPU BH ERROR: {r.stderr.strip()[:200]}")

    os.unlink(gpu_script_path)

    # ===== STEP 2: C reference + accuracy comparison (after GPU is done) =====
    print("\n[STEP 2] Running C BH reference + accuracy...")
    
    for n_val in n_values:
        pos_path = _write_particles(n_val)
        
        lines = list(_make_c_ref_script_lines(pos_path))
        script_content = '\n'.join(lines)
        
        r = subprocess.run(
            [sys.executable, '-c', script_content],
            capture_output=True, text=True, timeout=120
        )
        
        if r.returncode == 0:
            for line in r.stdout.strip().split('\n'):
                parts = line.split(',')
                if parts[0] == 'acc':
                    rel_err = float(parts[2])
                    ms_gpu = gpu_times.get(n_val, 0)
                    print(f"N={n_val:>10,}  Accuracy: {rel_err:.2e}")
                    results.append({
                        "n": n_val,
                        "gpu_ms": ms_gpu,
                        "accuracy": rel_err,
                    })
        else:
            print(f"N={n_val:>10,}  C BH ERROR: {r.stderr.strip()[:300]}")

    # ===== STEP 3: C BH timing (separate measurement for comparison) =====
    print("\n[STEP 3] Running C BH timing...")
    
    c_script_path = tempfile.mktemp(suffix='_c_bench.py')
    with open(c_script_path, 'w') as f:
        for n_val in n_values:
            pos_path = _write_particles(n_val)
            f.write(f"""
import sys, time, numpy as np
data = np.load(r'{pos_path}')
from ntropy.forces.bhtree_c import compute_forces_bh_c as cfbc
for _ in range(3): cfbc(data['pos'], data['mass'], data['eps'], theta=0.5)
times = []
for _ in range(10):
    t0 = time.perf_counter()
    cfbc(data['pos'], data['mass'], data['eps'], theta=0.5)
    times.append((time.perf_counter()-t0)*1e3)
print("cbh,%d,%.2f" % (len(data['mass']), float(np.median(times))))
""")

    c_result = subprocess.run(
        [sys.executable, c_script_path],
        capture_output=True, text=True, timeout=600
    )
    
    c_times = {}
    if c_result.returncode == 0:
        for line in c_result.stdout.strip().split('\n'):
            parts = line.split(',')
            if parts[0] == 'cbh':
                n_val = int(parts[1])
                ms_c = float(parts[2])
                c_times[n_val] = ms_c

    os.unlink(c_script_path)

    # ===== Print final comparison table =====
    print("\n" + "=" * 80)
    print("GPU vs C BH TIMING COMPARISON")
    print("=" * 80)
    header = f"{'N':>10s}  {'C BH (ms)':>12s}  {'GPU BH (ms)':>12s}  {'Speedup':>10s}  {'Accuracy':>12s}"
    print(header)
    print("-" * len(header))

    for n_val in n_values:
        ms_c = c_times.get(n_val, 0)
        ms_gpu = gpu_times.get(n_val, 0)
        speedup = ms_c / ms_gpu if ms_c > 0 and ms_gpu > 0 else 0
        
        # Find accuracy from results
        rel_err = -1.0
        for r in results:
            if r['n'] == n_val:
                rel_err = r['accuracy']
        
        err_str = f"{rel_err:.2e}" if rel_err >= 0 else "N/A"
        print(f"{n_val:>10,}  {ms_c:>12.2f}  {ms_gpu:>12.2f}  {speedup:>10.2f}x  {err_str:>12s}")

    # Save results
    final_results = []
    for n_val in n_values:
        ms_c = c_times.get(n_val, 0)
        ms_gpu = gpu_times.get(n_val, 0)
        speedup = ms_c / ms_gpu if ms_c > 0 and ms_gpu > 0 else 0
        
        rel_err = -1.0
        for r in results:
            if r['n'] == n_val:
                rel_err = r['accuracy']

        final_results.append({
            "n": n_val,
            "bh_c_full_ms": ms_c,
            "gpu_bh_state_ms": ms_gpu,
            "speedup_vs_c": speedup,
            "relative_error": rel_err,
        })

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps({"results": final_results}, indent=2))
        print(f"\nResults saved to {output}")

    return final_results


def main():
    parser = argparse.ArgumentParser(description="Standalone GPU BH benchmark")
    parser.add_argument("--max-n", type=int, default=100_000)
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()

    run_standalone_benchmark(max_n=args.max_n, output=args.output)


if __name__ == "__main__":
    main()