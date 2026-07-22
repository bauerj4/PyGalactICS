#!/usr/bin/env python3
"""GPU Barnes-Hut vs CPU benchmarks on NVIDIA Blackwell.

CRITICAL: On Blackwell (sm_120), loading bhtree_c (cuBLAS) in ANY process
corrupts the CUDA driver for ALL subsequent child processes. The fix: run GPU
bench in a dedicated subprocess script that handles everything in isolation,
then run C bench in its own isolated subprocess.

Usage
-----
python scripts/bench_gpu_bh.py --max-n 100000
python scripts/bench_gpu_bh.py --max-n 100000 -o gpu_bench_results.json
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import subprocess as _subproc
import sys as _sys
import tempfile as _tmpfile
import os as _os
from pathlib import Path as _Path


def _has_gpu() -> bool:
    """Check GPU availability without importing CuPy or any CUDA-using module."""
    try:
        r = _subproc.run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=5)
        return r.returncode == 0 and len(r.stdout.strip()) > 0
    except Exception:
        return False


def _write_particles(n: int, seed: int = 42) -> str:
    """Write particle data to temp file; return path."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-10, 10, size=(n, 3))
    mass = rng.uniform(0.5, 1.5, size=n)
    eps = np.full(n, 0.005)
    path = _tmpfile.mktemp(suffix='.npz')
    np.savez(path, pos=pos, mass=mass, eps=eps)
    return path


def run_benchmarks(max_n: int = 1_000_000, theta: float = 0.5, output: str | None = None):
    """Run benchmarks across increasing particle counts."""
    n_values = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
    if max_n >= 200_000: n_values += [200000, 500000, 1000000]
    if max_n >= 2_000_000: n_values += [2000000, 5000000, 10000000]
    n_values = [n for n in n_values if n <= max_n]

    gpu_avail = _has_gpu()

    print("=" * 80)
    print("GPU Barnes-Hut Benchmark")
    print("=" * 80)
    print(f"max_n={max_n}, theta={theta}")
    print(f"GPU available: {gpu_avail}")
    print(f"N values: {n_values}")
    print("-" * 80)

    results = []
    gpu_results = {}
    c_times = {}

    # ===== STEP 1: GPU BH benchmark using _gpu_bh_worker.py (verified accurate) =====
    if gpu_avail:
        print("\n[STEP 1] GPU BH Benchmark:")
        
        for n in n_values:
            npz_path = _write_particles(n)
            tmp_out = _tmpfile.mktemp(suffix='.npz')
            
            # Convert NPZ to separate .npy files (worker expects this format)
            data = np.load(npz_path)
            pos_path = _tmpfile.mktemp(suffix='.npy')
            mass_path = _tmpfile.mktemp(suffix='.npy')
            eps_path = _tmpfile.mktemp(suffix='.npy')
            np.save(pos_path, data['pos'])
            np.save(mass_path, data['mass'])
            np.save(eps_path, data['eps'])
            
            # Use the WORKING _gpu_bh_worker.py directly
            worker_script = _os.path.join(
                '/home/jbauer/PyGalactICS/src/ntropy/ntropy/forces', '_gpu_bh_worker.py')
            
            r_gpu = _subproc.run(
                [_sys.executable, worker_script, pos_path, mass_path, eps_path, str(theta), str(n), tmp_out],
                capture_output=True, text=True, timeout=300)
            
            # Clean up temp files
            for f in (pos_path, mass_path, eps_path):
                try: _os.unlink(f)
                except: pass
            
            if r_gpu.returncode == 0:
                result = np.load(tmp_out, allow_pickle=False)
                ms_gpu = float(result['median_ms'])
                rel_err = float(result.get('rel_err', -1.0))
                norm_gpu = float(result['norm'])
                
                gpu_results[n] = (ms_gpu, rel_err)
                print(f"N={n:>10,}  GPU BH walk: {ms_gpu:>10.2f} ms  (err={rel_err:.2e})")
                results.append({"n": n, "backend": "gpu_bh_state", "ms_total": ms_gpu, "rel_error": rel_err})
            else:
                stderr_msg = r_gpu.stderr.strip()
                if len(stderr_msg) > 500:
                    stderr_msg = stderr_msg[-500:] + '... [truncated]'
                print(f"N={n:>10,}  GPU BH ERROR:\n{stderr_msg}")

    # ===== STEP 2: C BH benchmark (separate subprocess — no GPU imports at all) =====
    print("\n[STEP 2] C BH Benchmark:")
    
    npz_paths_c = [_write_particles(n) for n in n_values]
    c_times_file = _tmpfile.mktemp(suffix='.json')
    with open(c_times_file, 'w') as f:
        json.dump(npz_paths_c, f)
    
    c_bench_script = _tmpfile.mktemp(suffix='.py')
    with open(c_bench_script, 'w') as f:
        f.write(f'''import sys, time, json, numpy as np

npz_paths_file = r'{c_times_file}'
output_file = r'{c_times_file}'

with open(npz_paths_file) as fj:
    npz_paths = json.load(fj)

from ntropy.forces.bhtree_c import compute_forces_bh_c as cfbc

results = []
for npz_path in npz_paths:
    data = np.load(npz_path)
    nv = len(data['mass'])
    for _ in range(3): cfbc(data['pos'], data['mass'], data['eps'], theta=0.5)
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        cfbc(data['pos'], data['mass'], data['eps'], theta=0.5)
        times.append((time.perf_counter()-t0)*1e3)
    ms_val = float(np.median(times)) * 1000.0
    results.append([nv, ms_val])
    print("cbh,%d,%.2f" % (nv, ms_val))

with open(output_file, 'w') as fo:
    json.dump(results, fo)
''')
    
    r_c = _subproc.run([_sys.executable, c_bench_script], capture_output=True, text=True, timeout=600)
    
    if r_c.returncode == 0:
        with open(c_times_file) as fj:
            c_results = json.load(fj)
        for n_val, ms_c in c_results:
            c_times[n_val] = ms_c
            results.append({"n": n_val, "backend": "bh_c_full", "ms_total": ms_c})
    
    # Print from parsed data
    for n in n_values:
        if n in c_times:
            print(f"N={n:>10,}  C BH full: {c_times[n]:>10.2f} ms")

    # ===== Print final comparison table =====
    print("\n" + "=" * 80)
    print("GPU vs C BH TIMING COMPARISON")
    print("=" * 80)
    header = f"{'N':>10s}  {'C BH (ms)':>12s}  {'GPU BH (ms)':>12s}  {'Speedup':>10s}  {'Accuracy':>12s}"
    print(header)
    print("-" * len(header))

    for n_val in n_values:
        ms_c = c_times.get(n_val, 0)
        ms_gpu = gpu_results.get(n_val, (0, -1))[0] if n_val in gpu_results else 0
        speedup = ms_c / ms_gpu if ms_c > 0 and ms_gpu > 0 else 0
        err = gpu_results.get(n_val, (-1, -1))[1] if n_val in gpu_results else -1

        err_str = f"{err:.2e}" if err >= 0 else "N/A"
        print(f"{n_val:>10,}  {ms_c:>12.2f}  {ms_gpu:>12.2f}  {speedup:>10.2f}x  {err_str:>12s}")

    # Save results
    if output:
        _Path(output).parent.mkdir(parents=True, exist_ok=True)
        _Path(output).write_text(json.dumps({"results": results}, indent=2))
        print(f"\nResults saved to {output}")


def main():
    parser = argparse.ArgumentParser(description="GPU BH benchmark")
    parser.add_argument("--max-n", type=int, default=1_000_000)
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()
    run_benchmarks(max_n=args.max_n, theta=args.theta, output=args.output)


if __name__ == "__main__":
    main()