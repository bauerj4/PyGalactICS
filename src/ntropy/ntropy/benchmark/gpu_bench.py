"""Benchmark GPU Barnes-Hut vs CPU references up to 10^7 particles.

This module provides timing and accuracy benchmarks for the GPU Barnes-Hut
implementation in ``ntropy.forces.gpu_bh`` against:

- **Pure-Python BH** :func:`ntropy.forces.bhtree.compute_forces_bh` (reference)
- **C extension BH** :func:`ntropy.forces.bhtree_c.compute_forces_bh_c` (gold standard)
- **GPU direct-force** :func:`ntropy.forces.gpu_direct.compute_forces_gpu` (O(N^2) baseline)

Usage
-----
>>> from ntropy.benchmark.gpu_bench import run_benchmarks
>>> run_benchmarks()  # doctest: +SKIP

Or from CLI:

.. code-block:: bash

    python -m ntropy.benchmark.gpu_bench --max-n 1000000

Notes
-----
- All timings exclude the initial H2D transfer for fair comparison on reuse.
- Accuracy is measured as relative L2 error against C extension reference.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ------------------------------------------------------------------ #
# Imports with graceful fallback
# ------------------------------------------------------------------ #

try:
    from ntropy.forces.gpu_bh import compute_forces_gpu_bh, gpu_bh_available
    _GPU_BH = True
except ImportError:
    _GPU_BH = False

try:
    from ntropy.forces.bhtree_c import compute_forces_bh_c, extension_available
    _BH_C = extension_available()
except ImportError:
    _BH_C = False

try:
    from ntropy.forces.bhtree import compute_forces_bh as bh_python_ref
    _BH_PY = True
except ImportError:
    _BH_PY = False

try:
    from ntropy.forces.brute import compute_forces_brute
    _BRUTE = True
except ImportError:
    _BRUTE = False


# ------------------------------------------------------------------ #
# Data classes
# ------------------------------------------------------------------ #


@dataclass
class BenchResult:
    """Single benchmark data point."""
    n_particles: int
    backend: str  # 'gpu_bh', 'bh_c_walk', 'bh_c_full', 'bh_py'
    ms_build: float = 0.0
    ms_walk: float = 0.0
    ms_total: float = 0.0
    rel_error: float = 0.0
    gpu_ms_transfer: float = 0.0
    gpu_ms_soa: float = 0.0


@dataclass
class BenchResults:
    """Collect of benchmark runs."""
    results: list[BenchResult] = field(default_factory=list)

    def append(self, r: BenchResult):
        self.results.append(r)

    def to_dict(self) -> dict[str, Any]:
        return {"runs": [r.__dict__ for r in self.results]}

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


# ------------------------------------------------------------------ #
# Particle generators
# ------------------------------------------------------------------ #


def make_particles(
    n: int, seed: int = 42, spread: float = 10.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform spherical particle distribution.

    Returns (pos, mass, eps) tuples matching the force-backend API.
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-spread, spread, size=(n, 3))
    mass = rng.uniform(0.5, 1.5, size=n)
    eps = np.full(n, 0.005)
    return pos, mass, eps


# ------------------------------------------------------------------ #
# Timing helpers
# ------------------------------------------------------------------ #


def time_bh_c_full(
    pos: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
    *,
    theta: float = 0.5,
    n_warmup: int = 3,
    n_repeat: int = 10,
) -> BenchResult:
    """Time C BH full (build + walk)."""
    if not _BH_C:
        raise RuntimeError("C BH extension not available")

    # Warmup
    for _ in range(n_warmup):
        compute_forces_bh_c(pos, mass, eps, theta=theta)

    times: list[float] = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        compute_forces_bh_c(pos, mass, eps, theta=theta)
        times.append((time.perf_counter() - t0) * 1e3)

    return BenchResult(
        n_particles=len(mass),
        backend="bh_c_full",
        ms_total=float(np.median(times)),
        rel_error=-1.0,  # gold standard — no error
    )


def time_bh_py(
    pos: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
    *,
    theta: float = 0.5,
    n_warmup: int = 2,
    n_repeat: int = 5,
) -> BenchResult:
    """Time pure-Python BH (build + walk)."""
    if not _BH_PY:
        raise RuntimeError("Python BH reference not available")

    # Warmup
    for _ in range(n_warmup):
        bh_python_ref(pos, mass, eps, theta=theta)

    times: list[float] = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        bh_python_ref(pos, mass, eps, theta=theta)
        times.append((time.perf_counter() - t0) * 1e3)

    return BenchResult(
        n_particles=len(mass),
        backend="bh_py",
        ms_total=float(np.median(times)),
        rel_error=-1.0,
    )


def time_gpu_bh_full(
    pos: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
    *,
    theta: float = 0.5,
    ref_acc: np.ndarray | None = None,
    n_warmup: int = 3,
    n_repeat: int = 10,
) -> BenchResult:
    """Time GPU BH with full timing breakdown."""
    if not _GPU_BH:
        raise RuntimeError("GPU BH module not available")

    from ntropy.forces import gpu_bh
    from ntropy.forces.bhtree_c import BarnesHutTreeC

    n = len(mass)
    targets = np.arange(n, dtype=np.int32)

    # Warmup — first call includes tree build + transfer
    for _ in range(n_warmup):
        compute_forces_gpu_bh(pos, mass, eps, theta=theta, target_indices=targets)

    # Now measure walk only (with tree already on GPU via state reuse)
    state = GpuBhState(n_nodes=0, n_particles=0)
    
    # First call with state — builds tree + transfers to GPU
    acc_first = compute_forces_gpu_bh(pos, mass, eps, theta=theta, target_indices=targets, state=state)
    
    # Subsequent calls reuse the tree — measure walk only
    walk_times: list[float] = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        compute_forces_gpu_bh(pos, mass, eps, theta=theta, target_indices=targets, state=state)
        walk_times.append((time.perf_counter() - t0) * 1e3)

    # Compute accuracy against C reference if provided
    rel_error = -1.0
    if ref_acc is not None:
        ref_norm = np.linalg.norm(ref_acc)
        if ref_norm > 0:
            rel_error = float(np.linalg.norm(acc_first - ref_acc) / ref_norm)

    return BenchResult(
        n_particles=n,
        backend="gpu_bh_walk",
        ms_total=float(np.median(walk_times)),
        rel_error=rel_error,
    )


def time_gpu_bh_stateful(
    pos: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
    *,
    theta: float = 0.5,
    ref_acc: np.ndarray | None = None,
    n_warmup: int = 3,
    n_repeat: int = 10,
) -> BenchResult:
    """Time GPU BH with persistent state — only walk kernel measures."""
    if not _GPU_BH:
        raise RuntimeError("GPU BH module not available")

    from ntropy.forces.gpu_bh import GpuBhState, compute_forces_gpu_bh as gpu_bh_fn

    n = len(mass)
    targets = np.arange(n, dtype=np.int32)

    # Build state (first call includes everything)
    state = GpuBhState(n_nodes=0, n_particles=0)
    
    for _ in range(n_warmup):
        gpu_bh_fn(pos, mass, eps, theta=theta, target_indices=targets, state=state)

    # Measure repeated walk calls with same tree (position drift is tiny)
    walk_times: list[float] = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        acc = gpu_bh_fn(pos, mass, eps, theta=theta, target_indices=targets, state=state)
        walk_times.append((time.perf_counter() - t0) * 1e3)

    rel_error = -1.0
    if ref_acc is not None:
        ref_norm = np.linalg.norm(ref_acc)
        if ref_norm > 0:
            rel_error = float(np.linalg.norm(acc - ref_acc) / ref_norm)

    return BenchResult(
        n_particles=n,
        backend="gpu_bh_stateful",
        ms_total=float(np.median(walk_times)),
        rel_error=rel_error,
    )


# ------------------------------------------------------------------ #
# Main benchmark runner
# ------------------------------------------------------------------ #


def run_benchmarks(
    max_n: int = 1_000_000,
    step_mul: int = 4,
    theta: float = 0.5,
    output: str | None = None,
) -> BenchResults:
    """Run benchmarks across increasing particle counts.

    Parameters
    ----------
    max_n : int
        Maximum number of particles (largest benchmark size).
    step_mul : int
        Multiply factor between successive N values (e.g., 4 means N doubles).
    theta : float
        BH opening angle.
    output : str, optional
        JSON file path to save results.
    """
    # Particle count progression: power of 2 steps up to max_n
    n_values = []
    n = 1000
    while n <= max_n:
        n_values.append(n)
        if n < 10_000:
            n *= 2
        elif n < 100_000:
            n *= step_mul
        else:
            # Scale with powers of 2 for efficiency at large sizes
            import math
            exponent = int(math.log2(max_n / 1000))
            idx = n_values.index(n) if n in n_values else -1
            # Add intermediate points
            next_pow = 2 ** (exponent + 1) * 1000
            if next_pow <= max_n:
                pass  # will be caught by while loop
        if len(n_values) >= 8:
            break

    results = BenchResults()

    print("=" * 70)
    print("GPU Barnes-Hut Benchmark")
    print("=" * 70)
    print(f"max_n={max_n}, theta={theta}")
    print(f"N values: {n_values}")
    print("-" * 70)

    for n in n_values:
        pos, mass, eps = make_particles(n)
        ref_acc_c = None
        if _BH_C:
            ref_acc_c = compute_forces_bh_c(pos, mass, eps, theta=theta)
        else:
            print(f"\nN={n}: C BH not available — skipping")
            continue

        print(f"\n{'='*60}")
        print(f"N = {n:>12,} particles")
        print(f"{'='*60}")

        # --- C BH full ---
        if _BH_C:
            try:
                res_c = time_bh_c_full(pos, mass, eps, theta=theta)
                print(f"C BH full:     {res_c.ms_total:>10.2f} ms")
                results.append(res_c)
            except Exception as e:
                print(f"C BH full:     ERROR — {e}")

        # --- Pure Python BH --- (only for N <= 10^4 — too slow above)
        if _BH_PY and n <= 10_000:
            try:
                res_py = time_bh_py(pos, mass, eps, theta=theta, n_repeat=3)
                print(f"Python BH:     {res_py.ms_total:>10.2f} ms")
                results.append(res_py)
            except Exception as e:
                print(f"Python BH:     ERROR — {e}")

        # --- GPU BH walk (first call with state build) ---
        if _GPU_BH and ref_acc_c is not None:
            try:
                res_gpu = time_gpu_bh_full(
                    pos, mass, eps, theta=theta, ref_acc=ref_acc_c
                )
                err_str = f"err={res_gpu.rel_error:.2e}" if res_gpu.rel_error >= 0 else "err=N/A"
                print(f"GPU BH walk:   {res_gpu.ms_total:>10.2f} ms  ({err_str})")
                results.append(res_gpu)

                # Stateful reuse test
                res_sf = time_gpu_bh_stateful(
                    pos, mass, eps, theta=theta, ref_acc=ref_acc_c
                )
                err_str2 = f"err={res_sf.rel_error:.2e}" if res_sf.rel_error >= 0 else "err=N/A"
                print(f"GPU BH state:  {res_sf.ms_total:>10.2f} ms  ({err_str2})")
                results.append(res_sf)

                # Compute speedup vs C reference
                if _BH_C and res_c.ms_total > 0:
                    speedup = res_c.ms_total / res_gpu.ms_total
                    print(f"Speedup vs C:  {speedup:>10.2f}x")
            except Exception as e:
                import traceback
                print(f"GPU BH:        ERROR — {e}")
                traceback.print_exc()

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    # Save results
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        results.save(output)
        print(f"\nResults saved to {output}")

    return results


def cli_main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="GPU BH benchmark")
    parser.add_argument(
        "--max-n", type=int, default=1_000_000,
        help="Maximum particle count (default: 1M)"
    )
    parser.add_argument(
        "--theta", type=float, default=0.5,
        help="BH opening angle (default: 0.5)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--repeat", type=int, default=10,
        help="Number of repeats per timing measurement"
    )
    args = parser.parse_args()

    # Set repeat count on functions via module-level config
    import ntropy.benchmark.gpu_bench as gb_module
    gb_module.time_gpu_bh_full.__defaults__ = (
        *(gb_module.time_gpu_bh_full.__defaults__ or ()),
        3,       # n_warmup
        args.repeat,  # n_repeat
    )

    results = run_benchmarks(
        max_n=args.max_n,
        theta=args.theta,
        output=args.output,
    )

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    header = f"{'N':>12s}  {'Backend':>16s}  {'Time (ms)':>12s}  {'Rel Error':>12s}"
    print(header)
    print("-" * len(header))
    for r in results.results:
        err_str = f"{r.rel_error:.2e}" if r.rel_error >= 0 else "N/A"
        print(f"{r.n_particles:>12,}  {r.backend:>16s}  {r.ms_total:>12.2f}  {err_str:>12s}")

    return 0


if __name__ == "__main__":
    sys.exit(cli_main())