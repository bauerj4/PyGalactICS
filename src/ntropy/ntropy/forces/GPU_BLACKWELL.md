# GPU Barnes-Hut on NVIDIA Blackwell (sm_120)

## Implementation Overview

The GPU Barnes-Hut backend (`ntropy.forces.gpu_bh`) implements a hybrid
**CPU-build / GPU-walk** force evaluation:

1. Build the octree on the CPU with `BarnesHutTreeC`
2. Pack nodes into a flat buffer and upload to the device
3. Unpack into compact SoA arrays (`bh_unpack_kernel`)
4. Walk the tree for every target in one multi-block launch (`bh_walk_kernel`)

Physics matches the CPU monopole Barnes–Hut path (Gadget-style Plummer
softening, leaf pairwise sums, cell acceptance when \(s^2 < \theta^2 r^2\)).

Use it from ntropy via:

```python
from ntropy.config import ForceConfig
cfg.force = ForceConfig(method="gpu_bh", theta=0.5)
```

**Full usage walkthrough:** [`notebooks/gpu_bh_dbh.ipynb`](../../../../notebooks/gpu_bh_dbh.ipynb)
(DBH ICs → OpenMP sampling → GPU BH evolution + IC potential-virial check).

## IC Poisson solver (GalactICS)

N-body `gpu_bh` is separate from the **IC multipole Poisson** path in
`galacticsics`. Polar shell integration defaults to **OpenMP on all CPUs** when
the C extension is built (`GALACTICSICS_POISSON_THREADS` unset). Force serial
Python with `GALACTICSICS_POISSON_THREADS=0`. Particle sampling (disk/halo/bulge)
also defaults to OpenMP `_sampler_c` — see [`ic_sampling.md`](../../../../docs/ic_sampling.md).
Optional CuPy batching for the Poisson polar fill:

```bash
GALACTICSICS_POISSON_GPU=1 GALACTICSICS_POISSON_THREADS=0 python scripts/benchmark_solve.py --gpu
```

## GPU BH + `bh_optimizations` preset

`BhOptimizationsConfig.from_preset("optimized")` enables `native_pack` for MPI
tree broadcast. The GPU walk unpacks **legacy** 19-float node rows only, so
`compute_forces_gpu_bh` forces `native_pack=False` when packing for the device.
Without that, `pack_buffers()["nodes"]` is empty and accelerations are all zero
(particles coast → violent “relaxation”). Other optimized flags (Morton build,
fast `1/r³`, …) still apply to the CPU tree build.

## Import order (Blackwell)

On sm_120, **import CuPy and establish a CUDA context before loading
`bhtree_c`** (which pulls in cuBLAS):

```python
import cupy as cp
cp.cuda.runtime.setDevice(0)
_ = cp.zeros(1)          # establish device pool
# now safe to import ntropy / bhtree_c
```

In-process evaluation is the default and is required for simulations (a
subprocess per force call would rebuild the tree and re-upload particles every
step). Force the isolated worker only when needed:

```bash
NTROPY_GPU_BH_SUBPROCESS=1 python ...
```

## Kernel design

| Kernel | Role |
|--------|------|
| `bh_unpack_kernel` | Flat `[n_nodes × 19]` → SoA (COM, size, mass, children, leaf pointers) |
| `bh_walk_kernel` | One thread per target; iterative stack walk (depth 64) |

Critical indexing:

```cuda
const int tid = blockIdx.x * blockDim.x + threadIdx.x;
```

An earlier bug used only `threadIdx.x`, which made multi-block launches write
the first 256 targets only. That was misdiagnosed as “Blackwell mid-kernel
corruption” and worked around with 256-wide chunked launches — which destroyed
scaling above ~10⁵ particles.

Force evaluation uses `rsqrt` (same algebra as C `fast_inv_r3`):

\[
f = m \, (r^2 + h^2)^{-3/2}
  = m \cdot \mathrm{rsqrt}(r^2+h^2)^3
\]

## Scaling

Measured walk times on RTX PRO 5000 Blackwell (θ = 0.5, uniform cube):

| N | Walk [ms] | ms / (N log₂ N) |
|---|-----------|-----------------|
| 10³ | ~1 | — |
| 10⁵ | ~8 | ~5 |
| 10⁶ | ~90 | ~4.6 |
| 10⁷ | ~2.7×10³ | ~11 |

From ~5×10⁴–10⁶ the cost tracks O(N log N). Beyond that, irregular tree
traffic increases the prefactor; still far faster than the old chunked path
(which timed out at N = 5×10⁵).

## Simulation wiring

`ForceConfig.method = "gpu_bh"` is handled in `ForceContext.accel_at_pos`, so
`Simulation` / leapfrog / tiered leapfrog all use the GPU walk with no special
driver. For large N, `total_energy` automatically returns kinetic energy only
(avoiding an O(N²) potential sum); use profile and structural diagnostics as
the primary equilibrium checks.

## Files

| Path | Role |
|------|------|
| `gpu_bh.py` | Public API, kernels, SoA upload, in-process launch |
| `_gpu_bh_worker.py` | Optional CuPy-first subprocess worker + accuracy check |
| `forces/context.py` | `ForceContext` dispatch for `"gpu_bh"` |
| `scripts/bench_gpu_bh.py` | Scaling benchmark harness |

## VRAM climb / reset-to-0 (leapfrog gates)

**What you see:** nvidia-smi VRAM climbs toward tens of GB during a
`gpu_bh` evolve, then drops to ~0 (or ComfyUI’s few hundred MiB) when the
Python process exits.

**Root cause (fixed in `compute_forces_gpu_bh`):** each leapfrog step rebuilds
the octree and allocates new CuPy buffers. Tree node count fluctuates, so
CuPy’s memory pool retained a bin for each prior size instead of returning
blocks to the driver. That is a **within-process pool leak** (not ComfyUI
fighting the job). Suite scripts also spawn a fresh
`evolve_component_slices.py` per gate — when that process dies, the CUDA
context is destroyed and VRAM resets (expected lifecycle).

**Monitoring:** correlate `nvidia-smi --query-compute-apps=pid,...` with the
evolve PID. Climb while one PID lives ⇒ pool/leak; drop when that PID exits
⇒ process lifecycle. ComfyUI (~300 MiB) is harmless background.

**Fix:** after each walk, drop device refs and call
`cupy.get_default_memory_pool().free_all_blocks()` (see `GpuBhState.detach`
and `_cupy_free_pooled_blocks`). Prefer `ForceContext(method="gpu_bh")` for
long runs; evolve scripts that call `compute_forces_gpu_bh` directly now get
the same free path.

## Known limitations

1. Tree build remains on the CPU (`BarnesHutTreeC`).
2. Warp divergence from independent walks — spatial sorting of targets would
   help further but is not required for correct O(N log N) scaling.
3. Subprocess mode cannot reuse `GpuBhState` across calls.

## References

[1] Barnes & Hut (1986), Nature 324, 446–449  
[2] NVIDIA CUDA C Programming Guide — Blackwell  
[3] CuPy RawModule documentation
