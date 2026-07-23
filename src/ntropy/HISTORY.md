# ntropy history

**ntropy** is a minimal self-gravitating N-body package for testing GalactICS
initial conditions. It is **not** a production cosmology or galaxy-evolution
code. It uses GalactICS internal units, JSON-driven runs, and Gadget-2-inspired
MPI domain decomposition without claiming Gadget fidelity.

## Design principles (persistent)

- Answer: *do equilibrium ICs stay approximately stable over self-gravitating evolution?*
- Collisionless gravity only (no gas/SPH).
- Configurable force backends: brute (reference), Python Barnes–Hut, C Barnes–Hut.
- Variable per-particle Plummer softening.
- Optional mpi4py parallel forces when OpenMPI is available.

## Chronology

| Era | Milestone |
|-----|-----------|
| **Origin** (`2e8ea1c`) | Minimal N-body package + notebook demo; brute forces; Plummer/NFW analytic ICs |
| **Integrators & MPI** (`dbeff2a`) | Leapfrog orders 1–2, Euler, RK2/3/4; Morton domain decomposition; MPI brute partial-target fix |
| **C Barnes–Hut** (`a21463e`) | `bh_c` extension (`bh_tree.c`), flat pack/bcast MPI path, `PARALLEL.md` roadmap |
| **Scaling & diagnostics** (`27e6dd6`) | Binney & Tremaine energy diagnostics; notebook scaling benchmarks (N, ranks, θ) |
| **GalactICS bridge** | `sample_galacticsics_*`, composite ICs, `nfw_halo_walkthrough.ipynb` |
| **Packaging** | mpi4py moved to optional extra; `install-python-deps` always reinstalls ntropy on `make install-dev` |
| **MW campaign** (current) | Integer particle types, tiered timestepping, tree persistence, OpenMP walk, DBH parameter grid |
| **C BH optimizations** (2026-07) | Optional `force.bh_optimizations` presets (`legacy` / `optimized`): fast softening math, squared opening test, iterative walk, Morton build, native MPI pack, fast `accel_all` |

### `force.bh_optimizations` (2026-07)

Backward-compatible performance flags for the ``bh_c`` backend. Default ``preset: legacy`` preserves
the original C kernels. ``preset: optimized`` enables:

| Flag | Effect |
|------|--------|
| `fast_inv_r3` | Replace `pow(r²+h², 1.5)` with `1/sqrt(r²+h²)³` |
| `squared_opening` | Opening criterion without `sqrt` |
| `iterative_walk` | Explicit stack traversal instead of recursion |
| `morton_build` | Morton-sorted particle insertion |
| `borrow_arrays` | Skip pos/mass/eps copy on build (Python holds refs) |
| `fast_coincident_check` | Squared distance for coincident-particle test |
| `native_pack` | MPI broadcast raw `BHNode` bytes instead of 19-float rows |
| `accel_all_fast` | Direct all-particle walk without index arange |
| `simd_leaves` | 4-wide unrolled leaf interaction loop |
| `omp_schedule` | `static` / `guided` / `dynamic` OpenMP scheduling |

Example JSON::

    "force": {
      "method": "bh_c",
      "theta": 0.5,
      "bh_optimizations": { "preset": "optimized" }
    }

## Performance evolution

See [../../docs/performance_roadmap.md](../../docs/performance_roadmap.md) for the
phased optimization plan (tree persistence, OpenMP, tiered dt, distributed tree).

## Related docs

- [README.md](README.md) — API and configuration
- [ntropy/forces/c/PARALLEL.md](ntropy/forces/c/PARALLEL.md) — MPI parallelization
- [../../docs/full_mw_roadmap.md](../../docs/full_mw_roadmap.md) — path to full MW particle counts
- [../../docs/ml_encoder_strategy.md](../../docs/ml_encoder_strategy.md) — learned encoder phased plan
