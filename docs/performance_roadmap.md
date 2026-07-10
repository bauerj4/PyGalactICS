# ntropy performance roadmap

This document tracks optimization phases for evolving Milky-Way-scale models
with ntropy. GPU offload is evaluated but **not implemented** in this campaign.

## Baseline (P0 — done)

Notebook `notebooks/nfw_halo_walkthrough.ipynb` §4 benchmarks at N=1024:

| Metric | Serial `bh_c` | MPI 4 ranks |
|--------|---------------|-------------|
| Force eval | ~12 ms | ~17 ms |
| Full leapfrog-2 step | ~42 ms | ~24 steps/s |
| 1 Gyr @ 2500 steps/Gyr | ~2 min | feasible |

**Crossover:** brute beats BH below N ~ 500–1000 (tree build dominates).

**Benchmark:**

```bash
pytest src/ntropy/tests/test_bh_c.py -v
python -m ntropy.benchmark.force_breakdown  # if available
```

## P1 — Tree persistence + buffer reuse (implemented)

**Problem:** `BarnesHutTreeC.build()` runs on every force evaluation (~2× per leapfrog step).

**Solution:** `Simulation` holds a `ForceContext` with a persistent tree; rebuild
every `force.rebuild_every` steps (default 1). Config: `force.rebuild_every`.

**Target:** N ~ 10⁴, reduced allocation churn.

## P2 — OpenMP target walk (implemented)

Parallelize the outer target loop in `bh_tree_accel_targets` (`bh_tree.c`).

**Env:** `OMP_NUM_THREADS` / `NTROPY_OMP_THREADS`. When using MPI, set
`OMP_NUM_THREADS = total_cores / n_ranks` to avoid oversubscription.

**Target:** N ~ 10⁴–10⁵ on multi-core nodes.

## P3 — Active-subset forces + tiered dt (implemented)

- Per-particle **dynamic** timestep bins from $\eta\sqrt{\varepsilon/|a|}$
  (GADGET-2), quantized to power-of-two multiples of ``dt_base``.
- Type-specific ``min_timestep_bin`` / ``max_timestep_bin`` bound the hierarchy;
  bins are not fixed by component type.
- ``target_indices`` in force APIs wired to tiered integrator active lists.

**Target:** N ~ 10⁵, 1 Gyr with acceptable ΔE/E₀.

## P4 — MPI domain rebalance throttling (planned)

Morton sort + domain slice every N steps instead of every force call.

**Target:** N ~ 10⁵ multi-rank without sort-dominated overhead.

## P5 — Distributed tree build (future)

Rank-0 build + `Bcast` dominates at N ≳ 10⁵. See `PARALLEL.md` phase 3.

## P6 — Ghost exchange / dual-tree walk (future)

Required for N ≳ 10⁶ without full tree replication per rank.

## P7 — Gadget/GIZMO HDF5 export (future)

Delegate full-MW production runs to external codes; ntropy validates ICs and
short evolution.

## GPU evaluation (not implemented)

Candidates for future study:

- CUDA/OpenACC tree walk over targets
- Kokkos portable kernels
- Delegate to Gadget4/ChaNGa via FFI

**Criterion for GPU:** CPU path exhausted at N ≳ 10⁶ with P1–P4 complete.

Design note: `ForceBackend` protocol in `ntropy.forces` allows plugging alternate backends.
