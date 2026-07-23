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

## P4 — MPI hot-path fixes (implemented, 2026-07)

Three separate problems produced *anti-scaling* (more ranks → slower) at
N ≲ 10⁴; all are fixed:

1. **Pickled tree broadcast.** MPI Barnes–Hut built the tree only on rank 0
   and broadcast a multi-MB *pickled* buffer on **every force call**; the
   other ranks sat idle during the serial build, then paid deserialization.
   At N=1024 this made 8 ranks ~15× slower than serial. The pickle path is
   gone entirely (see P5 for what replaced it).
2. **Redundant O(N²) work in MPI brute.** Every rank ran the full pairwise
   sum and then sliced its local targets — P ranks did ~P × N² work. Each
   rank now evaluates only its O(N_local × N) block.
3. **OpenMP oversubscription.** With `OMP_NUM_THREADS` unset, every rank
   spawned one walk thread per physical core (ranks × cores threads total).
   The `mpirun` harness (`ntropy.benchmark.mpi_subprocess`) now defaults
   `OMP_NUM_THREADS = cores / ranks` and the notebooks pin it explicitly per
   (ranks, threads) combination.

Accelerations are assembled with a buffer-based `Allgatherv`
(`MPI.DOUBLE`) — no pickling anywhere in the hot path. Worker stdout/stderr
can be teed to a log file via `log_path` on `run_mpirun_benchmark` /
`run_mpirun_simulation` (campaign evolve writes `evolve_mpirun.log`).

**Verified:** hybrid MPI × OpenMP sweep in `nfw_halo_walkthrough.ipynb` §4;
`src/ntropy/tests/test_parallel.py`.

## P5 — Gadget-style local trees + LET (implemented, 2026-07, default on)

`force.mpi_local_trees: true` (default). Per force evaluation, each rank:

1. Morton-sorts and takes its contiguous domain slice (Gadget-2 assignment).
2. Builds an octree over **only its domain particles** — no rank builds or
   stores the full O(N) tree.
3. Allgathers domain AABBs; the domain-root monopoles form the coarse
   global-tree skeleton.
4. Exports a **Local Essential Tree** to each peer: monopole nodes that pass
   `size / r_min < θ` against the peer's bounding box, opened down to leaf
   particles otherwise; payloads move with one `Alltoall`.
5. Folds the imported monopoles (as pseudo-particles) and leaf particles into
   one **combined force tree** with its own particles and evaluates targets in
   a single O(N/P · log N) walk; `Allgatherv` assembles the full acceleration
   array on every rank.

**Pitfall fixed (2026-07):** the first LET implementation applied imported
leaves as a *dense pairwise sum*. For centrally concentrated systems the
domain AABBs all overlap at the cusp, so the LET export degenerates to
mostly leaf particles (K ≈ N) and the pairwise application cost
O(N/P × N) per rank — near-brute-force work duplicated on every rank
(~2.5 s/force at N=10⁴ vs ~0.13 s after the combined-tree fix, and GB-scale
temporary arrays). The LET export walk was also vectorized (frontier BFS
over the packed node buffer) to remove a per-node Python loop.

**Follow-up (same week):** `BarnesHutTreeC.pack_buffers` leaked one Python
refcount per call (`Py_BuildValue` `"O"` vs `"N"` for newly created arrays).
LET called `pack_buffers` every force evaluation, so long notebook energy
runs ballooned RSS by hundreds of MB. Fixed in `bh_module.c`. For cuspy
halos where domain boxes overlap, the MPI path now *skips* LET export up
front and falls back to the replicated full-tree walk (same cost as
`mpi_local_trees=False`, without the pack/export overhead).

Peak tree memory drops from O(N) to O(N/P + |LET|) per rank.
`mpi_local_trees: false` restores the replicated full-tree walk (each rank
builds the identical tree locally — still no broadcast).

Wired through `ForceConfig`, `Simulation`, the campaign runner
(`force.mpi_local_trees` in the walkthrough config), and both walkthrough
notebooks. Details: `src/ntropy/ntropy/forces/c/PARALLEL.md`.

**Verified:** `test_local_essential_tree.py` (LET vs replicated force parity),
`test_mpi_local_trees_smoke.py` (galacticsics halo + disk/halo equilibrium
evolve under `mpirun`).

## P5b — Halo DF sampling fix (correctness, 2026-07)

Not an optimization but load-bearing for every benchmark above: `dfnfw.dat`
stores energies in **descending** order while the sampler interpolated with
`assume_sorted=True` (ascending), producing wrong velocities and ICs far from
equilibrium (2T/|W| ~ 1.87). After sorting the DF table correctly, sampled
halos are virialized (2T/|W| ~ 0.85). Stale pre-fix ICs were the main cause of
"strange" leapfrog energy curves. Regression: `tests/test_halo_df_interpolation.py`.

## P6 — MPI domain rebalance throttling (planned)

Morton sort + domain slice every N steps instead of every force call.

**Target:** N ~ 10⁵ multi-rank without sort-dominated overhead.

## P7 — Drop full particle replication (future)

Local trees already avoid replicating the *tree*; particle arrays
(`pos`/`mass`/`eps`) are still replicated on every rank. For N ≳ 10⁶, add
ghost exchange or ship leaf particle data inside the LET payload instead of
global indices. See `PARALLEL.md` phase 4.

## P8 — Gadget/GIZMO HDF5 export (future)

Delegate full-MW production runs to external codes; ntropy validates ICs and
short evolution.

## GPU evaluation (not implemented)

Candidates for future study:

- CUDA/OpenACC tree walk over targets
- Kokkos portable kernels
- Delegate to Gadget4/ChaNGa via FFI

**Criterion for GPU:** CPU path exhausted at N ≳ 10⁶ with P1–P5 complete.

Design note: `ForceBackend` protocol in `ntropy.forces` allows plugging alternate backends.

## Python DBH Poisson solver (galacticsics, implemented)

**Problem:** Coarse MW grid (`nr=4000`, `lmax=4`) spent ~3–5 min per solve; profiling showed
polar shell integration dominated (~88% of iteration time).

**Solution (Python):** Evaluate shell density once per radius (not per harmonic), batch
potential lookups, vectorize `appdiskdens` / `appdiskpot`, and use inline Simpson quadrature.

**Solution (C/OpenMP):** Optional extension `galacticsics.potential.poisson._poisson_c` parallelizes
the shell loop (`GALACTICSICS_POISSON_THREADS`; disable Python `n_workers` when using it).

**Benchmark** (`npsi=256`, `max_iter=20`, coarse MW): serial Python ~221s → optimized Python ~17s →
OpenMP (8 threads) ~2.4s. Build: `python setup.py build_ext --inplace` or `pip install -e .`.
See `docs/dbh_python_backend.md`.
