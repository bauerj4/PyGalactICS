# Parallelizing the C Barnes–Hut tree

This document describes how the C implementation in `bh_tree.c` maps to MPI
domain decomposition in `ntropy.parallel.mpi`, and the steps to scale further.

## Current model (phase 1 — implemented)

**Replicated global tree, domain-decomposed walk** (Gadget-style assignment).

Per force evaluation / timestep:

1. Every rank holds the full particle arrays `pos`, `mass`, `eps` (sources).
2. **Rank 0** builds the octree in C (`BarnesHutTreeC.build` → `bh_tree_build`).
3. Rank 0 **packs** flat buffers (`pack_buffers`):
   - `nodes` — `(n_nodes, 19)` float64 rows (center, com, size, mass, leaf meta, children)
   - `leaf_indices` — int32 particle index pool
   - `pos`, `mass`, `eps` — particle arrays (already on all ranks)
4. **`MPI_Bcast`** the packed buffers (not Python objects).
5. Each rank calls `BarnesHutTreeC.from_packed` — **no rebuild**, only memcpy.
6. Morton sort + `domain_slices` → `local_targets` (unchanged Python helper).
7. Each rank calls `accel_targets(local_targets, theta)` in C.
8. `MPI_Allgather` local indices + accelerations → full `acc` array.

### Python / C API used at each step

| Step | Python call |
|------|-------------|
| Build | `BarnesHutTreeC.build(pos, mass, eps)` |
| Pack | `tree.pack_buffers()` |
| Unpack | `BarnesHutTreeC.from_packed(packed)` |
| Walk subset | `tree.accel_targets(local_targets, theta)` |
| Walk all | `tree.accel_all(theta)` |

### Wiring in ntropy (phase 1 — implemented)

In `ntropy.parallel.mpi.compute_forces_mpi`, when `method == "bh_c"`:

```python
if rank == 0:
    tree = BarnesHutTreeC.build(pos, mass, eps)
    packed = tree.pack_buffers()
else:
    packed = None
packed = comm.bcast(packed, root=0)
tree = BarnesHutTreeC.from_packed(packed)
local_acc = tree.accel_targets(local_targets, theta, pos=pos, eps=eps)
```

Use `ForceConfig.method = "bh_c"` to select this path.

## Phase 2 — OpenMP within a rank

Add `#pragma omp parallel for` over targets in `bh_tree_accel_targets` (or over
the outer target loop only). Build stays serial on rank 0.

**Caution:** when also using `mpirun -n R`, set OpenMP threads to avoid
oversubscription, e.g. `OMP_NUM_THREADS = total_cores / R`.

## Phase 3 — Parallel tree build (optional, large N)

When rank-0 build + `Bcast` dominates at `N > 10^5`:

1. **Global Morton sort** — `MPI_Allgather` particle count, sort keys in parallel
   (or distributed sort).
2. **Parallel radix build** — each rank inserts its subdomain particles into a
   shared flat node array with atomic / owner rules, or build subtrees and merge.
3. **PEPC-style** — exchange tree nodes along a space-filling curve.

This is not required for GalactICS IC testing at `N ~ 10^3–10^4`.

## Phase 4 — Distributed walk without full replication

For `N > 10^6`, walking may need **remote nodes** when a target's opening
criterion reaches a cell owned by another rank. Options:

- **Dual-tree** walk (local tree + interaction list)
- **Ghost cells** — halo exchange of boundary tree nodes each step
- Adopt a production code kernel (Gadget4, ChaNGa) via FFI

## Testing checklist

- [ ] `test_bh_c_matches_python_bh` — C vs `bhtree.py` accelerations
- [ ] `test_bh_c_pack_roundtrip` — build → pack → unpack → same accel
- [ ] `test_mpi_bh_c_matches_brute` — MPI `bh_c` vs brute at small N
- [ ] Benchmark §4b: C BH vs Python BH vs brute crossover

## Buffer layout (`NODE_PACK_WIDTH = 19`)

| Index | Field |
|-------|--------|
| 0–2 | `center[3]` |
| 3–5 | `com[3]` |
| 6 | `size` |
| 7 | `mass` |
| 8 | `is_leaf` |
| 9–16 | `child[8]` ( -1 if absent ) |
| 17 | `leaf_start` |
| 18 | `leaf_count` |
