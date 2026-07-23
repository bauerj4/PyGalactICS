# Parallelizing the C Barnes–Hut tree

This document describes how the C implementation in `bh_tree.c` maps to MPI
domain decomposition in `ntropy.parallel.mpi`, and the steps to scale further.

## Current model (default) — Gadget-style local trees + LET

**Local octree per rank + Local Essential Tree exchange** (`force.mpi_local_trees: true`).

Per force evaluation / timestep:

1. Every rank holds the full particle arrays `pos`, `mass`, `eps` (sources are
   still replicated; only the **tree** is domain-local).
2. Morton sort + `domain_slices` → `local_targets` (Gadget-2 assignment).
3. Each rank builds a C octree on **only its domain particles**
   (`BarnesHutTreeC.build(local_pos, …)`).
4. Ranks Allgather domain AABBs.  The Allgathered **domain-root monopoles**
   form the coarse **global tree** skeleton.
5. Each rank walks its local tree and exports a **Local Essential Tree (LET)**
   to every peer: monopole nodes that satisfy `size / r_min < θ` for the peer
   AABB, otherwise opened down to leaf particles (global indices).
6. `MPI_Alltoall` of LET payloads; each rank builds one **combined force
   tree** over `[local particles, imported leaves, imported monopoles as
   pseudo-particles]` and evaluates its targets in a single O(n_local log N)
   walk. (Never applied as a dense pairwise sum: overlapping domain AABBs —
   the norm for centrally concentrated systems — make the LET export mostly
   leaves, and pairwise application would duplicate near-brute-force work on
   every rank.)
7. `MPI_Allgatherv` of local accelerations → full `acc` on every rank.

No rank stores or walks a full `O(N)` octree.  Peak tree memory is
`O(N / P + |LET|)`.

Config:

```json
{ "force": { "method": "bh_c", "theta": 0.5, "mpi_local_trees": true } }
```

Set `"mpi_local_trees": false` to restore the replicated full-tree path below.

## Fallback — replicated global tree, domain-decomposed walk

Used when `force.mpi_local_trees` is `false`.

1. Every rank builds the identical full octree locally from the replicated
   particle arrays (no pickled broadcast).
2. Each rank walks only its `local_targets`.
3. `MPI_Allgatherv` assembles accelerations.

### Python / C API used at each step

| Step | Python call |
|------|-------------|
| Build | `BarnesHutTreeC.build(pos, mass, eps)` |
| Pack | `tree.pack_buffers()` (LET export / diagnostics) |
| Unpack | `BarnesHutTreeC.from_packed(packed)` |
| Walk subset | `tree.accel_targets(local_targets, theta)` |
| Walk all (local) | `tree.accel_all(theta)` |

## Phase 2 — OpenMP within a rank (implemented)

`bh_tree_accel_targets` and `bh_tree_accel_all` use OpenMP over targets.
Schedule is configurable via `force.bh_optimizations.omp_schedule`.

**Caution:** when also using `mpirun -n R`, set OpenMP threads to avoid
oversubscription, e.g. `OMP_NUM_THREADS = total_cores / R`.

## Phase 2b — Optional C walk/build optimizations (2026-07)

Controlled by `force.bh_optimizations` in JSON config (`preset: legacy|optimized`).
See `ntropy/HISTORY.md` for the full flag table. Native pack format:

| `pack_format` | Broadcast payload |
|---------------|-------------------|
| `0` (legacy) | `nodes` `(n_nodes, 19)` float64 rows |
| `1` (native) | `nodes_native` `uint8` blob, `sizeof(BHNode)` per node |

`meta` is `(n_nodes, n_leaf_indices, pack_format)`.

## Phase 3 — Parallel tree build (optional, large N)

When local build still dominates at `N > 10^5` per rank:

1. **Global Morton sort** — distributed sort of keys.
2. **Parallel radix build** — each rank inserts its subdomain particles with
   atomic / owner rules, or build subtrees and merge.
3. **PEPC-style** — exchange tree nodes along a space-filling curve.

## Phase 4 — Drop full particle replication

For `N > 10^6`, also avoid replicating `pos`/`mass`/`eps`.  Options:

- **Ghost cells** — halo exchange of boundary particles each step
- Ship leaf particle data inside the LET instead of global indices
- Adopt a production code kernel (Gadget4, ChaNGa) via FFI

## Testing checklist

- [x] `test_bh_c_matches_python_bh` — C vs `bhtree.py` accelerations
- [x] `test_bh_c_pack_roundtrip` — build → pack → unpack → same accel
- [x] `test_mpi_bh_c_matches_brute` — MPI `bh_c` vs brute at small N
- [x] `test_mpi_local_trees_matches_replicated` — LET path vs full tree
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
