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

## Performance evolution

See [../../docs/performance_roadmap.md](../../docs/performance_roadmap.md) for the
phased optimization plan (tree persistence, OpenMP, tiered dt, distributed tree).

## Related docs

- [README.md](README.md) — API and configuration
- [ntropy/forces/c/PARALLEL.md](ntropy/forces/c/PARALLEL.md) — MPI parallelization
- [../../docs/full_mw_roadmap.md](../../docs/full_mw_roadmap.md) — path to full MW particle counts
- [../../docs/ml_encoder_strategy.md](../../docs/ml_encoder_strategy.md) — learned encoder phased plan
