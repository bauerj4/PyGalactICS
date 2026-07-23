# Full Milky Way evolution roadmap

Bridge from the **reduced-MW campaign** (N ~ 10⁵–10⁶, 1 Gyr in ntropy) to
production counts in `models/MilkyWay/` (3.5M disk + 1M halo).

## Scale tiers

| Tier | Particles | Evolution engine | DBH grid |
|------|-----------|------------------|----------|
| Test | ≤ 10³ | ntropy `bh_c` + MPI | `reference_disk_halo()` |
| Reduced MW | 10⁵–10⁶ | ntropy tiered dt + tree persistence | coarse `nr=2000–4000` screen → fine promote |
| Full MW | ~4.5M | Gadget/GIZMO (export) or future ntropy HPC | `milky_way_disk_halo()` `nr=20000` |

## When to stay in ntropy vs export

**Stay in ntropy when:**

- Validating IC stability (ρ drift, ΔE/E₀) over 0.1–1 Gyr
- Parameter-grid screening with N ≤ 10⁶
- Studying component-separated dynamics with typed particles

**Export to Gadget/GIZMO when:**

- N ≳ 10⁶ or full `models/MilkyWay/` particle files
- Need gas/hydro, star formation, or periodic/cosmological boundary conditions
- Production storage of frequent snapshots

## DBH solve cost management

Full MW solve (`nr=20000`, `dr=0.02`) takes minutes per model point.

1. **Coarse screen:** `nr=2000–4000`, `dr=0.05` for the parameter grid.
2. **Promote winners** to production grid only for selected models.
3. **Cache `h.dat`** when halo parameters are fixed; use `solve_halo_first` + `solve_baryons_in_fixed_halo` for baryon-only sweeps.
4. **Always re-run `diskdf`** when disk kinematics or baryonic potential changes.

## Campaign storage

Per-model directory `runs/mw_grid/{hash}/`:

- `model.json` — serialized `GalaxyModel`
- `dbh.dat`, `halo`, `disk`, `bulge` — IC artifacts
- `ntropy_config.json`, `evolution/` — simulation outputs
- `manifest.jsonl` row — diagnostics, wall times, ΔE/E₀, ρ drift

Snapshot cadence for 1 Gyr: every 100 Myr unless storage-constrained.

## GPU path (evaluation only)

See [performance_roadmap.md](performance_roadmap.md) P7. Full MW on GPU is out of
scope until CPU optimizations P1–P4 are measured at N ~ 10⁵–10⁶.

## Learned representations (optional)

For surrogate models and cross-run similarity, export particle tokens from
campaign checkpoints and encode with transformer or graph networks.  See
[ml_representation_roadmap.md](ml_representation_roadmap.md).  PyTorch is
optional (`pip install galacticsics[ml]`); feature export is numpy-only.

## Related

- [performance_roadmap.md](performance_roadmap.md)
- [ml_representation_roadmap.md](ml_representation_roadmap.md)
- [../src/ntropy/HISTORY.md](../src/ntropy/HISTORY.md)
- [../campaigns/mw_grid.json](../campaigns/mw_grid.json) — example grid spec
