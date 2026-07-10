# Python DBH backend

The Python Poisson solver (`galacticsics.potential.poisson`) replaces legacy Fortran
`dbh` for supported component combinations (halo + optional disk and/or bulge).

## Approximate disk potential subtraction

During harmonic density synthesis (`total_density_harmonic` in `densities.py`), the
solver evaluates:

```
rho_harmonic = rho_halo(psi) + rho_bulge(psi) + diskdens(psi) - appdiskdens(s, z)
```

This matches legacy `polardens.f` / `totdens.f`.

**Why subtract `appdiskdens`?** The approximate disk potential (`appdiskpot`) is
already included in the multipole potential sum during iteration. The self-consistent
disk density from the solved potential (`diskdens`) would otherwise double-count the
disk: once through the approximate potential harmonics and again through `diskdens`.
Subtracting `appdiskdens` leaves multipoles that represent departures of the true disk
from the softened `log(cosh)` vertical ansatz.

**Important:** Monopole seeding for the Poisson iteration uses a *different* density
estimate: `diskdensestimate` integrated by `diskpotentialestimate` (see
`appdisk.disk_densestimate` and `appdisk.integrated_disk_densestimate_on_shell`).
Do not use `appdiskdens` for monopole estimates — that was a common source of
~22% shallow potentials (`psi0 ≈ 13.8` vs literature `≈ 18.1`).

## No legacy fallback on the Python path

`python_ensure_disk_df` always runs the Python `diskdf` port. If `cordbh.dat` is
invalid, the run fails with `RuntimeError` rather than silently calling Fortran
`diskdf`. Use `physics_backend=legacy` explicitly when Fortran numerics are required.

## Diagnostic tests

Long stability and literature-consistency checks live in `tests/test_dbh_diagnostics.py`:

```bash
pytest tests/test_dbh_diagnostics.py -m slow -v
```

Quick parity tests (psi0, frequency sanity, appdisk subtraction) run in the default
`physics_python` subset without the `slow` marker.

`test_python_solved_density_stable_100_myr` evolves Python-solved disk+halo ICs for
≈100 Myr with `bh_c` gravity (theta=0.5, optimized preset), tiered leapfrog integration,
and asserts disk Σ(R) drift < 55% and halo ρ(r) drift < 75% (coarse grid, 8k particles).

## Performance knobs

| Knob | Location | Effect |
|------|----------|--------|
| `n_workers` | `solve_potential(..., n_workers=N)` | Thread-pool size for polar shell integration during each Poisson iteration |
| `GALACTICSICS_SOLVE_WORKERS` | environment | Default worker count when `n_workers` is omitted (`0` = serial) |
| Coarse grid auto-scaling | `solve.py` | Reduces polar nodes, DF table size, and iteration cap when `grid.nr` is small |
| `scripts/benchmark_solve.py` | CLI | Times solve + optional `freqdbh` retabulation |

Shell integration uses vectorized NumPy density evaluation (`halo_density_spherical_array`,
`disk_density_psi_batch`, tabulated `rho(psi)` lookup) instead of Python loops or
`np.vectorize`.

## Solver variable glossary

Legacy Fortran names in `dbh.dat` are preserved on :class:`~galacticsics.potential.harmonics.HarmonicPotential`.
The Python solver uses descriptive local names mapped as follows:

| Legacy / file name | Descriptive name | Meaning |
|--------------------|------------------|---------|
| `psi0` | `reference_potential` | Potential at the origin [100 km/s]² |
| `psic` | `cutoff_potential` | DF / tidal cutoff energy |
| `psid` | `inner_potential` | Innermost DF tabulation energy |
| `haloconst` | `halo_normalization` | NFW amplitude ρ₀ from v₀, a, cusp |
| `bulgeconst` | `bulge_normalization` | Sersic bulge normalization |
| `lmax` / `lmaxx` | `max_harmonic_degree` | Maximum even spherical-harmonic degree |
| `lmax_active` | `active_lmax` | Degree used in the current iteration (harmonic ramp) |
| `apot` | `potential_harmonics` | Potential multipole coefficients |
| `adens` | `density_harmonics` | Density multipole coefficients |
| `fr` | `force_harmonics` | Radial force harmonics |
| `fr2` | `force_second_harmonics` | Second radial derivatives of force harmonics |
| `hpot`, `dpot`, `bpot` | `halo/disk/bulge_monopole_potential` | Component monopole seeds |
| `rtidal` | `tidal_radius` | Radius where total potential crosses `psic` |
| `drtidal` | `tidal_radius_change` | Change in tidal radius between iterations |
| `frac` | `potential_relaxation_factor` | Under-relaxation (default 0.75) for new harmonics |
| `dr`, `nr` | `radial_step`, `n_radial_shells` | Radial grid spacing and shell count |
| `ntheta` | `n_polar_nodes` | Polar quadrature nodes on cos θ ∈ [0, 1] |
