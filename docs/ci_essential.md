# Essential PR CI tests

Fast, high-signal checks that must pass before merging. Full suites and GPU /
legacy jobs stay on `push` to `main` / `master` and the weekly schedule.

## Run locally

```bash
# Rebuild OpenMP extensions if you changed C sources
.venv/bin/python setup.py build_ext --inplace

# Essential PR gate (same as GitHub Actions on pull_request)
make test-essential
# equivalent:
.venv/bin/pytest -m "essential and not legacy_binary and not slow" -v --tb=short
```

## What the essential gate covers

| Area | Why it is gated | Primary tests |
|------|-----------------|---------------|
| Bulge IC equilibrium | densψ amplitude, spherical morphology, DF floor, `<v²>` vs Eddington, potential virial | `tests/test_bulge_ic.py` |
| OpenMP sampling | disk / halo / bulge rejection paths stay wired | `tests/test_sampler_openmp.py` (non-`slow`) |
| Halo DF / softened virial smoke | NFW sampling + pairwise virial sanity | `tests/test_halo_df_interpolation.py` |
| Python physics smoke | solve writes `dbh`, bulge DF table, Sersic force | selected `tests/test_physics_python.py` |
| Numerics / units | Legendre & unit conversions | `tests/test_numerics_parity.py`, `tests/test_units.py`, `src/ntropy/tests/test_units.py` |
| ntropy core | BH-C, Plummer virial, leapfrog | `test_bh_c.py`, `test_virial_diagnostic.py`, `test_ics_plummer.py`, `test_integrator.py` |
| Softening large-N KE proxy | energy API contract | `test_softening_large_n.py` |

**Excluded from PR essential (still run elsewhere):**

| Marker / suite | When |
|----------------|------|
| `slow` | Full physics / OpenMP timing on `push` to default branch |
| `legacy_binary` | Weekly schedule only |
| GPU BH (`test_forces_gpu.py`) | Local / machine with CuPy + CUDA (see [`GPU_BLACKWELL.md`](../src/ntropy/ntropy/forces/GPU_BLACKWELL.md)) |
| Full `tests/` + `src/ntropy/tests/` | `push` to `main`/`master` |

## Marker definition

```toml
# pyproject.toml [tool.pytest.ini_options]
markers = [
  "essential: fast high-signal tests required on every PR",
  ...
]
```

Add `@pytest.mark.essential` (or module-level `pytestmark`) when a regression would
break the DBH → sample → evolve path documented in the usage notebooks
([`gpu_bh_dbh`](../notebooks/gpu_bh_dbh.ipynb),
[`nfw_halo_walkthrough`](../notebooks/nfw_halo_walkthrough.ipynb),
[`campaign_density_walkthrough`](../notebooks/campaign_density_walkthrough.ipynb)).

## Related docs

- [`galacticsics_pipeline.md`](galacticsics_pipeline.md) — algorithm + virial convention
- [`dbh_python_backend.md`](dbh_python_backend.md) — Poisson / sampler performance knobs
- [`ic_sampling.md`](ic_sampling.md) — OpenMP gen* + bulge DF / potential virial
- [`GPU_BLACKWELL.md`](../src/ntropy/ntropy/forces/GPU_BLACKWELL.md) — GPU BH usage
