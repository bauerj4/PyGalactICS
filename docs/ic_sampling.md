# IC sampling (OpenMP gen* and bulge equilibrium)

Particle sampling for disk, halo, and bulge is **embarrassingly parallel
rejection sampling**. By default the Python entry points dispatch to an OpenMP
C extension when it is built.

## Default backends

| Component | Default (`SampleConfig.use_openmp=True`) | Fallback |
|-----------|------------------------------------------|----------|
| Disk (`gendisk`) | OpenMP `_sampler_c.sample_disk` | Pure Python |
| Halo (`genhalo`) | OpenMP `_sampler_c.sample_halo` | Pure Python |
| Bulge (`genbulge`) | OpenMP `_sampler_c.sample_bulge` | Pure Python |

```python
from galacticsics.sampling.sampler import SampleConfig
from galacticsics.sampling.openmp import openmp_sampler_status

cfg = SampleConfig(n_disk=10_000, n_halo=10_000, n_bulge=10_000, use_openmp=True)
print(openmp_sampler_status(cfg).log_label())  # e.g. "OpenMP/all cores"
```

- `n_openmp_threads=0` → let OpenMP use all cores (`omp_set_num_threads` omitted).
- `use_openmp=False` → force the serial Python loops (tests / debugging).

### Build the extension

```bash
.venv/bin/python setup.py build_ext --inplace
# or: pip install -e ".[dev]"  (requires setuptools + gcc + OpenMP)
```

If the extension is missing, sampling warns and falls back to Python
(`warn_python_sampler_fallback`). Restart the Jupyter kernel after rebuilding.

## Bulge sampling details

Positions use a **spherical** proposal $(r,\mu,\phi)$ with weight $\propto r^2\rho(r)$
(not the legacy cylindrical $R$, $z=R\tan v$ can).

Velocities for the isotropic Eddington DF are drawn in **Cartesian**
$(v_x,v_y,v_z)$ inside the local escape sphere $|v|^2 < 2(\Psi-\Psi_c)$. Optional
`stream_bulge` then flips the **cylindrical** $v_\phi$ sign in the $(x,y)$ basis:

- `stream_bulge=0.5` → isotropic signs (no net rotation). **Default.**
- `stream_bulge=0` or `1` → fully retrograde / prograde streaming.

Rejection uses `fmax = max_{E\le\Psi}(f(E)-f_\mathrm{cut})` from a running max on
`dfsersic.dat`, not bare $f(\Psi)$. A flat high-$E$ DF floor previously made
`fmax=f(Ψ)` accept near-escape speeds in the inner bulge.

### DF table construction

`compute_sersic_df_table` Eddington-inverts analytic Sersic $\rho(\psi)$ via a
spline $d^2\rho/d\psi^2$ quadrature (`eddington_log_df_from_dens_psi`). The old
chain-rule `sersic_d2rho_dpsi2` path went negative near the centre and wrote a
tiny constant floor — do not revive that for sampling.

Poisson `denspsibulge.dat` is written from analytic $\rho(r(\psi))$, not the DF
round-trip (which underweights the bulge).

## Virial checks (important)

GalactICS ICs are drawn for the **fixed continuous** `dbh` potential $\Psi=-\Phi$
(up to a constant), not for softened pairwise self-gravity.

| Diagnostic | Use for ICs? | Notes |
|------------|--------------|-------|
| `virial_diagnostic_potential` (`galacticsics.diagnostics.virial`) | **Yes** | $W=\sum m_i\mathbf{x}_i\cdot\nabla\Psi$; expect $2K/\|W\|\approx 1$ |
| `ntropy.softening.virial_diagnostic` | Informational only | Softened $N$-body $W$; dense bulges look “too hot” when $\varepsilon$ is not tiny |

```python
from galacticsics.diagnostics import virial_diagnostic_potential

diag = virial_diagnostic_potential(pos, vel, mass, work_dir / "dbh.dat", rtol=0.25)
assert diag["is_virial_equilibrium"]
```

Notebook usage:

- [`notebooks/gpu_bh_dbh.ipynb`](../notebooks/gpu_bh_dbh.ipynb) — full DBH + GPU BH (§ IC validation)
- [`notebooks/nfw_halo_walkthrough.ipynb`](../notebooks/nfw_halo_walkthrough.ipynb) — halo-only potential virial (§2b)
- [`notebooks/campaign_density_walkthrough.ipynb`](../notebooks/campaign_density_walkthrough.ipynb) — MW campaign diagnostics

## Related tests

Essential PR gate: `pytest -m essential` (see [`ci_essential.md`](ci_essential.md)).

- `tests/test_bulge_ic.py` — densψ, morphology, DF floor, moments, potential virial
- `tests/test_sampler_openmp.py` — OpenMP disk / halo / bulge
