# galacticsics

Python library for **GalactICS**-style galaxy initial conditions: self-consistent
multipole potentials, distribution-function corrections, particle sampling, and
observational fitting.

This repository separates the **modern Python API** (`src/galacticsics/`) from
the **original Fortran/C implementation** (`legacy/`), connected only through
documented subprocess runners and file formats.

## Repository layout

```
GalactICSIsoWithGas/
├── src/galacticsics/       # Python package (public API)
├── src/ntropy/             # Minimal N-body IC tester (separate package)
│   ├── models.py           # GalaxyModel, NFWHalo, ExponentialDisk, …
│   ├── potential/          # HarmonicPotential, evaluate, solve_potential
│   ├── distribution/       # Disk DF corrections, epicycle tables
│   ├── sampling/           # ParticleSet
│   ├── fitting/            # Rotation curve fit, optimizers, MCMC
│   ├── integrations/       # galpy wrapper
│   ├── legacy/             # Paths + subprocess runner (not legacy source)
│   └── io/                 # dbh.dat I/O, legacy stdin generators
├── legacy/                 # ISOLATED original code (do not import from Python)
│   ├── fortran/            # dbh.f, gendisk.c, Makefile, commonblocks
│   ├── bin/                # Compiled executables (dbh, gendisk, …)
│   └── python/             # Superseded Python 2 scripts + example data
├── models/MilkyWay/        # Reference Milky Way model + makefile workflow
├── examples/               # Runnable Python examples
├── notebooks/              # Tutorial notebooks (outputs → notebooks/artifacts/)
├── tests/                  # pytest (Milky Way + spherical fixtures)
├── docs/                   # Sphinx documentation
├── Makefile                # Top-level developer targets
└── pyproject.toml
```

## ntropy (N-body IC tester)

The sibling package **[`src/ntropy/`](src/ntropy/)** provides a minimal self-gravitating
N-body integrator for short stability checks on initial conditions. It reads the same
GalactICS ASCII particle format, supports brute-force and Barnes–Hut forces, variable
softening, JSON-driven runs, and MPI parallel forces (mpi4py). Installed automatically by
`make install-dev`. See [`src/ntropy/README.md`](src/ntropy/README.md) for full
documentation.

## Notebooks

Tutorial walkthroughs live in [`notebooks/`](notebooks/). Start with
[`nfw_halo_walkthrough.ipynb`](notebooks/nfw_halo_walkthrough.ipynb) — **GalactICS
IC generation** (`dbh` + `genhalo`) fed into **ntropy**; force accuracy and density
plots. Generated
figures and particle files go to `notebooks/artifacts/` (gitignored).

```bash
pip install jupyter
jupyter notebook notebooks/nfw_halo_walkthrough.ipynb
```

### Isolation principle

| Layer | Location | Role |
|-------|----------|------|
| **Python API** | `src/galacticsics/` | Typed models, SciPy numerics, tests, docs |
| **Legacy numerics** | `legacy/fortran/` | Unmodified graduate-school Fortran/C |
| **Bridge** | `galacticsics.legacy` | Writes `in.dbh`, runs `legacy/bin/dbh`, reads `dbh.dat` |

Python code never `#include`s or `import`s Fortran. New numerics use
`scipy.interpolate.CubicSpline`, `scipy.special.eval_legendre`, and
`scipy.integrate.simpson` instead of `splined.f`, `plgndr1.f`, and `simpson.c`.

## Units

### Internal (solver) units

All Fortran executables and Python APIs store values in **GalactICS units** with
`G = 1`:

| Quantity | Internal symbol | Physical value |
|----------|-----------------|----------------|
| Length | 1 | 1 kpc |
| Velocity | 1 | 100 km/s (`v=2.2` → 220 km/s) |
| Mass | 1 | 2.325×10⁹ M☉ |
| G | 1 | Natural units |

### Physical (user) units

Specify kpc, km/s, and M☉ at the API boundary; conversion is explicit:

```python
from galacticsics import GalaxyModel, UnitSystem, DEFAULT_UNITS
from galacticsics.units import mass_to_msun, velocity_to_kms

# Convenience constructor (defaults ≈ Milky Way disk+halo)
model = GalaxyModel.from_physical(
    disk_mass_msun=3.95e10,
    halo_v0_kms=370,
    halo_a_kpc=33,
    grid_nr=2000,          # internal grid still uses reduced nr for speed
)

# Manual conversion
mass_to_msun(model.disk.mass)   # → solar masses
velocity_to_kms(model.halo.v0)  # → km/s

# Custom unit system (advanced; default matches legacy code)
units = UnitSystem(velocity_kms=100.0, mass_msun=2.325e9, length_kpc=1.0)
```

See `galacticsics.units` for `mass_from_msun`, `velocity_from_kms`, `length_from_kpc`, etc.

## Prerequisites

- **Python** ≥ 3.10, `venv` (`sudo apt install python3-venv python3-pip`)
- **gfortran**, **gcc** (`sudo apt install gfortran gcc`)
- **make** (`sudo apt install make`) — or use `scripts/build_legacy_nmake.sh`

Optional: **galpy** (`pip install galacticsics[galpy]`)

## Installation

```bash
# Full developer install (venv + legacy build + generated test artifacts)
make install-dev

# Or step by step:
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e ".[dev]"
make legacy-build legacy-samplers
galacticsics-generate-artifacts generate   # → tests/generated/reference/

pytest tests/ -v
```

**Test artifacts** are **not** read from `models/MilkyWay/`. They are generated
fresh via `solve_potential` + `diskdf` + sampling, then checked with
`galacticsics-generate-artifacts verify`. Set `GALACTICSICS_ARTIFACT_DIR` to
override the output location.

## Makefile targets

From the repository root:

| Target | Description |
|--------|-------------|
| `make help` | List targets |
| `make install-dev` | Venv, `pip install -e ".[dev]"`, generate test artifacts |
| `make generate-artifacts` | Run `dbh` + `diskdf` + sampling → `tests/generated/reference/` |
| `make legacy-build` | Compile `legacy/fortran` → `legacy/bin/` |
| `make test` | Run pytest |
| `make example-mw` | Load Milky Way `dbh.dat`, print potential samples |
| `make example-solve` | Solve NFW halo via legacy `dbh` |
| `make example-sample` | Sample 500 disk+halo particles from Milky Way model |
| `make example-halo-first` | Two-step halo-first workflow demo |
| `make clean` | Remove object files and `.venv` |

## Examples

### Load precomputed Milky Way potential

```bash
make example-mw
# or
python examples/mw_default.py
```

```python
from galacticsics.builder import GalaxyBuilder
from galacticsics.models import GalaxyModel
from galacticsics.potential import evaluate_potential

builder = GalaxyBuilder(
    model=GalaxyModel.milky_way_disk_halo(),
    model_dir="models/MilkyWay",
).load_artifacts()

psi = evaluate_potential(builder.potential, s=8.0, z=0.0)
print(f"Psi(solar neighborhood) = {psi:.4f}")
```

### Solve a new potential (calls legacy dbh)

```bash
make example-solve
# or
python examples/solve_potential.py
```

```python
from galacticsics.models import GalaxyModel
from galacticsics.potential.solver import solve_potential

model = GalaxyModel.nfw_halo_only()
result = solve_potential(model, cleanup=False)
print("Tidal radius:", result.diagnostics.tidal_radius, "kpc")
print("Work dir:", result.diagnostics.work_dir)
```

### Two-step halo-first workflow

When the dark matter halo is specified by particles (e.g. `models/MilkyWay/Xhalo`)
or must be held fixed while disk/bulge/gas are added:

```bash
make example-halo-first
# or
python examples/halo_first_workflow.py
```

```python
from galacticsics.builder import GalaxyBuilder
from galacticsics.models import GalaxyModel

model = GalaxyModel.milky_way_disk_halo()
builder = GalaxyBuilder(model=model)

# Step 1: halo-only dbh → h.dat (isolated halo harmonics)
halo = builder.solve_halo_first(work_dir="runs/halo", cleanup=False)

# Step 2: baryons with halo fixed → merged dbh.dat + getfreqs
combined = builder.solve_baryons_in_fixed_halo(
    halo_work_dir=halo.work_dir,
    work_dir="runs/baryons",
)

# Sample disk; supply external halo particles instead of genhalo
builder.model_dir = str(combined.work_dir)
builder.sample(
    n_disk=10_000,
    n_halo=0,
    external_halo_path="models/MilkyWay/Xhalo",
)
```

Or use the one-shot helper:

```python
from galacticsics.potential import run_halo_first_workflow

result = run_halo_first_workflow(model, work_dir="runs/full", cleanup=False)
```

To fit NFW parameters from particles before step 1:

```python
from galacticsics.fitting import estimate_nfw_from_particles, apply_nfw_fit
from galacticsics.io import read_particles_ascii

data = read_particles_ascii("models/MilkyWay/Xhalo", max_particles=50_000)
fit = estimate_nfw_from_particles(data["x"], data["y"], data["z"], data["mass"])
model = apply_nfw_fit(GalaxyModel.milky_way_disk_halo(), fit)
```

### Classic makefile workflow (unchanged)

```bash
cd models/MilkyWay
make disk    # uses ../../legacy/bin/gendisk
```

The Milky Way `makefile` encodes the file dependency graph:

```
in.dbh  →  dbh  →  dbh.dat
dbh.dat + h.dat  →  getfreqs  →  freqdbh.dat
dbh.dat + freqdbh.dat + in.diskdf  →  diskdf  →  cordbh.dat
dbh.dat + cordbh.dat + in.disk  →  gendisk  →  disk
```

Note `freqdbh.dat` requires **both** `dbh.dat` (total potential) and `h.dat`
(halo-only harmonics). This is the hook for the two-step workflow.

## Python API overview

### Models

```python
from galacticsics import GalaxyModel, NFWHalo, ExponentialDisk

model = GalaxyModel(
    halo=NFWHalo(r_outer=200, v0=3.7, a=33, dr_trunc=12),
    disk=ExponentialDisk(
        mass=17, scale_length=2.5, outer_radius=20.25,
        scale_height=0.25, trunc_width=3.0,
    ),
)
```

Factory fixtures: `GalaxyModel.milky_way_disk_halo()`, `.nfw_halo_only()`,
`.sersic_bulge_only()`, `.nfw_plus_bulge()`.

### I/O

```python
from galacticsics.io import read_harmonic_potential, read_disk_correction

pot = read_harmonic_potential("models/MilkyWay/dbh.dat")
corr = read_disk_correction("models/MilkyWay/cordbh.dat")
```

### Potential evaluation (pure Python + SciPy)

```python
from galacticsics.potential import evaluate_potential, evaluate_force

psi = evaluate_potential(pot, s=8.0, z=0.0)
fr, fz, psi_total = evaluate_force(pot, s=8.0, z=0.0)
```

### galpy integration

```python
from galacticsics.integrations import GalactICSPotential

gp = GalactICSPotential.from_harmonic(pot).to_galpy()
```

---

## Legacy archive structure and reorganization

### Before (graduate-school layout)

The original GalactICS tree mixed Fortran sources, compiled binaries, Python 2
scripts, and model data in one working tree. Typical layout:

```
GalactICS/
├── *.f, *.c              # dbh.f, gendisk.c, diskdf.f, commonblocks, …
├── Makefile              # built dbh, gendisk, genhalo in-place
├── python/               # Python 2 helpers (generate_ics, plotting)
├── models/MilkyWay/      # in.dbh, dbh.dat, cordbh.dat, Xhalo, makefile
└── bin/ or ./dbh         # executables colocated with sources
```

Characteristics:

- **Monolithic build**: `make` in the Fortran directory produced `dbh`, `gendisk`,
  etc. next to `.f` files.
- **stdin-driven executables**: every program read interactive prompts from
  `in.dbh`, `in.disk`, `in.halo` (see `legacy/fortran/dbh.f` lines 16–127).
- **COMMON blocks**: shared state in `legacy/fortran/commonblocks` (`apot`,
  `fr`, `adens`, `gparameters`, …).
- **No Python 3 package**: orchestration was shell `make` plus Python 2 post-processing.

### After (this repository)

```
GalactICSIsoWithGas/
├── src/galacticsics/     # ONLY the Python 3 package
├── legacy/fortran/       # original sources moved here, unchanged numerics
├── legacy/bin/           # compiled executables (isolated from src/)
├── legacy/python/        # superseded Python 2 scripts (reference only)
└── models/MilkyWay/      # reference model; makefile points to ../../legacy/bin
```

**What changed**

| Item | Before | After |
|------|--------|-------|
| Fortran/C location | `src/` or repo root | `legacy/fortran/` |
| Python package | none / `python/` scripts | `src/galacticsics/` |
| Build output | beside sources | `legacy/bin/` |
| Python ↔ Fortran link | manual | `LegacyRunner` subprocess + file I/O |
| Interpolation / Legendre | `splined.f`, `plgndr1.f` | SciPy in `galacticsics.numerics` |
| Tests | informal | `pytest` + Milky Way fixtures |

**What did not change**

- File formats: `dbh.dat`, `h.dat`, `cordbh.dat`, `freqdbh.dat`, particle ASCII.
- Numerical algorithms in `dbh.f`, `diskdf.f`, `gendisk.c`, `genhalo.c`.
- Unit system (kpc, 100 km/s, GalactICS mass unit).
- `models/MilkyWay/` makefile target graph (paths updated to `legacy/bin`).

---

## Algorithms (mathematical detail)

### Coordinate system and units

GalactICS uses cylindrical coordinates \((s,\phi,z)\) with \(G=1\). Masses are in
\(M_\mathrm{unit} = 2.325\times 10^9\,M_\odot\). Velocities are in \(100\,\mathrm{km\,s^{-1}}\).
The potential \(\Psi\) is defined so that the circular velocity satisfies
\(v_c^2 = s\,\partial\Psi/\partial s\) on the midplane.

### Multipole Poisson solver (`dbh`)

The total mass density is expanded in even Legendre polynomials \(P_l(\cos\theta)\)
on a radial grid \(r_i = i\,\Delta r\), \(i=0\ldots N_r\):

\[
\rho(s,z) = \sum_{l=0}^{l_\mathrm{max}} \rho_l(r)\,P_l(\cos\theta),
\qquad r=\sqrt{s^2+z^2},\quad \cos\theta = z/r.
\]

Each iteration of `dbh.f`:

1. **Density harmonics** — for each \((l, r_i)\), integrate \(\rho(s,z)\,P_l(\cos\theta)\)
   over the meridional plane (Simpson in \(\cos\theta\); see lines 253–270 of `dbh.f`).

2. **Potential from density** — for each harmonic, apply the spherical Poisson
   solution (Binney & Tremaine eq. 2-208, implemented in `halopotentialestimate.f`
   and the main loop):

\[
\Phi_l(r) = \frac{4\pi}{2l+1}\left[
  \frac{1}{r^{l}}\int_0^r \rho_l(r')\,r'^{l+2}\,dr'
  + r^{l+1}\int_r^\infty \rho_l(r')\,\frac{dr'}{r'^{l-1}}
\right].
\]

The radial force harmonic follows from \(F_{r,l} = -\partial\Phi_l/\partial r\)
with the same split integrals (`s1`, `s2` arrays in `dbh.f`).

3. **Self-consistency** — the density \(\rho\) is evaluated in the *current*
   potential via DFs and analytic disk/gas approximations (`dens`, `totdens`,
   `polardens`). The tidal radius \(R_t\) is where \(\Psi(R_t,0)=\Psi_c\) on the
   midplane; iteration continues until \(|R_t^{(n)}-R_t^{(n-1)}| < \Delta r\).

4. **Harmonic ramp** — \(l_\mathrm{max}\) increases in steps of 2 after the
   monopole tidal radius stabilizes.

**Outputs in `dbh.dat`**: three blocks of coefficients on the same grid —
`adens[l/2+1, ir]`, `apot[l/2+1, ir]`, `fr[l/2+1, ir]`.

### Halo isolation (`halopotential.f` → `h.dat`)

After the full solve, if `ihaloflag=1`, subroutine `halopotential` recomputes the
Poisson harmonics of the **halo density alone** (NFW profile + Eddington DF) and
writes them to `h.dat`. The monopole at \(r=0\) is extrapolated quadratically;
higher multipoles are zero at the origin; each harmonic is shifted so \(\Phi_l\to 0\)
at \(r=R_\mathrm{edge}\).

`getfreqs.f` reads `dbh.dat` first (total model), then `h.dat` (halo only), and
tabulates epicyclic frequencies \(\Omega_h(s)\), \(\nu_h(s)\) along the major axis
and an inclined axis — needed by `diskdf` for the asymmetric drift integrals.

### NFW halo

Volume density (`halodensity`):

\[
\rho_\mathrm{NFW}(r) = \frac{\rho_0}{r/a\,(1+r/a)^2}
\quad\text{with}\quad
\rho_0 = \frac{2^{1-c}\,v_0^2}{4\pi a^2}.
\]

The distribution function is obtained by **Eddington inversion** on the spherical
potential \(\Psi(r)\):

\[
f(E) = \frac{1}{\sqrt{8\pi^2}}\left[
  \int_0^E \frac{d^2\rho}{d\Psi^2}\,\frac{d\Psi}{\sqrt{E-\Psi}}
  + \left.\frac{d\rho/d\Psi}{\sqrt{E-\Psi}}\right|_{\Psi=0}
\right],
\]

tabulated on an energy grid (`gendfnfw`, `denspsihalo.dat`, `dfnfw.dat`).

### Exponential stellar disk

Midplane surface density with erfc truncation:

\[
\Sigma(R) = \frac{M}{2\pi R_d^2}\,
\exp(-R/R_d)\cdot \tfrac{1}{2}\mathrm{erfc}\!\left(\frac{R-R_\mathrm{out}}{\Delta R}\right).
\]

Vertical profile:

\[
\rho(R,z) = \frac{\Sigma(R)}{2z_d}\,\mathrm{sech}^2(z/z_d).
\]

The disk contributes a **non-multipole** approximate potential (`appdiskpot.f`)
that is added in `pot.f` / `force.f` alongside the harmonic part.

### Disk distribution function (`diskdf`)

The action-based disk DF is corrected iteratively (`diskdf.f`):

1. Assume epicyclic approximation with \(\sigma_R(s)\) from `sigr2`.
2. For each radius, integrate over \((v_R, v_z, v_\phi)\) to match the imposed
   surface density and velocity dispersions.
3. Update spline correction factors `fdrat`, `fszrat` → written to `cordbh.dat`.

Toomre \(Q\) at \(R=2.5 R_d\):

\[
Q = \frac{\sigma_R}{\sigma_\mathrm{crit}},
\qquad
\sigma_\mathrm{crit} = \frac{3.36\,\Sigma}{ \kappa }
\]

where \(\kappa\) is the radial epicyclic frequency from `omekap`.

### Particle sampling

- **`genhalo`**: rejection sampling from \(f(E)\) with optional streaming fraction.
- **`gendisk`**: rejection in \((R,z,v_R,v_z,v_\phi)\) using `cordbh.dat` corrections.
- **`genbulge`**: spherical Sersic/NFW bulge from `dfsersic.dat`.

Particle format: ASCII lines `mass x y z vx vy vz` (GalactICS units).

---

## Two-step halo-first workflow (detailed)

### Motivation

In many science cases the halo is **not** generated by GalactICS:

1. Halo particles come from a cosmological simulation (`Xhalo`, 100k particles in
   `models/MilkyWay/`).
2. An axisymmetric NFW (or multipole) potential is **fit** to that particle
   distribution.
3. Disk, bulge, and gas are generated **in the fixed halo potential** without
   re-solving the halo in the joint Poisson iteration.

The legacy code always supported this via `h.dat` + the `in.dbh_disk` pattern
(halo stdin = `n`, disk stdin = `y`).

### Step 1 — Fit / solve the halo

**Option A — analytic NFW**: choose \((v_0, a, R_\mathrm{outer})\), run `dbh`
with only the halo enabled (`GalaxyModel.with_halo_only()`).

**Option B — particles**: center with `centre1.c` logic
(`galacticsics.fitting.center_particles_on_core`), bin \(M(<r)\), fit NFW scale
(`estimate_nfw_from_particles`), then run step A.

**Artifacts** (in `work_dir`):

| File | Content |
|------|---------|
| `dbh.dat` | Halo-only total multipole |
| `h.dat` | Isolated halo \(\Phi_l\), \(F_{r,l}\) (`halopotential.f`) |
| `denspsihalo.dat`, `dfnfw.dat` | Eddington DF tables |
| `mr.dat` | Integrated halo mass / scale radius |

### Step 2 — Baryons in fixed halo

1. Copy `h.dat` and DF aux files to the baryon run directory.
2. Run `dbh` with `ihaloflag=0` (disk/bulge/gas only) —
   `GalaxyModel.with_baryons_only()`.
3. **Merge harmonics** (Python: `merge_harmonic_potentials`):

\[
\Phi_l^\mathrm{tot} = \Phi_l^\mathrm{halo} + \Phi_l^\mathrm{baryon},
\quad
F_{r,l}^\mathrm{tot} = F_{r,l}^\mathrm{halo} + F_{r,l}^\mathrm{baryon}.
\]

4. Write merged `dbh.dat`; run `getfreqs` (needs both `dbh.dat` and `h.dat`).
5. Run `diskdf` → `cordbh.dat`; sample with `gendisk` / `genbulge`.

### Step 3 — Halo particles

Do **not** call `genhalo` when using external particles. Pass `external_halo_path`
to `GalaxyBuilder.sample()` or concatenate `Xhalo` with `disk` manually
(`cat disk bulge halo > galaxy` in the Milky Way makefile).

### Comparison with single-pass solve

| | Single-pass (`in.dbh` all `y`) | Halo-first (two-step) |
|--|-------------------------------|------------------------|
| Halo in Poisson iteration | yes, jointly with disk | step 1 only |
| `h.dat` | extracted post-solve | required input to step 2 |
| External halo particles | optional | primary use case |
| Self-consistent disk–halo coupling | full | approximate (fixed halo field) |

The Milky Way reference model (`models/MilkyWay/in.dbh`) uses the **single-pass**
mode for maximum self-consistency. Use halo-first when the halo is externally
imposed.

---

## Testing

```bash
pytest tests/ -v
```

| Test module | Coverage |
|-------------|----------|
| `test_io_milkyway.py` | `dbh.dat`, `cordbh.dat`, `mr.dat` round-trip |
| `test_potential_milkyway.py` | `evaluate_potential` / `evaluate_force` |
| `test_models_spherical.py` | NFW-only, bulge-only, combined fixtures |
| `test_sampling_milkyway.py` | Particle mass, kinematics statistics |
| `test_numerics_parity.py` | SciPy splines vs `cordbh.dat` nodes |
| `test_legacy_inputs.py` | `in.dbh` generation |
| `test_halo_first.py` | Halo-first helpers, `h.dat` merge, NFW particle fit |
| `test_sampling_legacy.py` | `gendisk` / `genhalo` integration |

## Documentation

```bash
pip install sphinx
cd docs && sphinx-build -b html . _build/html
```

## Roadmap

- [x] Phase 1: Package scaffold, I/O, tests
- [x] Phase 2: Python harmonic evaluator (SciPy Legendre)
- [x] Phase 3: `solve_potential()` via isolated legacy `dbh`
- [x] Phase 4: `DiskCorrectionTable`, `FrequencyTable` (CubicSpline)
- [x] Phase 5: Particle sampling wrappers (`gendisk`, `genhalo`, …)
- [x] Halo-first two-step workflow (`solve_halo_potential`, `h.dat` merge)
- [x] Phase 6: Rotation-curve fitting, optimizers, MCMC skeleton
- [x] Phase 7: galpy bridge, CI, docs scaffold

## Citation

Based on the GalactICS code of Kuijken & Dubinski for generating self-consistent
galaxy initial conditions. If you use this library in published work, cite the
original GalactICS papers and document the `galacticsics` version (`pip show galacticsics`).

## License

MIT (see package metadata). Legacy Fortran/C retains its original authorship and
academic-use expectations.
