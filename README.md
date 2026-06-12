# galacticsics

Python library for **GalactICS**-style galaxy initial conditions: self-consistent
multipole potentials, distribution-function corrections, particle sampling, and
observational fitting.

This repository separates the **modern Python API** (`src/galacticsics/`) from
the **original Fortran/C implementation** (`legacy/`), connected only through
documented subprocess runners and file formats.

A sibling package **`ntropy`** (`src/ntropy/`) evolves those initial conditions
with a short self-gravitating N-body test (brute, Python/C Barnes–Hut, MPI,
multiple integrators). The notebook
[`nfw_halo_walkthrough.ipynb`](notebooks/nfw_halo_walkthrough.ipynb) demonstrates
the full **GalactICS IC → ntropy stability** pipeline.

## Repository layout

```
GalactICSIsoWithGas/
├── src/galacticsics/       # Python package (public API)
├── src/ntropy/             # N-body IC stability tester (separate package)
│   ├── ntropy/forces/c/    # C Barnes–Hut octree (_bh_c extension)
│   ├── ntropy/benchmark/   # MPI subprocess workers, timing breakdowns
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

## ntropy (N-body IC stability tester)

The sibling package **[`src/ntropy/`](src/ntropy/)** is a self-gravitating N-body
integrator for **short equilibrium checks** on GalactICS (or analytic) initial
conditions. It answers: *does ρ(r) and |ΔE/E₀| stay reasonable over a few Gyr of
evolution?* It is not a production cosmology code.

Installed automatically by `make install-dev` (compiles the optional C extension).
Package-level API docs: [`src/ntropy/README.md`](src/ntropy/README.md).

### Capabilities (current)

| Area | Details |
|------|---------|
| **Particle I/O** | GalactICS ASCII (`mass x y z vx vy vz`); optional `nobj flag` header |
| **Softening** | Per-particle Plummer; pairwise \(h_{ij} = \tfrac{1}{2}(\varepsilon_i+\varepsilon_j)\) |
| **Forces** | `brute` (vectorized O(N²)), `bh` (pure-Python octree), **`bh_c`** (C octree) |
| **Integrators** | Leapfrog orders 1–2 (symplectic); Euler, RK2, RK3, RK4 (explicit, non-symplectic) |
| **Parallelism** | mpi4py Morton (Z-order) domain decomposition; Gadget-2-style assignment |
| **MPI BH** | Python: `bcast` tree object; **C**: rank-0 `pack_buffers` → flat `(n_nodes, 19)` broadcast → `from_packed` |
| **Config** | JSON runs (`ntropy-run`); default **5 Gyr** at **1000 steps/Gyr** (Δt ≈ 0.102 code units ≈ 1 Myr) |
| **IC models** | Plummer, truncated NFW, Sersic bulge, exponential disk, composites |
| **galacticsics bridge** | `sample_galacticsics_halo` / `sample_galacticsics_galaxy` → `Simulation` |
| **Analysis** | Spherical ρ(r), disk Σ(R), energy drift, density stability assertions |

### Force backends

**Brute force** (`force.method: "brute"`) — NumPy pairwise kernel; exact reference for
small N and validation.

**Python Barnes–Hut** (`force.method: "bh"`) — Monopole octree in pure Python; rebuilt
every force call. Accurate but slow at notebook scales (timed up to N = 4096 in §4); useful for debugging.

**C Barnes–Hut** (`force.method: "bh_c"`) — Same physics as Python BH, implemented in
`src/ntropy/ntropy/forces/c/bh_tree.c` and exposed via `bhtree_c.py`. Built on
`pip install -e src/ntropy` (requires gcc + NumPy headers). MPI path documented in
[`forces/c/PARALLEL.md`](src/ntropy/ntropy/forces/c/PARALLEL.md).

```python
from ntropy.forces.bhtree_c import BarnesHutTreeC, compute_forces_bh_c, extension_available

if extension_available():
    tree = BarnesHutTreeC.build(pos, mass, eps)
    acc = tree.accel_all(theta=0.5)
    packed = tree.pack_buffers()  # MPI broadcast payload
```

### Time units

GalactICS code time: **1 unit ≈ 9.78 Myr ≈ 0.0098 Gyr** (1 kpc / (100 km/s)).
Defaults: `dt ≈ 0.102`, `n_steps = 5000` → **5 Gyr** total. Helpers in
`ntropy.units` (`code_time_to_gyr`, `format_simulation_duration`, etc.).

### MPI usage

```bash
# Single-rank (falls back to serial kernels)
ntropy-run src/ntropy/configs/plummer_short.json

# Multi-rank domain decomposition
mpirun -n 4 ntropy-run src/ntropy/configs/plummer_short.json
```

`make install-dev` attempts to install OpenMPI (`scripts/ensure_openmpi.sh`) and
rebuilds mpi4py. Launch simulations under `mpirun` for parallel force evaluation;
progress and |ΔE/E₀| print on rank 0 when using the MPI simulation worker.

### Integrated GalactICS → ntropy workflow

```python
from ntropy.config import ForceConfig, IntegratorConfig, ParallelConfig, RunConfig
from ntropy.integrations.galacticsics import sample_galacticsics_halo, particle_state_from_galacticsics
from ntropy.simulation import Simulation

cfg = RunConfig(
    force=ForceConfig(method="bh_c", theta=0.5),
    integrator=IntegratorConfig(type="leapfrog", order=2),
    parallel=ParallelConfig(enabled=True, n_workers=8),
)
ic = sample_galacticsics_halo(n_particles=1024, seed=42)
result = Simulation(cfg, state=ic.state).run(show_progress=True)
```

## Notebooks

Tutorial walkthroughs live in [`notebooks/`](notebooks/). Generated figures and
particle files go to `notebooks/artifacts/` (gitignored).

### [`nfw_halo_walkthrough.ipynb`](notebooks/nfw_halo_walkthrough.ipynb)

End-to-end **GalactICS IC → ntropy validation** for a spherical NFW halo:

| Section | Content |
|---------|---------|
| **§1** | `GalaxyModel` → legacy `dbh` → `genhalo`; typed Python API |
| **§2** | Initial ρ(r) vs analytic NFW; sanity check on sampled IC |
| **§3** | Brute vs Barnes–Hut force accuracy vs opening angle θ |
| **§4** | Serial/MPI/integrator scaling: brute vs `bh_c` (N and ranks 1–8; leapfrog order 1 vs 2) |
| **§4b** | Why BH C can lose to brute at N=1024; MPI overhead; brute partial-sum fix |
| **§4c** | **Python vs C BH** — accuracy check and build/walk timing (`bh` vs `bh_c`) at N=1024 |
| **§4d** | **Large-N crossover** — force-only timings at N=10⁴ and 10⁵ (C BH vs brute) |
| **§4e** | **θ sweep** — BH build/walk timing and force accuracy vs opening angle θ |
| **§5** | 5 Gyr stability run (N=1024, `bh_c`, mpirun -n 4); ρ(r) drift and \|ΔE/E₀\| |
| **§5b** | Symplectic leapfrog vs explicit Euler/RK energy drift comparison |

Default notebook parameters: **N = 1024**, **Δt ≈ 0.041** code units (~0.4 Myr/step),
**12 500 steps** (5 Gyr at 2500 steps/Gyr), energy runs via `mpirun -n 4`.

```bash
pip install jupyter matplotlib tqdm
source .venv/bin/activate
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

pytest tests/ src/ntropy/tests/ -v
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
| `make install-dev` | Venv, galacticsics + ntropy (incl. mpi4py, **C BH extension**), OpenMPI if available, generate test artifacts |
| `make install-system-mpi` | Install OpenMPI packages (`scripts/ensure_openmpi.sh`) |
| `make generate-artifacts` | Run `dbh` + `diskdf` + sampling → `tests/generated/reference/` |
| `make legacy-build` | Compile `legacy/fortran` → `legacy/bin/` |
| `make test` | Run `pytest tests/ src/ntropy/tests/` (full galacticsics + ntropy suite) |
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

Run the **full suite** (galacticsics + ntropy) from the repository root:

```bash
make test
# equivalent to:
pytest tests/ src/ntropy/tests/ -v --tb=short
```

**Prerequisites:** `make install-dev` (venv, legacy binaries, generated reference
artifacts in `tests/generated/reference/`). Some ntropy tests require **mpi4py**
and **mpirun**; C BH tests require the compiled `_bh_c` extension (skipped if
missing). MPI multi-rank tests are skipped when `mpirun` is unavailable.

---

### galacticsics tests (`tests/`)

Generated Milky Way reference artifacts drive most integration tests. Regenerate with
`make generate-artifacts` if hashes drift after model changes.

#### `test_io_milkyway.py` — harmonic potential I/O

| Test | What it checks |
|------|----------------|
| `test_read_harmonic_potential_reference` | `dbh.dat` parses; grid dimensions and coefficient arrays load |
| `test_harmonic_round_trip` | Write/read round-trip preserves multipole coefficients |
| `test_read_component_masses` | `mr.dat` component masses match manifest |
| `test_read_disk_correction` | `cordbh.dat` spline correction factors load |
| `test_read_frequency_table` | `freqdbh.dat` epicyclic frequency tables load |
| `test_read_rtidal_and_toomre` | Tidal radius and Toomre Q metadata consistent with model |
| `test_artifact_consistency` | Cross-file consistency (grid, l_max, nr) across artifact set |
| `test_manifest_matches_model` | Generated manifest matches `GalaxyModel.milky_way_disk_halo()` parameters |

#### `test_potential_milkyway.py` — Python potential evaluator

| Test | What it checks |
|------|----------------|
| `test_potential_at_solar_neighborhood` | Ψ(s=8, z=0) in expected range for MW model |
| `test_force_midplane` | Midplane force components finite and physically signed |
| `test_circular_velocity_scale` | v_c from force matches expected km/s scale at solar radius |
| `test_potential_force_consistency` | Numerical ∂Ψ/∂s matches returned force component |

#### `test_models_spherical.py` — model fixtures

| Test | What it checks |
|------|----------------|
| `test_nfw_halo_only_fixture` | `GalaxyModel.nfw_halo_only()` enables halo, disables disk/bulge |
| `test_sersic_bulge_only_fixture` | Bulge-only model flags and parameters |
| `test_nfw_plus_bulge_fixture` | Combined halo+bulge model construction |

#### `test_sampling_milkyway.py` — particle sampling statistics

| Test | What it checks |
|------|----------------|
| `test_disk_particle_mass` | Sampled disk particle masses sum to model disk mass |
| `test_halo_particle_com_near_origin` | Halo COM near origin after sampling |
| `test_disk_velocity_dispersion_order_of_magnitude` | σ_R, σ_z order-of-magnitude vs epicyclic expectations |

#### `test_numerics_parity.py` — SciPy vs legacy numerics

| Test | What it checks |
|------|----------------|
| `test_cubic_spline_matches_cordbh_nodes` | `CubicSpline` matches `cordbh.dat` node values |
| `test_legendre_at_mu_one` | SciPy Legendre at μ=1 matches Fortran convention |
| `test_simpson_integrate_polynomial` | Simpson integrator exact on low-order polynomials |

#### `test_legacy_inputs.py` — stdin file generation

| Test | What it checks |
|------|----------------|
| `test_write_dbh_input_milky_way` | Generated `in.dbh` matches Milky Way reference prompts |
| `test_write_gendenspsi` | Halo density/potential aux input files well-formed |

#### `test_halo_first.py` — two-step halo workflow

| Test | What it checks |
|------|----------------|
| `test_model_halo_only_disables_baryons` | `with_halo_only()` disables disk/bulge/gas flags |
| `test_model_baryons_only_disables_halo` | `with_baryons_only()` disables halo solve |
| `test_read_halo_harmonics_generated` | `h.dat` from generated halo run parses |
| `test_merge_harmonics_generated` | Merged baryon+halo harmonics sum correctly |
| `test_estimate_nfw_from_synthetic_particles` | NFW fit recovers scale radius from synthetic profile |
| `test_from_physical_matches_reference` | `GalaxyModel.from_physical` matches reference units |

#### `test_builder_milkyway.py` — artifact loading

| Test | What it checks |
|------|----------------|
| `test_load_reference_artifacts` | `GalaxyBuilder` loads generated `dbh.dat`, corrections, frequencies |
| `test_load_reference_particles` | Reference disk/halo particle files load with expected counts |

#### `test_sampling_legacy.py` — legacy executable integration

| Test | What it checks |
|------|----------------|
| `test_sample_reference_small` | `gendisk`/`genhalo` subprocess sampling produces valid particle file |

#### `test_units.py` — unit conversions

| Test | What it checks |
|------|----------------|
| `test_default_unit_system` | Default `UnitSystem` matches GalactICS conventions |
| `test_mass_velocity_conversions` | M☉ ↔ code mass, km/s ↔ code velocity |
| `test_length_conversions` | kpc ↔ code length |
| `test_from_physical_round_trip` | Physical → internal → physical round-trip |
| `test_custom_unit_system` | Custom `UnitSystem` scaling |

---

### ntropy tests (`src/ntropy/tests/`)

#### `test_forces.py` — force kernel agreement

| Test | What it checks |
|------|----------------|
| `test_brute_bh_agreement` | Python BH accelerations match brute at θ=0.3 (rtol≈15%) |
| `test_bh_handles_duplicate_positions` | Octree does not overflow when two particles share coordinates |
| `test_brute_targets_matches_full` | Partial brute sum on target indices matches full-array result |
| `test_variable_softening_symmetry` | Asymmetric ε_i, ε_j still yields equal-and-opposite pairwise force |

#### `test_bh_c.py` — C Barnes–Hut extension (skipped if `_bh_c` not built)

| Test | What it checks |
|------|----------------|
| `test_bh_c_matches_python_bh` | C and Python BH accelerations agree to machine precision |
| `test_bh_c_matches_brute_small` | C BH matches brute at small N (θ=0.3, rtol≈15%) |
| `test_bh_c_targets_subset` | `accel_targets` on index subset matches rows of `accel_all` |
| `test_bh_c_pack_roundtrip` | build → `pack_buffers` → `from_packed` preserves accelerations |

#### `test_softening.py` — Plummer kernel

| Test | What it checks |
|------|----------------|
| `test_pairwise_symmetry` | h_ij = h_ji for pairwise softening matrix |
| `test_acceleration_at_large_distance` | Softened force → Newtonian at r ≫ ε |

#### `test_integrator.py` — leapfrog

| Test | What it checks |
|------|----------------|
| `test_leapfrog1_step_matches_symplectic_euler` | Order-1 leapfrog matches symplectic Euler kick-drift |
| `test_leapfrog2_step_is_velocity_verlet_half_kick` | Order-2 leapfrog matches velocity-Verlet half-kick |
| `test_two_body_symplectic_energy_preserved` | Softened two-body over **10 Gyr**: circular and mildly eccentric (e=0.1) orbits; leapfrog orders 1–2 |
| `test_plummer_energy_drift_bounded` | Plummer sphere: \|ΔE/E₀\| bounded over 50 leapfrog steps |

#### `test_explicit_integrators.py` — Euler and Runge–Kutta

| Test | What it checks |
|------|----------------|
| `test_euler_step_forward` | Forward Euler advances linear motion correctly |
| `test_rk_linear_oscillator_converges_with_order` | RK2/RK3/RK4 convergence orders on harmonic oscillator |
| `test_integrator_metadata` | Registry exposes expected integrator names |
| `test_euler_energy_drift_exceeds_symplectic_leapfrog` | Non-symplectic Euler drifts more than leapfrog on Plummer |

#### `test_config.py` — JSON schema

| Test | What it checks |
|------|----------------|
| `test_load_minimal_config` | Minimal valid JSON loads; defaults for force/integrator |
| `test_invalid_force_method` | Rejects unknown `force.method` (accepts `brute`, `bh`, `bh_c`) |
| `test_load_rk4_integrator` | RK4 integrator type loads from JSON |
| `test_invalid_integrator_type` | Rejects unknown integrator enum |
| `test_missing_particles_file` | Missing `particles.file` raises clear error |

#### `test_units.py` — simulation time constants

| Test | What it checks |
|------|----------------|
| `test_default_simulation_span_is_five_gyr` | Default n_steps × dt spans 5 Gyr |
| `test_dt_is_one_myr_per_step` | Default dt ≈ 1 Myr in physical units |
| `test_code_time_per_gyr` | Gyr ↔ code time conversion constants |
| `test_format_simulation_duration_mentions_gyr` | Human-readable duration string includes Gyr |

#### `test_io.py` — particle file I/O

| Test | What it checks |
|------|----------------|
| `test_roundtrip_ascii` | Write/read ASCII preserves pos, vel, mass |
| `test_skip_header_line` | `nobj flag` header line skipped on read |
| `test_particle_state_write` | `ParticleState.write_ascii` format compatible with reader |

#### `test_ics_plummer.py` — Plummer IC generator

| Test | What it checks |
|------|----------------|
| `test_abel_matches_analytic_plummer_df` | Abel-inversion DF matches analytic Plummer f(E) |
| `test_plummer_virial_ratio` | Sampled Plummer sphere 2T + \|W\| ≈ 0 (virial equilibrium) |

#### `test_disk_density.py` — exponential disk IC

| Test | What it checks |
|------|----------------|
| `test_exponential_disk_volume_density_matches_legacy` | ρ(R,z) matches legacy GalactICS erfc-truncated formula |
| `test_sampled_disk_matches_target_sigma` | Sampled surface density matches target Σ(R) |

#### `test_ic_composites.py` — multi-component stability

| Test | What it checks |
|------|----------------|
| `test_composite_spherical_density_stability` | Halo/bulge/disk combos: ρ(r) drift < 25% after short run |
| `test_exponential_disk_surface_density_stability` | Disk Σ(R) stable for halo+bulge+disk combinations |

#### `test_reproducibility.py` — deterministic runs

| Test | What it checks |
|------|----------------|
| `test_same_seed_same_final_positions` | Identical seed → identical final positions |

#### `test_parallel.py` — MPI force dispatch (requires mpi4py)

| Test | What it checks |
|------|----------------|
| `test_mpi_serial_matches_brute` | Single-rank MPI path matches brute force |
| `test_pool_dispatch_matches_brute` | `compute_forces_parallel` matches brute |
| `test_mpirun_benchmark_worker` | `mpirun` benchmark worker writes timing JSON (needs `mpirun`) |
| `test_mpirun_simulation_worker` | MPI simulation worker records energy history (needs `mpirun`) |
| `test_mpi_bh_c_matches_python_bh` | MPI `bh_c` path matches Python BH reference |
| `test_mpi_bh_matches_brute` | MPI Python BH matches brute at small N |

#### `test_parallel_density.py` — MPI density evolution (requires mpi4py + mpirun)

| Test | What it checks |
|------|----------------|
| `test_component_density_evolution_serial_vs_parallel` | Per-component ρ(r) serial vs MPI parallel agree |
| `test_mpi_multirank_component_density_matches_serial` | `mpirun -n 2` density evolution matches serial run |

#### `test_galacticsics_integration.py` — end-to-end GalactICS → ntropy

| Test | What it checks |
|------|----------------|
| `test_particle_state_from_galacticsics_halo` | `particle_state_from_galacticsics` loads halo IC with tags |
| `test_galacticsics_halo_density_stable_under_ntropy` | GalactICS NFW halo ρ(r) stable under ntropy evolution |
| `test_galacticsics_reference_disk_halo_through_ntropy` | Full reference disk+halo IC stable through ntropy |

---

### Running subsets

```bash
# galacticsics only
pytest tests/ -v

# ntropy only
pytest src/ntropy/tests/ -v

# Force backends (incl. C extension if built)
pytest src/ntropy/tests/test_forces.py src/ntropy/tests/test_bh_c.py -v

# MPI (skip automatically without mpirun)
pytest src/ntropy/tests/test_parallel.py src/ntropy/tests/test_parallel_density.py -v

# GalactICS IC stability
pytest src/ntropy/tests/test_galacticsics_integration.py -v
```

## Documentation

```bash
pip install sphinx
cd docs && sphinx-build -b html . _build/html
```

## Roadmap

### galacticsics

- [x] Phase 1: Package scaffold, I/O, tests
- [x] Phase 2: Python harmonic evaluator (SciPy Legendre)
- [x] Phase 3: `solve_potential()` via isolated legacy `dbh`
- [x] Phase 4: `DiskCorrectionTable`, `FrequencyTable` (CubicSpline)
- [x] Phase 5: Particle sampling wrappers (`gendisk`, `genhalo`, …)
- [x] Halo-first two-step workflow (`solve_halo_potential`, `h.dat` merge)
- [x] Phase 6: Rotation-curve fitting, optimizers, MCMC skeleton
- [x] Phase 7: galpy bridge, CI, docs scaffold

### ntropy

- [x] JSON-driven runs, Plummer/NFW/Sersic/disk ICs, density stability tests
- [x] Brute + Python Barnes–Hut forces, variable softening
- [x] mpi4py Morton domain decomposition; MPI brute partial-target fix
- [x] Leapfrog (orders 1–2), Euler, RK2/3/4 integrators; energy drift tests
- [x] galacticsics integration (`sample_galacticsics_*`, notebook walkthrough)
- [x] **C Barnes–Hut** (`bh_c`): flat pack/bcast MPI path, `PARALLEL.md` roadmap
- [x] Default 5 Gyr / 1000 steps·Gyr⁻¹ simulation span; time-unit helpers
- [ ] OpenMP over targets in C walk; distributed tree build at N ≳ 10⁵
- [ ] BH crossover benchmark at production N on CI hardware

## Citation

Based on the GalactICS code of Kuijken & Dubinski for generating self-consistent
galaxy initial conditions. If you use this library in published work, cite the
original GalactICS papers and document the `galacticsics` version (`pip show galacticsics`).

## License

MIT (see package metadata). Legacy Fortran/C retains its original authorship and
academic-use expectations.
