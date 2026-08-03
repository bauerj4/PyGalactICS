# Learnable charts for time-dependent galactic DFs

Formal claim supporting Part~1 of the MNRAS draft
([`papers/mnras_noneq_ics/`](../papers/mnras_noneq_ics/)):
FFT-teacher field reconstructions track $N$-body evolution in morphology /
centring metrics, analogous (not identical) to action–angle charts for
near-static DFs.

Honest posture: **proposition + Grönwall-style sketch + empirical verification**,
not a uniqueness theorem for Vlasov solutions.

---

## 1. Collisionless DF on GalactICS / MW-like systems

Let a galactic system consist of collisionless components
$c\in\{\mathrm{disk},\mathrm{halo},\mathrm{bulge}\}$ with total distribution
function

$$
f(x,v;t)=\sum_c f_c(x,v;t),\qquad
\int f_c\,\mathrm{d}^3x\,\mathrm{d}^3v = M_c,
$$

evolving under the collisionless Boltzmann (Vlasov–Poisson) equation with
self-gravity

$$
\partial_t f + v\cdot\nabla_x f - \nabla\Phi\cdot\nabla_v f = 0,
\qquad
\nabla^2\Phi = 4\pi G\,\rho,\quad
\rho(x;t)=\int f\,\mathrm{d}^3v.
$$

Denote the exact self-gravitating flow on measures / DFs by the (nonlinear)
operator $\mathcal{U}_t$:

$$
f(\,\cdot\,;t)=\mathcal{U}_t f_0.
$$

In the $N$-body realisation used here, $f$ is approximated by equal- or
stratified-count particles with shared global centre-of-mass (COM) frame
([`ml/fields/frame.py`](../src/galacticsics/ml/fields/frame.py)); forces are
Barnes–Hut / direct OpenMP (`bh_c`). We work at fixed particle-number class
(disk alone $=N_{\mathrm{disk}}$, mix $\approx4{:}2{:}1$).

**Scope.** Isolated MW analogues: disc-dominated, no major mergers, finite
horizon $T\lesssim 0.5$–$1\,\mathrm{Gyr}$. Soft pairwise virial ratios are
*not* used as hard consistency metrics (dens-resampled ICs are hot).

---

## 2. Learned encode / decode maps

Deposit a snapshot into multi-scale dens+moment field stacks
$\Sigma=\{ \Sigma_c \}$ (per-component slice channels). The frozen multi-tower
U-Net teacher defines

$$
\mathcal{E}_\phi:\ \Sigma \mapsto F=(B,S),\qquad
z=\Pi(B),
$$

where $B$ are tower bottlenecks, $S$ are U-Net skips (morphology carriers),
and $\Pi$ is spatial pooling + (library) PCA. The decoder reconstructs maps

$$
\hat\Sigma=\mathcal{D}_\phi(F)=\mathcal{D}_\phi\circ\mathcal{E}_\phi(\Sigma).
$$

Particles are obtained by dens+moment resampling $\mathcal{R}$
([`resample.py`](../src/galacticsics/ml/fields/resample.py)):

$$
\hat f_0 \;=\; \mathcal{R}\circ\mathcal{D}_\phi\circ\mathcal{E}_\phi(\Sigma[f_0]).
$$

We write $\mathcal{P}_\phi:=\mathcal{R}\circ\mathcal{D}_\phi\circ\mathcal{E}_\phi$
for the full particle-level recon map. Features $F$ are *not* claimed to be a
complete DF; $z$ alone is coarser still (skips carry bars).

---

## 3. Precise claim (theorem-level proposition)

### Proposition A — Approximate dynamical consistency (finite horizon)

Fix a snapshot $f_0$ in the GalactICS/MW corpus class and horizon $T>0$.
Let $\hat f_0=\mathcal{P}_\phi(f_0)$. Evolve both under the same $N$-body
operator $\mathcal{U}_t$ (same force, $\mathrm{d}t$, softening class, shared
COM convention). Then there exist morphology / centring distances
$\mathrm{dist}$ (disk dens MSE or face-on $\Sigma$ MSE; median disk $A_2$;
$|\mathrm{COM}|$) and constants $\varepsilon_0,\varepsilon_T$ such that

$$
\mathrm{dist}(f_0,\hat f_0)\le\varepsilon_0
\quad\Rightarrow\quad
\sup_{t\in[0,T]}
\mathrm{dist}\bigl(\mathcal{U}_t f_0,\,\mathcal{U}_t\hat f_0\bigr)
\le\varepsilon_T,
$$

with $\varepsilon_T$ controlled by $\varepsilon_0$ and a short-time Lipschitz
constant of $\mathcal{U}_t$ in that metric (Grönwall). Empirically we require
additionally: recon remains barred when data is barred
($A_2(t)\gtrsim 0.25$–$0.30$), quiet stays quiet ($A_2\lesssim 0.05$), and
COM drift $\lesssim 0.05\,\mathrm{kpc}$.

### Proposition B — Chart / slow feature manifold (weaker)

Along a data orbit $f(t)=\mathcal{U}_t f_0$, the encoded features
$F(t)=\mathcal{E}_\phi(\Sigma[f(t)])$ lie near a low-dimensional set relative
to raw particle coordinates $(x_i,v_i)_{i=1}^N$, and vary on a dynamical
(orbital / pattern) timescale rather than a particle-crossing timescale.
When dumps exist along an orbit, measure relative drift
$\|F(t)-F(0)\|/\|F(0)\|$.

Proposition A is the **primary** empirical claim of Part~1 evolve gates.
Proposition B is supportive when time-series dumps are available.

---

## 4. What is provable (sketch)

### 4.1 Continuity of recon in dens/moments

Training drives $\mathcal{D}_\phi\circ\mathcal{E}_\phi$ close to the identity
on dens+moment stacks in an $L^2$-type loss (plus optional FFT / $A_m$ terms).
Dens maps control the Poisson source for $\Phi$ at the grid scale used for
resampling; low-order velocity moments control the leading kinetic structure
of $\mathcal{R}$. Thus small field recon error $\Rightarrow$ small force /
streaming mismatch for the resampled IC *at the resolved scales*.

### 4.2 Short-time Lipschitz of $\mathcal{U}_t$ (Grönwall)

Work in a Banach space $X$ of measures / field observables that metrizes the
diagnostics (e.g.\ dens maps in $W^{-1,p}$ or grid $L^2$, plus $A_2$, COM).
On a short interval where the continuum Vlasov–Poisson flow is well-posed and
the $N$-body discretisation shadows it, one has a local Lipschitz bound

$$
\|\mathcal{U}_t g - \mathcal{U}_t h\|_X
\le e^{Lt}\,\|g-h\|_X
\qquad(t\le T_{\mathrm{loc}}).
$$

**Corollary (sketch).** If $\|\hat f_0-f_0\|_X\le\varepsilon_0$ and $T\le T_{\mathrm{loc}}$,

$$
\sup_{t\le T}\|\mathcal{U}_t\hat f_0-\mathcal{U}_t f_0\|_X
\le e^{LT}\varepsilon_0.
$$

This is standard stability for ODEs / mild Vlasov solutions; we do **not**
claim global-in-time uniqueness control for barred galaxies, nor that our
empirical $\mathrm{dist}$ is exactly a Vlasov norm.

### 4.3 Analogy with action–angle (not identity)

| | Near-equilibrium AA | Learned field chart (this work) |
|--|---------------------|----------------------------------|
| Object | Integrable / near-integrable Hamiltonian | Time-dependent self-gravitating DF |
| Coordinates | $(J,\theta)$ on tori | $F=(B,S)$ (+ pooled $z$) |
| Ideal dynamics | $\dot J=0$, $\dot\theta=\Omega(J)$ | No claim of trivialised flow |
| Actual claim | Exact (or adiabatic) chart | Finite-dim embedding that **approximately intertwines** with $\mathcal{U}_t$ over finite $T$ |

Action–angle makes the Hamiltonian flow *trivial* on invariant tori. Here the
claim is weaker: $\mathcal{P}_\phi$ is an approximate projector onto a
morphologically faithful particle IC whose forward orbit stays close to the
data orbit in dens / $A_2$ / COM for $t\le T$.

---

## 5. Assumptions and non-claims

**Assumptions**

- Finite horizon $T$ (paper: $0.5\,\mathrm{Gyr}$, $\mathrm{d}t\approx0.01$).
- Isolated, disc-dominated MW-like systems; no major mergers.
- Fixed particle-number class (disk alone quoted; mix $4{:}2{:}1$).
- Shared global COM / morphological origin conventions as in evolve scripts.
- Teacher $\phi$ frozen after FFT-morphology fine-tune
  (`fft_morph_ft_long_2026-07-25`).
- Diagnostics are dens / $A_m$ / COM / optional clump score — not full DF
  Wasserstein distance.

**Non-claims**

- $z$ alone is **not** a complete DF; skips still carry bars.
- $\mathcal{P}_\phi$ is **not** a learned Vlasov integrator (no time stepper in
  feature space).
- Dens+moment resample is **not** a DF sample; expect mild $A_2$ bias and hot
  soft-$Q$.
- No claim of pattern-speed fidelity or $\gtrsim 1\,\mathrm{Gyr}$ morph tracking.
- No claim that OOD structural $\theta$ generative retrieval invents unseen
  mass models (library is morphology-indexed).

---

## 6. Empirical theorem (verification protocol)

**Empirical Theorem E.** For the FFT long teacher and ≥3–5 corpus systems
spanning strong bars + ≥1 quiet, with disk alone $=N_{\mathrm{disk}}$
(feasibly $10^6$, else documented max), $T=0.5\,\mathrm{Gyr}$,
$\mathrm{d}t=0.01$, shared COM:

1. dens MSE$(\Sigma,\hat\Sigma)$ at $t=0$ is small on the disk tower
   (order few $\times10^{-2}$ normed);
2. $\Delta A_2(t)$ of recon tracks data within $\sim0.12$ over $[0,T]$
   (absolute $A_2$ may sit below data if Am under-predicted — HOLD vs dens;
   scored separately from dynamical track);
3. COM drift remains $\lesssim 0.05\,\mathrm{kpc}$ for both arms;
4. face-on morphology remains coherent (no explode / full wash).

**Result (2026-07-26).** Driver
`scripts/evolve_recon_track_multisystem.py` at disk $=2.5\times10^5$
(5 systems): **dynamical track CONSISTENT (5/5)**; absolute $A_2$ soft on
1/5 mild bar (FFT↔Am HOLD). Corpus-scale disk $=10^6$ for 54a8:
`evolve_compare_disk1e6` recon $0.302\to0.310$ tracks data $0.435\to0.430$.
Feature drift on 54a8 dumps: $\|z(t)-z(0)\|/\|z(0)\|\approx0.17$
(Proposition B). Artefacts: `runs/ml/field_maps/recon_track_multisystem_*/`,
paper `figures/fig_recon_track_*`, `results/recon_track_SUMMARY.md`.

No modeling change required for Prop A; optional Am-match / Am+FFT blend
addresses absolute-$A_2$ soft cases only.

