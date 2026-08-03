# Matching the distribution function without a 5D/6D mesh

Alternatives for forcing generative / reconstructed galaxy ICs closer to a
target distribution function $f(x,v)$ (equilibrium or not), **without**
building a huge phase-space histogram. Companion narratives:
[`field_maps.md`](field_maps.md) (encode chart),
[`ml_latent_ic_methods.md`](ml_latent_ic_methods.md) (latent-IC chronology),
[`dynamical_consistency_charts.md`](dynamical_consistency_charts.md)
(finite-horizon evolve consistency).

**Part~1 framing (2026-08):** the shipped absolute-DF path is **particle
retrieve + remass to GalactICS $f_0(\theta)$** (path-LOO), not residual paint
or soft AE resample. Options A/B/F below remain useful **controls / ablations**
for dens+moment fidelity; they are **not** the primary product. Manuscript:
[`papers/mnras_noneq_ics/mnras_noneq_ics.tex`](../papers/mnras_noneq_ics/mnras_noneq_ics.tex).
Empirical OOD dens+vel numbers (historical):
[`papers/mnras_noneq_ics/results/ood_theta_df_SUMMARY.md`](../papers/mnras_noneq_ics/results/ood_theta_df_SUMMARY.md).

**Status.** Options **A** and **B** are **implemented and measured**
(2026-07-27 run `runs/ml/field_maps/df_match_BA_2026-07-27/`). Summary:
cylindrical + cell-moment resample (**B**) alone does not move OOD
$\langle v_\phi\rangle$; dens-weighted moment / $v_\phi$ FT from the FFT-long
teacher (**A**) + **B** cuts recon $\langle v_\phi\rangle$ MSE
(thick_stable $0.506\to0.474$; heavy disks $\sim10\%$ relative) and dens
med$|\log_{10}|$ (thick $0.042\to0.012$). Dyn-consistency: dens_ultra+B does
**not** rescue $\alpha=1$; fftlong **A**+**B** + m2 amplify improves
906c4 $\alpha=1$ ($0.357\to0.275$ vs prior $\sim0.20$), clears $\ge0.35$ at
$\alpha=1.5$ ($0.469\to0.360$), and reaches data-flat at $\alpha=2.5$
($0.572\to0.437$). Morph-heavy / dens_ultra+A FTs aimed at $\alpha\to1$
**regressed** 906c4 (`alpha1_push_2026-07-27/`). Option **F** (OT-lite)
is implemented: with IC dens/moment targets, OOD $\langle v_\phi\rangle$ MSE
drops to $\sim10^{-3}$ (thick_stable $0.474\to0.001$). **C, D, E, G, H**
remain proposals. See §6 and
[`alpha1_push JOURNAL`](../runs/ml/field_maps/alpha1_push_2026-07-27/JOURNAL.md).

**Relative to Part~1 retrieve+decode:** treat A/B/F as diagnostic tools that
improved soft-field kinematics; the evolve scoreboard (906c4 MATCH / no-bulge
FADE MATCH) uses remassed path-LOO particles, not painted residuals.

---

## 1. Problem statement

### What “match the DF” means

A collisionless galactic component is described by a phase-space density
$f_c(x,v;t)$ with

$$
\int f_c\,\mathrm{d}^3x\,\mathrm{d}^3v = M_c,\qquad
\rho_c(x)=\int f_c\,\mathrm{d}^3v,\qquad
\langle v\rangle_c,\ \sigma_c,\ \ldots
=\text{velocity moments of }f_c.
$$

For **equilibrium** GalactICS ICs, $f$ is (approximately) a function of
integrals of motion — e.g. disk $f(E,L_z,E_z)$, halo/bulge Eddington $f(E)$ —
and particles are drawn by accept/reject against that DF in a fixed multipole
potential ([`galacticsics_pipeline.md`](galacticsics_pipeline.md),
[`ic_sampling.md`](ic_sampling.md)). For **non-equilibrium** dumps (bars,
transients), there is no low-dimensional integral chart; the “true DF” *is*
the empirical particle measure

$$
\hat\mu_N = \sum_{i=1}^{N} m_i\,\delta_{x_i}\otimes\delta_{v_i}
$$

(or its continuum idealisation). Matching the DF then means: a reconstructed or
generated IC $\hat\mu$ should be close to $\mu_{\mathrm{target}}$ in a metric
that controls both **mass morphology** and **kinematics**, not only face-on
$\Sigma$ or particle $\mathrm{A}_2$.

Operationally (this repo’s diagnostics), that includes at least:

| Layer | Observable | Role |
|-------|------------|------|
| Dens | $\Sigma(x,y;z\text{-slab})$, $\rho(r)$ | Poisson source / morphology |
| Streaming | $\langle v\rangle$, esp. $\langle v_\phi\rangle(R)$ | rotation / bar streaming |
| Dispersion | $\sigma_R,\sigma_z,\ldots$ | heat / Toomre / thickness |
| Structure | $A_m(R)$, COM | bar / spiral / centring |
| Dynamics | short evolve track | whether mismatches amplify |

A **full** match would control all of $f(x,v)$; we never claim that dens+moments
or $A_m$ alone certify that.

### Why a full 5D/6D mesh is intractable

Naïve binning of $(x,v)$ at useful resolution is catastrophic. Even a modest
grid — e.g. $64^3$ spatial × $32^3$ velocity ≈ $8.6\times10^{10}$ cells —
exceeds memory and sample count long before Poisson noise is acceptable
($N\sim10^6$ particles ⇒ most cells empty). Adaptive trees and KD histograms
help locally but do not change the curse of dimensionality for a *global*
training loss over a corpus.

Hence every practical approach replaces $\|f-\hat f\|_{L^1(\mathbb{R}^6)}$ by
one of:

1. **Low-order field moments** on spatial meshes (what we already use);
2. **Particle-pair / OT metrics** that never allocate a $v$-grid;
3. **Physically chosen 1–3D projections** ($f(E,L_z)$, $\Sigma(R)$, $A_m(R)$, …);
4. **Transport / flow models** that push a reference measure toward the target.

This note catalogues those alternatives in the context of the current
multi-scale field + teacher AE + library amplify + dens+moment resample stack.

---

## 2. What we already match vs full $f(x,v)$

### Pipeline chart (implemented)

```
snapshot particles
    → shared COM / frame          (ml/fields/frame.py)
    → deposit dens + ⟨v⟩ + σ      (ml/fields/binning.py; moment_set="disp")
    → MultiTowerSliceAE encode/decode   (ml/fields/autoencoder.py)
    → (generative) teacher feature library + amplify / θ-nn retrieve
    → denormalize
    → resample dens (+ cell ⟨v⟩ ± σ·N(0,1))   (ml/fields/resample.py)
    → optional shell / hybrid bulge
    → evolve gate / OOD DF compare
```

**Field loss (AE / VAE).** Channel-weighted MSE on log1p dens and tanh / log1p
moments (`reconstruction_loss` / `multitower_reconstruction_loss`), with
defaults along the crisp track `dens_w≈4`, `moment_w≈6`. Optional extras
(Fourier $A_m$, FFT morphology, dens residual, dens-grad) act on **dens maps**,
not on a joint dens–kinematics objective beyond the flat moment MSE.

**Resample.** Positions ~ dens mass; velocities = cell mean $\pm$ isotropic
Gaussian with deposited $\sigma$ (optional). This matches the **first two
velocity moments in each slab cell** in expectation, *if* the moment maps are
accurate. It does **not** sample a true DF (higher cumulants, $v$-anisotropy
beyond diagonal $\sigma$, correlations with $E$/$L_z$, etc.). Soft virial $Q$
is typically hot after dens resample.

**What is *not* matched today**

- Full $f(x,v)$ or even $f(E,L_z)$ histograms.
- Joint consistency constraints between dens and moments beyond separate MSE
  channels (e.g. linearized dens-weighted $\langle v_\phi\rangle$ error).
- Particle-space OT / MMD between recon and target clouds.
- Native structural-$\theta$ generative conditioning (library is morphology /
  PCA-$z$ indexed; OOD “generative” arm is θ-nearest quiet retrieve).

### Empirical gaps that motivate DF-aware work

From OOD θ DF compare (`scripts/ood_theta_df_compare.py`,
`ood_theta_df_compare_2026-07-27`):

| Fact | Implication |
|------|-------------|
| FFT recon disk med $\|\log_{10}\rho\|\sim 0.04$ | Dens path is already strong on-disk. |
| θ-nearest systematically offset in $\Sigma(R)$ | Generative retrieve hits a *train* mass model, not held-out $\theta$. |
| $\langle v_\phi\rangle$ MSE: recon $\sim0.20$–$0.26$ (heavy) but $\sim0.51$ on `thick_stable` | Velocity moments — especially rotation — are the honest kinematics gap. |
| $\sigma_R$ often colder than GalactICS | Dispersion maps / resample under-heat. |
| Deposit dens+moments ≈ oracle on dyn-consistency; $\alpha=1$ (no m2 amplify) still fails bar persistence | Soft AE dens needs a morph crutch; moments-heavy FT helps but DF match ≠ morph amplify. |
| Dyn-consistency / Prop A metrics are dens / $A_2$ / COM | Passing evolve gates does **not** certify DF proximity. |

**Bottom line.** Morphology + dens are ahead of kinematics and of true
phase-space proximity. Closing “DF match” should target **velocity moments and
particle kinematics**, not another dens-only sharpening pass.

---

## 3. Alternatives

Each option: idea, math sketch, compute/memory, pros/cons, fit to this codebase,
risks. None requires a full $(x,v)$ mesh.

---

### A. Joint dens+moments field loss (linearized, weighted)

**Idea.** Keep the spatial field representation, but stop treating dens and
moments as independent channel MSEs in normalised space. Penalise errors in
**physical** dens and in dens-weighted streaming / dispersion, so rare high-$\Sigma$
cells and $\langle v_\phi\rangle$ matter where the mass lives.

**Math sketch.** Let $\Sigma$ be linearized dens (`expm1` of log1p channels;
already used for Fourier / FFT terms via `dens_scale`). For a moment field $m$
(e.g. $v_x,v_y$ or cylindrical $v_\phi$ deposited as an extra diagnostic):

$$
\mathcal{L}_{\mathrm{joint}}
=\lambda_\Sigma\|\hat\Sigma-\Sigma\|^2_{w_\Sigma}
+\lambda_m\bigl\|
\sqrt{\Sigma}\,(\hat m-m)
\bigr\|^2
+\lambda_\sigma\bigl\|
\sqrt{\Sigma}\,(\hat\sigma-\sigma)
\bigr\|^2.
$$

Optional cylindrical rewrite on midplane:

$$
\mathcal{L}_{v_\phi}
=\bigl\|
\sqrt{\Sigma}\,\bigl(\widehat{\langle v_\phi\rangle}-\langle v_\phi\rangle\bigr)
\bigr\|^2_{R\text{-bins or pixels}}.
$$

This is still a **field** loss — $O(\text{pixels})$ — not a 6D histogram.

**Compute / memory.** Same as current AE train; one extra denormalize / cyl
projection per batch. Negligible vs U-Net forward.

**Pros.** Directly attacks the measured $\langle v_\phi\rangle$ gap; compatible
with frozen-teacher library path (fine-tune then re-encode library); no new
particle metric machinery.

**Cons.** Still only low-order moments; cannot fix non-Gaussian $v$ structure.
Over-weighting $v_\phi$ can fight FFT morph / bar residual terms if not staged.

**Fit.** Natural extension of `reconstruction_loss` in
`ml/fields/autoencoder.py` and smoke knobs in `scripts/smoke_field_maps.py`.
Cylindrical profiles already exist in `ml/profiles.py` and OOD DF scripts.

**Risks.** Log-space vs linear dens inconsistency if weights are naive; quiet
disks with small $\Sigma$ outer bins can dominate if not mass-masked; need
hold-out that $\mathrm{A}_2$ / quietness do not regress.

---

### B. Moment-consistent / local kinematics resampling

**Idea.** Improve $\mathcal{R}$ so that, **given** (possibly imperfect) moment
maps, drawn particles realise those moments more faithfully — and, where maps
are weak, borrow local kinematics from a reference (deposit oracle, nearest
library member, or GalactICS DF sample) under a dens lock.

**Math sketch.** Current resample (per cell / slab pixel):

$$
x\sim\mathrm{Cat}(\Sigma),\qquad
v=\langle v\rangle(x)+\sigma(x)\odot\varepsilon,\quad\varepsilon\sim\mathcal{N}(0,I).
$$

Moment-consistent upgrades (increasing strength):

1. **Local frame.** Draw in $(v_R,v_\phi,v_z)$ with deposited cylindrical
   moments, then rotate to Cartesian (reduces $\langle v_\phi\rangle$ leakage
   from Cartesian $\sigma$ anisotropy).
2. **Exact cell moment matching.** After sampling $n_c$ particles in a cell,
   affine-correct velocities so sample mean / cov match $(\langle v\rangle,\sigma)$
   (or Winsorised targets).
3. **Dens-lock + kinematics transplant.** Keep positions from $\hat\Sigma$;
   assign velocities by $k$NN / OT in $x$ from a reference particle set with
   trusted DF (deposit snapshot or GalactICS IC), optionally blending with AE
   $\langle v\rangle$.
4. **Shell / hybrid bulge** (already implemented): spherical-shell dens+moments
   for the cusp; stitch retained GalactICS bulge — shows dens-lock + better
   geometry already helps one component.

**Compute / memory.** (1)–(2) cheap $O(N)$. (3) $O(N\log N)$ or Sinkhorn on
spatial subsets; memory = particle buffers, not 6D grids.

**Pros.** Fixes a known under-specified map→particle step; orthogonal to AE
weights; can use deposit moments as teacher at train time and AE moments at
pure-gen time; shell bulge precedent in-repo.

**Cons.** Cannot invent correct $\langle v_\phi\rangle(R)$ if maps are wrong —
must pair with **A**. Transplant (3) blurs “learned” kinematics; θ-OOD still
needs a good reference.

**Fit.** `ml/fields/resample.py` (`resample_particles_from_*`,
`fuse_shell_bulge_with_multiscale`). Diagnostics:
`scripts/ood_theta_df_compare.py`.

**Risks.** Affine moment matching can create shot-noise spikes in low-$n$ cells;
cylindrical draws need careful empty-cell handling; over-reliance on deposit
oracle reintroduces the amplify / oracle gap for pure generative ICs.

---

### C. Particle-space metrics without a mesh (sliced Wasserstein, MMD, energy distance)

**Idea.** Compare recon/generated particle clouds to targets with metrics that
are consistent for empirical measures on $\mathbb{R}^6$ (or $\mathbb{R}^3$ with
mass) without binning $v$.

**Math sketch.** For empirical measures $\mu,\nu$ and random 1D projections
$\theta\sim\mathrm{Unif}(\mathbb{S}^{d-1})$:

$$
\mathrm{SW}_p^p(\mu,\nu)
=\mathbb{E}_\theta\bigl[
W_p^p\bigl(\theta_\#\mu,\theta_\#\nu\bigr)
\bigr],
$$

with $W_p$ on $\mathbb{R}$ computable by sorting. MMD with kernel $k$:

$$
\mathrm{MMD}^2(\mu,\nu)
=\mathbb{E}_{x,x'\sim\mu}k(x,x')
+\mathbb{E}_{y,y'\sim\nu}k(y,y')
-2\mathbb{E}_{x\sim\mu,\,y\sim\nu}k(x,y).
$$

Energy distance is the special case of a distance-induced kernel. Restrict to
disk particles, or to $(R,\phi,z,v_R,v_\phi,v_z)$, or to mass-stratified
subsample $N\sim 2\times10^4$–$10^5$.

**Compute / memory.** SW: $O(N_p\,N\log N)$ per batch ($N_p$ projections).
MMD: $O(N^2)$ naive or $O(N)$ with random features. Fits in RAM at subsample
sizes; no 6D array.

**Pros.** True phase-space discrepancy signal; catches errors invisible to
slab moments (e.g. wrong $v_\phi$–$R$ coupling). Good **eval** metric even if
not in the train loss.

**Cons.** High variance; scale / units sensitive (kpc vs km s$^{-1}$); can
fight morphology if used as sole generative loss; expensive inside AE pixel
training loop unless applied post-resample on a student head.

**Fit.** New eval module + hooks in `ood_theta_df_compare.py` /
`evolve_resample_compare.py`. Train-time use: better on a small kinematics
head or after freeze of dens towers (see **E**).

**Risks.** Without whitening / separate spatial vs velocity kernels, SW is
dominated by position mismatch; θ-nn offset will look “bad” for the wrong
reason (wrong mass model) unless compared within matched-$\theta$ pairs.

---

### D. Low-dim physically motivated projections

**Idea.** Match a small set of astrophysically meaningful marginals —
cheaper and more interpretable than SW on raw $\mathbb{R}^6$.

**Math sketch.** Examples already partially in the stack:

| Projection | Status |
|------------|--------|
| $\Sigma(R)$, $\rho(r)$ | Soft profiles / OOD dens panels |
| $\langle v_\phi\rangle(R)$, $\sigma_R(R)$, $\sigma_z(R)$ | OOD DF compare |
| $A_m(R)$ | Soft Fourier loss (optional; invent risk on quiet) |
| $f(E,L_z)$ or $f(E)$ histograms | GalactICS eq path; **not** used as ML loss today |
| Toomre $Q(R)$, $\beta(r)$ | Diagnostics / full moment_set |

Loss sketch:

$$
\mathcal{L}_{\mathrm{proj}}
=\sum_k w_k\,d_k\bigl(\Pi_k\hat\mu,\Pi_k\mu\bigr),
$$

with $d_k$ = MSE on radial profiles, KL on coarse $(E,L_z)$ bins, or 1D
Wasserstein on $v_\phi$ at fixed $R$ shells.

**Compute / memory.** Profile MSE: trivial. Coarse $(E,L_z)$: needs $\Phi$
(Plummer / tree) + 2D hist with $\sim 32\times32$–$64\times64$ bins —
tractable, **not** 6D.

**Pros.** Aligns with paper / OOD reporting; $f(E,L_z)$ is the right language
for near-eq ICs; $A_m(R)$ already taught morph lessons.

**Cons.** Projections can match while full $f$ differs; $A_m$ overweight invents
quiet structure (documented); $E$ needs a consistent $\Phi$ definition across
recon vs data.

**Fit.** `ml/profiles.py`, Fourier helpers in `autoencoder.py`, DF validation
ideas from `galacticsics.diagnostics`. Extend OOD DF script rather than AE
core first.

**Risks.** Equilibrium-centric projections mislead on strong bars; mixing
$A_m$ into DF objectives reopens the quiet-invention failure mode.

---

### E. Density-lock then kinematics head / flow

**Idea.** Two-stage generative map: (1) lock positions / dens from the strong
morphology path; (2) learn velocities conditional on $x$ (and $\theta$, skips).

**Math sketch.** Factor

$$
f(x,v)\approx \rho(x)\,f(v\mid x).
$$

Keep $\hat\rho$ from frozen teacher + amplify / shell bulge. Train a
conditional model $v\sim p_\psi(v\mid x,F)$ — e.g. diagonal Gaussian head
(repredict $\langle v\rangle,\sigma$), mixture, or lightweight normalizing flow
/ RealNVP on $(v_R,v_\phi,v_z)$.

**Compute / memory.** Dens path unchanged. Kinematics head: MLP / small flow on
particle batches; memory ≪ full field VAE if dens towers stay frozen.

**Pros.** Matches empirical strength ordering (dens OK, vel weak); avoids
re-fighting FFT morph; clear ablation (AE moments vs learned $f(v\mid x)$).

**Cons.** Conditional independence of particles given $x$ still misses
collective DF structure; flows on $v$ need careful scaling; another model to
maintain beside the teacher library.

**Fit.** Post-`resample` velocity replacement, or replace σ-sampling in
`resample.py`. Conditioning features: teacher skips pooled at particle $x$,
or deposited AE moment maps as a baseline to beat.

**Risks.** If $\hat\rho$ is wrong (θ-nn offset), kinematics cannot fix mass;
amplify-soft dens + perfect $f(v\mid x)$ may still fail 2 Gyr bar gates.

---

### F. Phase-space transport from nearest GalactICS DF (OT-lite / displacement)

**Idea.** For OOD / generative ICs, start from a **nearest reference** with a
trusted DF (θ-nearest GalactICS IC or library member), then apply a cheap
transport so dens / moments match the desired target field — instead of
decoding a mismatched mass model and hoping moments follow.

**Math sketch.** Let $\mu_0$ be particles from nearest train / GalactICS IC.
Solve for a displacement field $T$ (spatial, or spatial+velocity):

$$
\mu = T_\#\mu_0,\qquad
\Sigma[T_\#\mu_0]\approx\Sigma_\star,\quad
\langle v\rangle[T_\#\mu_0]\approx m_\star,
$$

via: (i) radial mass remapping + $v_\phi(R)$ rescaling; (ii) sliced-OT /
Sinkhorn on positions then velocity copy; (iii) learned residual displace
$\Delta x,\Delta v = g_\psi(x,v,\theta_\star-\theta_0)$.

**Compute / memory.** Classical radial remap: $O(N\log N)$. Sinkhorn on
$N\sim10^5$ subsample: GPU-friendly. Full $N=10^6$ OT is heavy — use
minibatch / hierarchical (disk annuli).

**Pros.** Directly addresses θ-nearest **mass-model offset**; preserves a real
DF’s higher-order structure better than dens+σ resample from scratch; natural
OOD story (“transport the nearest eq DF toward target $\theta$ / morph”).

**Cons.** Not a pure latent generative prior; quality tracks library coverage;
barred non-eq targets may need morph amplify *after* transport.

**Fit.** New script beside `ood_theta_compare.py` / feature library retrieve;
reuse θ coverage JSON. Pair with **B** dens-lock kinematics.

**Risks.** Blind velocity copy after only spatial OT yields wrong streaming in
a new potential; must re-virialise or re-solve moments in the target $\Phi$.

---

### G. Evolve-gated DF match at short times

**Idea.** Treat short $N$-body evolution as a critic: ICs that match $f$ only
at $t=0$ in soft metrics but diverge in dens / $A_2$ / $\langle v_\phi\rangle(R)$
by $t\sim 0.1$–$0.5\,\mathrm{Gyr}$ fail. Add a **gate** (and optionally a
trainable penalty via unrolled or surrogate dynamics).

**Math sketch.** Proposition A style
([`dynamical_consistency_charts.md`](dynamical_consistency_charts.md)):

$$
\mathrm{dist}_t
=\mathrm{dist}\bigl(\mathcal{U}_t\mu,\,\mathcal{U}_t\hat\mu\bigr),
\qquad
\mathcal{L}_{\mathrm{gate}}
=\sum_{t\in\mathcal{T}} w_t\,\mathrm{dist}_t,
$$

with $\mathrm{dist}$ extended beyond $A_2$/COM to include profile
$\langle v_\phi\rangle$ MSE and optional SW on disk subsets. Train-time full
unroll is expensive; practical pattern: **eval gate** now, **distillation**
from deposit-oracle evolves later.

**Compute / memory.** Eval: existing evolve scripts (`gpu_bh`, $T=0.5$).
Train: only if surrogate / few-step — otherwise keep offline.

**Pros.** Aligns with the paper’s dynamical-consistency claim; catches cold
disks and amplify artefacts that look fine at $t=0$; deposit≈oracle already
shows the ceiling.

**Cons.** Does not by itself fix maps; $\alpha=1$ failures show morph soft
dens still needs A/B or amplify; costly as a primary train loss.

**Fit.** Extend `scripts/ood_theta_df_compare.py` post-evolve tables and
`dyn_consistency_*` scoreboards with explicit DF profile metrics.

**Risks.** Overfitting gate thresholds; mistaking bar fade for DF error when
$N$ / force soft differ.

---

### H. Continuous DF / normalizing flow / score model (longer-shot)

**Idea.** Learn $f_\psi(x,v\mid\theta,z)$ or $\nabla\log f$ directly as a
continuous density, sample ICs by flow / Langevin, bypass slab moments.

**Math sketch.** CNF / FFJORD / RealNVP on whitened $(x,v)$, or score matching
on deposited noise; condition on structural $\theta$ and morph $z$/skips.
Loss: NLL, score MSE, or flow-matching OT path.

**Compute / memory.** High: many NFEs, careful architectures for $10^6$
particles; training on full phase space is research-scale.

**Pros.** True generative DF; θ-conditioning is natural; no slab cusp geometry
hack if base measure is well designed.

**Cons.** Bars in skips / library path already beat end-to-end field VAEs;
particle Morton VAE underperformed on morph; risk of repeating skip-wash in a
new costume.

**Fit.** Research track only (`ml/models/sequence_vae.py`, latent_code flows).
Do **not** block A–G.

**Risks.** High implementation cost; likely loses to teacher-library morph
unless kinematics-only (then collapses to **E**).

---

## 4. Comparison table

| | Attacks dens | Attacks $v$ moments | Mesh-free 6D signal | Code fit | Effort | Best role |
|--|:---:|:---:|:---:|:---:|:---:|--|
| **A** Joint field loss | partial | **strong** | no | AE loss | low | Train teacher / FT |
| **B** Moment-consistent resample | lock | **strong** | no | `resample.py` | low–mid | Map→particle |
| **C** SW / MMD / energy | yes | yes | **yes** | eval (+ optional loss) | mid | OOD metric / regulariser |
| **D** Physical projections | yes | yes | no (1–3D) | profiles / OOD | low | Report + aux loss |
| **E** Dens-lock + $f(v\mid x)$ | uses A/B dens | **strong** | optional | new head | mid–high | Pure-gen kinematics |
| **F** OT from nearest DF | **strong** (OOD) | medium | OT-lite | retrieve + transport | mid | θ-OOD generative |
| **G** Evolve-gated DF | indirect | indirect | optional | evolve scripts | mid | Acceptance / ceiling |
| **H** Continuous DF / score | yes | yes | yes | research | high | Long-shot prior |

---

## 5. Recommendation

### Primary: **B + A** first

**Do moment-consistent resampling (B) and joint dens-weighted moment field loss
(A) before any particle OT or flow rewrite.**

**Why (opinionated, from failure modes):**

1. **$\langle v_\phi\rangle$ gap is the clearest DF miss** on recon
   (`thick_stable` MSE $\sim0.5$ while dens med $|\log_{10}|\sim0.04$). That is
   exactly what dens-weighted $v_\phi$ / streaming losses (**A**) and
   cylindrical moment-consistent draws (**B**) target — without a 6D mesh.
2. **Amplify is a dens-morph crutch** ($\alpha=1$ fails dyn-consistency;
   deposit moments ≈ oracle). Improving kinematics should not wait on another
   amplify knob; A/B reduce dependence on “hope the AE moments are good enough
   for Cartesian σ sampling.”
3. **θ-nearest offset is a mass-model problem**, not fixed by SW alone. A/B
   improve *recon* DF fidelity and the map→particle step; OOD generative then
   needs **F** (or real θ-conditioning), with **C** as the honest metric.
4. Implementation cost is low and local: `autoencoder.py` loss terms +
   `resample.py` kinematics — same charts the paper already plots.

**Secondary for OOD (next):** **F** (transport / remap from nearest GalactICS
or library DF toward target dens+moments) and/or **C** (sliced Wasserstein /
MMD on disk phase space) as evaluation and light regularisation. Use **D**
profile losses as cheap auxiliaries aligned with `ood_theta_df_compare`.
Keep **G** as the acceptance gate (extend dist to $v_\phi(R)$). Defer **H**.

### Staged roadmap

| Stage | Horizon | Work |
|-------|---------|------|
| 1 | days | **B1–B2**: cylindrical velocity frame + cell moment matching in `resample.py`; ablate vs current Cartesian σ draw on FFT recon |
| 2 | ~1 week | **A**: dens-linearized, $\sqrt{\Sigma}$-weighted moment / $v_\phi$ terms in `reconstruction_loss`; smoke FT from `fft_morph_ft_long` or `moments_barheavy_ft` |
| 3 | ~1–2 weeks | Re-run `ood_theta_df_compare` + short evolve; success = lower $v_\phi$ MSE without dens / quiet regression |
| 4 | follow-on | **F** radial/OT-lite from θ-nearest IC; **C** as primary OOD DF metric; optional **E** if AE moments plateau |

---

## 6. Concrete next experiment (1–2 weeks)

### Goal

Cut disk $\langle v_\phi\rangle$ MSE on OOD recon (esp. `thick_stable`) while
holding disk med $|\log_{10}\rho|\lesssim 0.05$ and quiet $A_2\lesssim 0.02$.

### Loss / algorithm changes

1. **`resample.py`**
   - Add `velocity_frame="cylindrical"` draws from deposited
     $(\langle v_R\rangle,\langle v_\phi\rangle,\langle v_z\rangle)$ and
     $(\sigma_R,\sigma_\phi,\sigma_z)$ (deposit these or derive from Cartesian
     maps).
   - Optional `match_cell_moments=True`: after sampling, shift/scale velocities
     per cell to hit target mean/σ.
2. **`autoencoder.py` / `reconstruction_loss`**
   - Add `moment_phys_weight` (or per-key weights): on linearized dens scale,
     $\mathcal{L}\supset \|\sqrt{\Sigma_+}(\hat m-m)\|^2$ for streaming
     channels; optional explicit midplane $v_\phi$ map term.
   - Keep `moment_w ≥ dens_w`; do **not** raise `a2_weight` / `fft_weight` in
     this experiment (isolate kinematics).
3. **Smoke train** — short FT via `scripts/smoke_field_maps.py` from
   moments-capable teacher (`moments_barheavy_ft` or FFT-long); log
   `mse_mom`, dens MSE, and a $v_\phi$ profile MSE callback.

### Scripts to touch / run

| Script / module | Role |
|-----------------|------|
| `src/galacticsics/ml/fields/resample.py` | **B** kinematics |
| `src/galacticsics/ml/fields/autoencoder.py` | **A** joint loss |
| `scripts/smoke_field_maps.py` | FT + recon smoke verdict |
| `scripts/ood_theta_df_compare.py` | Primary success metric |
| (optional) `scripts/evolve_resample_compare.py` | Short morph/COM sanity |

### Success metrics (pass / fail)

| Metric | Pass if |
|--------|---------|
| FFT recon disk med $\|\log_{10}\rho\|$ | $\lesssim 0.05$ (no dens regression vs ~0.04) |
| `thick_stable` $\langle v_\phi\rangle$ MSE | clear drop vs **0.506** baseline (target $\lesssim 0.30$) |
| heavy-disk $\langle v_\phi\rangle$ MSE | $\le$ current ~0.20–0.26 (no harm) |
| $\sigma_R$ MSE | not worse by $\gtrsim 2\times$; prefer less cold bias |
| Quiet / OOD $A_2$ | stays $\lesssim 0.015$ at $t=0$ and $t=0.5$ |
| θ-nearest arm | **raw** GalactICS snapshot particles by default (`via_ae=False`); legacy AE decode optional — AE moments collapse outer ⟨v_φ⟩ |

### Explicit non-goals for this sprint

- Full SW train loop (**C** eval-only OK).
- Continuous DF / score models (**H**).
- Replacing m2 amplify for 2 Gyr bar gates (**G** / morph track stays separate).

### Results (2026-07-27) — shipped

Run root: `runs/ml/field_maps/df_match_BA_2026-07-27/`.
Teacher for best DF: `joint_df_fftlong_ft/multitower_slice_ae.pt`
(warm `fft_morph_ft_long` + `moment_phys_w=6`, `vphi_phys_w=10` + **B** resample).

**OOD $t=0$ fft_recon $\langle v_\phi\rangle$ MSE / dens med$|\log_{10}|$:**

| Case | baseline fft_long | **A+B fftlong** | Δ vφ |
|------|-------------------|-----------------|------|
| heavy_ext_quiet | 0.199 / 0.038 | **0.180** / 0.031 | −9.5% |
| heavy_bar_forming | 0.258 / 0.041 | **0.231** / 0.035 | −11% |
| thick_stable | 0.506 / 0.042 | **0.474** / **0.012** | −6.4% |

- **B alone** on fft_long ≈ null on $\langle v_\phi\rangle$ (maps are the bottleneck).
- Moments-barheavy A FT alone: dens better, $\langle v_\phi\rangle$ slightly worse than fft_long.
- thick_stable still $>$ 0.30 target — partial pass; dens holds (improved).

**A-loss axis bug + retrain (same day).** The $v_\phi$ map term had
**swapped axes** vs `histogram2d` (fixed in `autoencoder.py`). Retrain with
the corrected loss (`runs/ml/field_maps/vphi_fixed_A_ft_2026-07-27/`) is a
**null** on OOD $\langle v_\phi\rangle$ vs buggy A
($0.180/0.231/0.474\to0.180/0.230/0.475$; min_count particle profiles).
Attribute the table gains to **moment_phys + dens**, not `vphi_phys`.
Summary: [`vphi_fixed_A_SUMMARY.md`](../papers/mnras_noneq_ics/results/vphi_fixed_A_SUMMARY.md).

**906c4 2 Gyr gates (m2 amplify + shell bulge, gpu_bh):**

| Recipe | α=1 | α=1.5 | α=2.5 |
|--------|-----|-------|-------|
| dens_ultra + B | 0.352→0.200 FAIL | — | 0.604→0.335 |
| **fftlong A+B** | **0.357→0.275** (fails data-flat) | **0.469→0.360** (≥0.35) | **0.572→0.437** (≈ data-flat 0.425) |
| morph_a_ft | 0.355→0.143 L | 0.483→0.198 L | 0.597→0.269 L |
| ultra_A_ft | 0.368→0.171 L | 0.500→0.231 L | 0.612→0.308 L |
| 54a8 dens_ultra+B | — | — | 0.473→0.311 |

Do **not** claim α=1 data-flat. α=1.5 clears ≥0.35; α=2.5 still best.
morph_a_ft / ultra_A_ft **regressed** — do not recommend. Journal:
[`alpha1_push JOURNAL`](../runs/ml/field_maps/alpha1_push_2026-07-27/JOURNAL.md).

### Results (2026-07-27) — OT-lite **F** executed

Implemented `transport_ot_lite` in `ml/fields/resample.py`: mass-weighted
radial CDF remap (disk $R$; bulge/halo $r$) + disk $\langle v_\phi\rangle(R)$
residual swap (+ optional $\sigma$ rescale) from θ-nearest library DF → OOD
GalactICS IC dens/moments target. Wired as `ot_lite` arm in
`scripts/ood_theta_df_compare.py` (`--with-ot-lite`).

Run: `runs/ml/field_maps/alpha1_push_2026-07-27/ood_ot_lite/`
(teacher = fftlong A+B). **Target = GalactICS IC dens+moments** (oracle morph
target for transport — not a pure latent generative prior).

| Case | fft_recon $v_\phi$ | θ-nn $v_\phi$ | **ot_lite $v_\phi$** | dens disk med$\|\log_{10}\|$ ot |
|------|-------------------|---------------|---------------------|--------------------------------|
| heavy_ext_quiet | 0.180 | 0.415 | **0.0026** | 0.464 (θ-nn 0.465) |
| heavy_bar_forming | 0.231 | 0.474 | **0.00047** | 0.292 (θ-nn 0.265) |
| thick_stable | **0.474** | 0.381 | **0.0011** | 0.188 (θ-nn 0.212) |

**W:** OT-lite F crushes OOD $\langle v_\phi\rangle$ MSE on all three cases
(thick_stable $0.474\to0.001$; similar −99% for heavy disks). $\sigma_R$
likewise. Quiet $A_2$ stays $\lesssim0.012$.
**L:** disk dens med$|\log_{10}|$ only modestly improved (radial CDF ≠ full
face-on morph); halo dens medlog **regresses** (~0.2→0.67) — spherical remap
needs a tighter FOV / mass lock. Does not replace m2 amplify for 2 Gyr bars /
bar morph.

Figs (paper dir): `papers/mnras_noneq_ics/figures/fig_dfmatch_ood_*_otF.png`;
summary
[`ood_theta_df_ot_lite_SUMMARY.md`](../papers/mnras_noneq_ics/results/ood_theta_df_ot_lite_SUMMARY.md).

**α→1 morph push (same day):** morph-heavy FT from fftlong A+B
(`alpha1_push_2026-07-27/morph_a_ft`) and dens_ultra+A (`ultra_A_ft`) both
**regressed** 906c4 vs fftlong A+B. Best α=1 remains **0.357→0.275**; amplify
crutch not killed. Keep morph track separate; next candidate is baked/learned
m2 amplify or dens⊕moments hybrid decode — not dens-weight cranking on the
A+B teacher.

---

## References in-repo

- Field morph path: [`field_maps.md`](field_maps.md)
- Latent IC chronology: [`ml_latent_ic_methods.md`](ml_latent_ic_methods.md)
- Dyn-consistency charts: [`dynamical_consistency_charts.md`](dynamical_consistency_charts.md)
- OOD DF results: [`papers/mnras_noneq_ics/results/ood_theta_df_SUMMARY.md`](../papers/mnras_noneq_ics/results/ood_theta_df_SUMMARY.md)
- Code: `ml/fields/{autoencoder,resample,binning,feature_library}.py`,
  `scripts/{smoke_field_maps,ood_theta_df_compare,ood_theta_compare}.py`
