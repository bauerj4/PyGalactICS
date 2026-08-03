# ML findings (field VAE + Morton particle baseline)

**Part~1 status (2026-08):** primary product is **θ-conditioned latent
retrieve+decode** for *slightly non-equilibrium* ICs (particle remass to
GalactICS $f_0(\theta)$; disk-COM $A_2$). Evolve scoreboard:
**906c4 MATCH**, **no-bulge FADE MATCH**, **54a8 demoted**. Bar sweep +
OOD transfer hold; cosmological variant ensembles = outlook only.
Paint-on-bar / `residual_f0` / FFT dual-product / free skip-synth are
**demoted** relative to retrieve+decode. Manuscript:
[`papers/mnras_noneq_ics/`](../papers/mnras_noneq_ics/).

Experimental notes from the **snapshot field-map** track (recommended for
morphology) and the Morton particle-set baseline. Accurate, not marketing —
numbers are from CPU smokes / small ablations unless noted. Full algorithmic
history with detailed math, forward pathways, and post-resample evolve evidence:
[`ml_latent_ic_methods.md`](ml_latent_ic_methods.md).

## Part~1 scoreboard (latent retrieve+decode)

Primary scripts: `scripts/latent_theta_{gen,evolve_suite,ood,bar_sweep}.py`.
Evidence under `runs/ml/field_maps/latent_theta_*` and
[`papers/mnras_noneq_ics/results/`](../papers/mnras_noneq_ics/results/).

| System | Role | Verdict |
|--------|------|---------|
| **906c4** | disk+halo+bulge | **MATCH** (disk-COM $A_2$ gen $0.248\to0.150$ vs data $0.207\to0.128$ over $2\,\mathrm{Gyr}$) |
| **no-bulge** | barred disk+halo | **FADE MATCH** (gen $0.316\to0.266$ vs data $0.300\to0.173$) |
| **54a8** | t=0 LOO only | **DEMOTED** (t=0 dens/kin OK; dens-matched ICs grow while data fades) |

- **Bar sweep (fixed θ):** quiet→mild→corpus-limited strong; max true $A_2(R_d)\approx0.208$ on 906c4 (target 0.5 unreachable); $0.5\,\mathrm{Gyr}$ hold.
- **OOD θ transfer:** win/loss/partial **3/0/0** at $1\,\mathrm{Gyr}$.
- **Diagnostics:** particle $A_2(R)$ after **disk mass-COM** recenter (not halo-dominated global COM).
- **Claim:** slightly non-eq ICs via retrieve+decode. Free continuous `decode(z)` and cosmo variants are **not** claimed (outlook / Part~2).

### Demoted relative to retrieve+decode

| Path | Why demoted |
|------|-------------|
| `residual_f0` / paint-on-bar m2 | Locks radial mass to quiet $f_0$; fades / OOD mismatch |
| FFT dual-product / Am-match as primary | Useful teacher/ablation, not the shipped IC product |
| Free skip synthesis from $z$ | Washes bars ($A_2\lesssim0.08$) |
| Data-disk graft / `full_dyn_replace` | Excellent t=0 match but not generative |

## Particle-set Morton VAE

Code: `galacticsics.ml.models.sequence_vae`, `scripts/train_morton_vae.py`,
docs in [`morton_generative.md`](morton_generative.md). Virial ablation writeup:
[`runs/ml/vae_virial_ablation/NOTE.md`](../runs/ml/vae_virial_ablation/NOTE.md).

### What failed

| Failure | Observation |
|---------|-------------|
| Index-wise MSE on Morton-ordered decode | Samples collapsed; order-aligned pixel/token MSE does not preserve morphology. |
| Global `mix_head(z, θ)` soft CE | Stuck near-uniform (~3.33 nats ≈ log 28 for the mix vocab); component identity not learned from a single global head. |

### What helped

| Change | Observation |
|--------|-------------|
| Chamfer + geometric bases (`Σ(R)`, `ρ(r)`, `⟨v_φ⟩`, maps, Fourier `A_m`) | Morphology began to appear; bars/quiets distinguishable in soft maps. |
| Phase-space **per-particle** CE + stratified `c → x,v \| c` | CE became easy (~0.23–0.32, ~91–95% component accuracy). |
| Soft virial target `2K/\|W\| → 1` (stratified Plummer subsample) | Better evolve stability (COM, outer `r90`) vs matching hot equal-mass data ratios (~10–20). Strong bars can be **under-produced** if virial fights non-axisymmetric residuals — keep Fourier / map weights high. |

### Scale targets

- Training sample sizes to push toward: **10⁵–10⁶** particles (chunked decode).
- Architecture direction: **DeepSets**-style set encoder + **subsampled Chamfer** for large `N` (full pairwise Chamfer is too costly).
- While the corpus evolve holds the GPU, keep VAE training on **CPU** with small `N` / `d_model`.

### Virial ablation (summary)

From `NOTE.md` (N=384, 8 epochs): virial lowers COM drift and outer `r90`;
prod fine-tune (N=2048, `λ_virial=2`, target→1) further reduces COM on barred
examples but can suppress A₂. Prefer soft **Q→1** over matching data pairwise
virial ratios for BH evolves.

## Field maps (slices + voxels)

Code: `galacticsics.ml.fields`, smoke `scripts/smoke_field_maps.py`, artifacts
under `runs/ml/field_maps/` (see `LATEST`). Narrative:
[`field_maps.md`](field_maps.md).

### Design

- **Shared centering (required):** one mass-weighted global COM for the whole
  snapshot; never recenter disk/bulge/halo independently.  One common in-plane
  rotation after centering.  Φ uses the same origin.
  Helper: `galacticsics.ml.fields.frame.prepare_shared_frame`.
- **Per-component multi-scale FOVs**: bulge fine/small, disk midplane-focused,
  halo coarse/large — avoid a single shared canvas (but **share the origin**).
- **Slice stacks first**; anisotropic **voxels** as a vertical/3-D complement.
- Optional Plummer **Φ** on native grids; CPU AE smoke usually trains **without** Φ
  (Plummer on every batch is slow).

### Channel layout (default `moment_set="disp"`)

Per z-slab (slices) or per voxel cell:

| Index | Key | Meaning | Norm |
|------:|-----|---------|------|
| 0 | `dens` | mass / area (or / volume) | `log1p` |
| 1–3 | `vx,vy,vz` | mass-weighted mean velocity | `tanh` |
| 4–6 | `sx,sy,sz` | velocity dispersion √(⟨v²⟩−⟨v⟩²) | `log1p` |
| 7 | `beta` | optional anisotropy `1 − 2σ_z²/(σ_x²+σ_y²)` (`moment_set="full"`) | `tanh` |

Slice stack channels = `n_z · n_mom` (e.g. disk 10×7 = 70 at smoke defaults).

### Baseline 32² smoke (2026-07-24)

Shallow multi-tower CNN, dens+⟨v⟩ only, `dens_weight=3`, 25 epochs:

| Metric | Value |
|--------|-------|
| Train loss | 3.28 → **0.70** |
| Quiet resampled A₂ | ~0.04 (ok) |
| Barred data A₂ | ~0.37 |
| Barred resampled A₂ | **0.12** (weak — resolution / dens weight next) |

### Near-prod follow-up (hires + U-Net + σ)

Defaults now: disk **64²** (progressive 96–128²), bulge 48², halo 32²;
`SliceUNet` with skip connections, separate dens/moment heads, optional
cross-tower attention; soft A₂ term on disk dens; higher `dens_weight`.
Shared global COM enforced in dataset / smoke / voxel path helpers.

`runs/ml/field_maps/hires_2026-07-24/`:

| Metric | 32² baseline | Hires |
|--------|--------------|-------|
| Best train loss | 0.70 | **0.31** |
| Barred A₂ data → resampled | 0.37 → 0.12 | 0.37 → **0.26** |
| Quiet A₂ resampled | 0.036 | **0.013** |
| Disk dens MSE (norm) | ~0.39 | **~0.04** |
| `morphology_preserved` | false | **true** |

### Hifid (96² + radial A_m(R) + evolve)

`runs/ml/field_maps/hifid_2026-07-24/`:

| Metric | Hires 64² | Hifid 96² |
|--------|-----------|-----------|
| Best train loss | 0.31 | **0.38** |
| Barred A₂ med data → resampled | 0.37 → 0.26 | 0.37 → **0.26** |
| Quiet A₂ resampled | 0.013 | can invent ~0.29 if Fourier-forced + bad count split |
| Barred A₂(R) MSE (m=2) | (scalar A₂ only) | ~0.025–0.23 |

**``A_m`` is not constant in radius.** Prefer ``A_m(R)`` panels over a single
median. Count-stratified resample (disk:halo:bulge ≈ 4:2:1), not mass-weighted N.

### Crisp track (128²+, dens+moment, Fourier off) — current best

User ask: higher res, larger latent, velocity moments ≥ dens, maybe drop Fourier.
Artifacts: `runs/ml/field_maps/crisp_2026-07-24/` (`LATEST`).

| What | Result |
|------|--------|
| **Best recipe** | disk **128²**, latent **128**, `dens_w=4`, `moment_w=6`, **`a2_weight=0`**, count mix 4:2:1 |
| Bar A₂ data → resampled | 0.37 → **0.33** (was 0.26 @64²) |
| Quiet A₂ resampled | **0.016** (Fourier-on hifid invented ~0.29) |
| Short bh_c evolve | A₂ 0.33→0.31, COM drift ~0.009 kpc |
| dens-grad loss (v3) | **hurt** bars (resamp A₂ → 0.25); leave off |
| 160² / latent 192 (v2) | dens MSE↓, recon dens map A₂ ~0.11 still soft |

**What worked:** Fourier off + moment_weight ≥ dens_weight + ≥128² + count-stratified
resample. Quiet stays quiet; particle ``A_m(R)`` tracks data peaks better.

**Still open:** AE dens maps remain visually softer than data (recon dens-map A₂
median ~0.08–0.11 vs data ~0.37). Next: barred-heavy train mix / residual dens
head — not dens-grad MSE, not Fourier-forcing quiet ICs.

## Field conditional VAE (latent sampling) — washed bars

Code: `galacticsics.ml.fields.vae`, smoke `scripts/smoke_field_vae.py`, artifacts
under `runs/ml/field_maps/vae_latent_*/`. Overnight sweep:
`scripts/overnight_field_vae.sh` → `vae_latent_sweep.md`.

Goal: one global ``z`` + structural ``θ`` → field stacks → particles, so
non-equilibrium ICs can be **sampled** (not only reconstructed).  Training uses
dens+moment recon (same weights as crisp AE), weak ``β`` KL, skip-dropout, and a
prior-decode recon term so ``z ~ N(0,I)|θ`` stays informative.

### Sampling API (legacy end-to-end VAE)

```python
z = torch.randn(1, latent_dim)        # or encode → μ / interpolate μ's
fields = model.sample(theta, z=z)     # normalized multi-tower stacks
# denormalize → resample_particles_from_multiscale (count mix ≈4:2:1) → evolve
```

### Baseline vs larger latent (CPU)

See full table: [`runs/ml/field_maps/OVERNIGHT_SUMMARY.md`](../runs/ml/field_maps/OVERNIGHT_SUMMARY.md)
and [`vae_latent_sweep.md`](../runs/ml/field_maps/vae_latent_sweep.md).

| Lever | Result |
|-------|--------|
| latent 192–384 @128² | dens MSE↓ slightly; **bars stay washed** (recon A₂ ~0.02–0.04) |
| longer+lower-LR from scratch | **failed** (bar A₂ 0.017; undertrained vs 1e-4) |
| z-conditioned multi-scale skips + enc_grid=8 | best dens MSE (~0.22); bars still soft |
| warm-start + mild A₂=2.5 @3e-5 | **best so far**: bar μ **0.058**, quiet μ **0.015**; still ≪ crisp AE 0.33 |
| stronger A₂=5 | dens MSE↓ further; bar μ **regressed** to ~0.042 |

RAM: latent **512** fits batch=1 (~1.3 GB model / ~4–5 GB with data).

**Root cause:** bars live in U-Net **skips**; pure ``decode(z)`` erases them.
Prefer the feature library below.

## Frozen teacher AE feature library (**recommended sampling path**)

**Teacher** = frozen crisp multi-tower field AE (encoder features + decode), not
a human/LLM. **Students** = models that imitate teacher skips from $z$ (still
wash bars). Rank corpus by particle A₂ → stratified library (strong bars + quiet)
with bar rotations → ``z`` = PCA of pooled bottleneck → ``uniform_knn`` /
``z``-retrieve → frozen **teacher** decode.

```bash
OMP_NUM_THREADS=6 python scripts/sample_latent_ic.py --method uniform_knn --also-z-retrieve
# method sweep: scripts/sample_feature_library_v2.py
# → creative_feature_library_v2_* / creative_latent_ic_* ; CREATIVE_LATENT_SUMMARY.md
```

| Metric | Library v2 (`amplify_residual`) | `uniform_knn` | Library v1 | Overnight VAE |
|--------|----------------------------------|---------------|------------|---------------|
| Bar prior particle A₂ | **~0.36** | ~0.29–0.32 | ~0.15 | 0.058 |
| Quiet prior particle A₂ | **~0.02** | ~0.02 | ~0.03 | 0.015 |
| Continuous ``z`` (local jitter) | **~0.31** | — | — | washed |
| Quiet↔bar interp (bar end) | **~0.37** | — | ~0.17 | washed |

API: ``galacticsics.ml.fields.feature_library.TeacherFeatureLibrary``.
Skip-synth ``FrozenAECodeVAE`` remains research; long skip-L2 washes bars.

**Library index (PCA vs LDA):** keep **PCA** of the pooled bottleneck as the
continuous ``z`` for sampling. Binary bar/quiet **LDA** improves class
separation but collapses within-class morph diversity (bad for ``z_amplify``).
Prefer **PLS vs continuous A₂** or multi-class LDA on A₂ strata over binary LDA
if a supervised index is desired; see ``scripts/ablate_pca_lda_library.py`` /
``fit_index_basis``.

## Practical recommendation

1. **Shipped Part~1 product:** θ-conditioned **library retrieve+decode** —
   path-LOO neighbour → particle remass to GalactICS $f_0(\theta)$ → evolve-gate
   with **disk-COM** $A_2(R_d;t)$ (`scripts/latent_theta_gen.py`,
   `--decode-mode particle_retrieve`). Scoreboard: 906c4 MATCH, no-bulge FADE
   MATCH, 54a8 demoted. Cosmo variants = outlook only.
2. Prefer **multi-scale field slices** as the encode chart for the library
   index; keep the frozen teacher AE for $(B_k,S_k)$. End-to-end field VAE /
   free skip-synth / paint-on-bar / FFT dual-product are weaker or demoted
   baselines — see [`ml_latent_ic_methods.md`](ml_latent_ic_methods.md).
3. Always use a **shared global COM** for field maps (and one common rotation)
   — never per-component recenter. For $A_2$ diagnostics, recenter on **disk**
   mass COM.
4. Weight **⟨v⟩/σ ≥ dens** in the teacher; keep separate Fourier loss **off**
   until quiet ICs stay quiet under dens+moments alone.
5. Resample / remass with **count fractions** (≈4:2:1), not halo-dominated mass
   fractions, when comparing to corpus gates.
6. Keep **particle-set VAE** (`ml/models`) as a phase-space / mix / virial
   baseline after field paths are understood.
7. Soft **virial→1** on the particle path for evolve stability when using
   Morton-style decodes.
8. Sample barred vs quiet from **library neighbours / $A_2$ tiers** rather than
   inventing morphology from structural ``θ`` alone.
