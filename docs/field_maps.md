# Field maps: multi-slice + voxels + conditional field VAE

**Part~1 status (2026-08):** field maps remain the **encode chart** for the
θ-conditioned latent library, but the **shipped IC product** is
**retrieve+decode** (particle remass to $f_0(\theta)$), not AE resample /
amplify paint / FFT dual-product. Scoreboard: 906c4 MATCH, no-bulge FADE
MATCH, 54a8 demoted (disk-COM $A_2$). See
[`ml_findings.md`](ml_findings.md) and
[`papers/mnras_noneq_ics/`](../papers/mnras_noneq_ics/).

**Encode / morphology path** for the latent library (and historical AE
recon work). Parallel to the particle-set Morton baseline
([`morton_generative.md`](morton_generative.md)). Older campaign encoder
notes: [`archive/ml_encoder_strategy.md`](archive/ml_encoder_strategy.md).
Findings: [`ml_findings.md`](ml_findings.md). Math-heavy chronology:
[`ml_latent_ic_methods.md`](ml_latent_ic_methods.md).

**Dynamical consistency / learnable charts** (Proposition A + Grönwall sketch):
[`dynamical_consistency_charts.md`](dynamical_consistency_charts.md).
DF match without a 5D/6D mesh (kinematics / OT / resample options):
[`df_matching_alternatives.md`](df_matching_alternatives.md).

## Idea

Represent each snapshot as **fields**, not (only) as a particle set:

1. **Vertical-slice channel images** — dens + ⟨v⟩ + σ (optional β / Φ) on many
   ``z`` slabs, stacked as channels, **per component**.
2. **Voxel grids** — anisotropic 3-D dens + ⟨v⟩ + σ per component.
3. **Potential** — cheap Plummer Φ on the same grids (swap for tree Φ later).
4. **Encode → (optional) reconstruct → resample** — multi-tower U-Net AE recovers
   the maps; historical path drew particles from reconstructed dens. **Part~1
   product instead loads retrieved neighbour particles and remasses to
   $f_0(\theta)$.**
5. **Conditional field VAE** — research baseline; bottleneck → global latent
   ``μ,logσ² → z``; free ``decode(z)`` washes bars (not the shipped path).

Random in-plane rotation is a natural train-time augmentation (**after** shared
centering — one φ for the whole snapshot).

## Generative objective (Part~1: θ-conditioned retrieve+decode)

**Teacher** = frozen multi-tower **field autoencoder** (not a human / LLM):
encoder bottlenecks + skips index the library; decoder is optional for
ablations. **Students** that synthesise skips from $z$ currently wash bars.

**Shipped path (Part~1):**

1. Freeze the multitower teacher AE; deposit dumps → field maps $X_k$ with
   **shared global COM**.
2. Pool bottlenecks → PCA code $z$ ($n_{\mathrm{pc}}=64$); store
   $(z,A_2(R_d),\theta)$ with **disk-COM** particle $A_2$.
3. At fixed $\theta$, retrieve a path-LOO / $A_2$-tier neighbour (no eval
   particle copy) → **particle retrieve + remass** to GalactICS $f_0(\theta)$.
4. Evolve-gate $A_2(R_d;t)$ under `gpu_bh` (MATCH / FADE MATCH / OOD).

```bash
OMP_NUM_THREADS=2 .venv/bin/python scripts/latent_theta_gen.py \
  --decode-mode particle_retrieve --loo-exclusion path \
  --evolve-gyr 0.5 --force gpu_bh
```

**Demoted relative to retrieve+decode** (keep as controls / history):

| Path | Role now |
|------|----------|
| `amplify_knn_hybrid` / `z_amplify` feature morph | Strong $A_2$ demos; not absolute-DF remass product |
| Soft AE dens+moment resample | Ablation only |
| FFT morph / Am-match dual product | Teacher training / diagnostics |
| Paint-on-bar / `residual_f0` | Demoted (radial lock to quiet $f_0$) |
| End-to-end field VAE / free skip-synth | Washes bars |

Historical marathon amplify numbers (for context only):
``amplify_knn_hybrid`` → bar particle A₂ ≈ **0.418±0.011**
([`CREATIVE_LATENT_SUMMARY.md`](../runs/ml/field_maps/CREATIVE_LATENT_SUMMARY.md)).
Part~1 claims use the retrieve+decode scoreboard in [`ml_findings.md`](ml_findings.md).

### Library index: PCA vs LDA

The library compresses pooled teacher bottlenecks to a retrieval ``z``. **Keep PCA**
as the default continuous index:

| Index | Role | Verdict |
|-------|------|---------|
| **PCA** | Unsupervised variance axes | **Default** — preserves within-class morph diversity for continuous IC sampling (`z_amplify`, amplify hybrid). |
| Binary **LDA** (bar/quiet) | Fisher class separation | Better bar/quiet pools, but **collapses within-class** diversity — do not use as the sole generative ``z``. |
| Whitened PCA / **PLS vs A₂** / multi-class LDA on A₂ strata | Supervised compromises | Prefer these over binary LDA if a label-aware metric is needed; still pad with residual PCA. |

Ablation: `scripts/ablate_pca_lda_library.py` → `runs/ml/field_maps/pca_lda_ablate_*/SUMMARY.md`
(`fit_index_basis` in `feature_library.py`).

CPU RAM (batch=1, 128²): library build ~few GB; hybrid student ~5 GB with data.

## Shared centering (critical)

**Do not** center disk / bulge / halo on their own COMs before binning.
That misaligns components relative to each other (bar vs bulge, lopsided halo).

Correct pipeline:

1. Compute the **mass-weighted global COM** of *all* particles once
   (`prepare_shared_frame` in `ml/fields/frame.py`).
2. Subtract that COM / VCOM from the full state.
3. Apply **one** common in-plane rotation to all particles (train-time augment).
4. Bin each component — and Φ — on that **same origin**; only FOV / resolution
   differ per component.

Binning APIs never recenter.  Dataset loaders and smoke panels call
`prepare_shared_frame` once.  Regression:
`tests/test_field_maps.py::test_shared_centering_preserves_component_offsets`.

## Per-component scale differences

Disk, bulge, and halo live on very different scales.  A **single shared FOV**
either wastes resolution on the bulge or truncates the halo.  Defaults:

| Component | Slice FOV (``r_max``, ``z_max``) | ``n_pix`` × ``n_z`` | Notes |
|-----------|----------------------------------|---------------------|-------|
| bulge | 4 × 4 kpc | **48** × 8 | fine, near-cubic |
| disk | 12 × 1.5 kpc | **64** × 10 | midplane-focused Δz (sinh) |
| halo | 40 × 40 kpc | **32** × 4 | coarse, large box |

Presets in `MultiScaleSliceConfig`:

- `smoke_defaults` — table above (+ σ moments)
- `baseline_32_defaults` — original 32² dens+⟨v⟩ smoke
- `progressive_defaults(disk_n_pix=96|128)` — hifid step (finer bulge, more z-slices, better halo angular sampling)
- `cusp_bulge_defaults` — disk/halo match FFT-long progressive; bulge **128² × 32**
  (Δx≈0.0625 kpc, Δz≈0.25 kpc) for cusp recovery; warm-start disk/halo only via
  `load_compatible_towers` / `--init-compatible-towers disk,halo`

**Hybrid IC (cusp fallback).** When the bulge tower still under-resolves the
spheroidal cusp, stitch GalactICS bulge particles onto FFT-recon disk/halo with
`stitch_retained_components` / `--with-hybrid-bulge` on
`scripts/evolve_component_slices.py`.

Voxel smoke defaults (anisotropic for the disk):

| Component | ``r_xy``, ``r_z`` | ``n_xy`` × ``n_z`` |
|-----------|-------------------|--------------------|
| bulge | 4, 4 kpc | 32 × 32 |
| disk | 12, 1.5 kpc | 48 × 16 |
| halo | 40, 40 kpc | 28 × 28 |

**Fusion.** `MultiTowerSliceAE` keeps a native-resolution U-Net (or shallow CNN)
per component.  Optional cross-tower attention mixes pooled encoder cues without
resampling FOVs.  Alternatively `fuse_slice_maps_to_common` bilinear-rescales to
one canvas for a single Conv2d stack.

## Channel layout

Default `moment_set="disp"` (7 channels per slab / voxel):

```
dens, vx, vy, vz, sx, sy, sz
```

- `sx,sy,sz` = mass-weighted velocity dispersion.
- `moment_set="base"` → dens + ⟨v⟩ only (4).
- `moment_set="full"` → adds `beta ≈ 1 − 2σ_z²/(σ_x²+σ_y²)`.

Slice stack for one component: shape `(n_z · n_mom, n_pix, n_pix)`.
Voxel stack: `(n_mom, n_z, n_xy, n_xy)`.

Normalization: dens/σ → `log1p`; mean vel / β → `tanh`; optional Φ → affine MAD.

## Architecture

- **`SliceUNet` / `MultiTowerSliceAE`**: recon-only baseline (crisp dens+moment).
  Deeper dens/moment heads (3×3 stacks, not 1×1 alone); optional dens-gradient
  match (usually **off** — did not help bars). ``deep_heads=False`` loads
  crisp_2026-07-24 (1×1 heads, best particle A₂).
- **`FrozenAECodeVAE`** (research student prior): frozen **teacher** AE + single
  ``z`` skip synthesizer + optional morph ``A₂`` / RealNVP prior (washes bars).
- **`MultiTowerSliceVAE`**: encoder towers → fused ``μ,logσ²`` → global ``z``;
  FiLM-conditioned decoders on ``(z, θ)``; prior-path skips from ``(z,θ)`` still
  wash bars (overnight best bar μ ≈ 0.06). Optional mild disk ``A_m(R)`` via
  ``--a2-weight`` (≳5 can invent quiet structure — prefer ~2–3).
- **Loss (crisp track)**: dens + moment MSE with ``moment_weight ≥ dens_weight``;
  **Fourier / ``A_m`` loss off by default**. Hybrid adds skip distill + ``β·KL``.
- **Recommended generative path**: stratified **teacher** feature library
  (`TeacherFeatureLibrary` + amplify / ``z_amplify``), not end-to-end VAE.

## Module roles (fields vs models)

| Package | Role |
|---------|------|
| `ml/fields/` | **Morphology path** — slice/voxel maps, recon AE, **field VAE**, resample |
| `ml/models/` | **Particle baseline** — Morton set VAE + AR transformer (weaker bars) |
| `ml/conditioning.py` | Shared structural ``θ`` keys for both VAEs |
| `ml/morton/` | Snapshot index + particle tokenization (feeds both tracks) |
| `ml/profiles/` | Soft particle auxiliaries (Fourier, virial) for the set VAE |

## Code map

| Module | Role |
|--------|------|
| `ml/fields/frame.py` | Shared global COM + common rotation |
| `ml/fields/binning.py` | Multi-scale slice + voxel deposit (incl. σ / β) |
| `ml/fields/potential.py` | Subsample Plummer Φ on same shared frame |
| `ml/fields/normalize.py` | log1p dens/σ / tanh vel / Φ affine |
| `ml/fields/autoencoder.py` | Shallow AE + U-Net + multi-tower fusion (recon) |
| `ml/fields/latent_code.py` | Frozen AE + skip-distill **student** code VAE (research) |
| `ml/fields/vae.py` | End-to-end conditional field VAE (weaker bars) |
| `ml/fields/dataset.py` | Corpus → multi-scale field batches (+ ``θ``) |
| `ml/fields/resample.py` | Particles from reconstructed dens (+ σ) |
| `scripts/smoke_field_maps.py` | Recon AE smoke + verdict |
| `scripts/sample_latent_ic.py` | **Recommended IC sampling** (stratified library + ``z``) |
| `scripts/sample_feature_library_v2.py` | Library method ablation (knn / amplify / PCA / KDE) |
| `scripts/ablate_pca_lda_library.py` | PCA vs LDA/PLS index ablation for library ``z`` |
| `scripts/sample_teacher_feature_library.py` | Legacy v1 stride library (~0.15 bar A₂) |
| `ml/fields/feature_library.py` | ``TeacherFeatureLibrary`` + ``fit_index_basis`` |
| `scripts/train_latent_code_prior.py` | Skip-distill research prior |
| `scripts/sample_field_latent.py` | Sample from skip-synth ckpt |
| `scripts/smoke_field_vae.py` | Legacy end-to-end field VAE train |

## Smoke

```bash
. .venv/bin/activate
# Crisp track (default): 128² disk, dens+moment-heavy, Fourier OFF, count-stratified resample
OMP_NUM_THREADS=6 python scripts/smoke_field_maps.py --with-evolve
# → runs/ml/field_maps/crisp_YYYY-MM-DD/ ; pointer: runs/ml/field_maps/LATEST

# Optional Fourier ablate (can invent structure on quiet ICs)
OMP_NUM_THREADS=6 python scripts/smoke_field_maps.py --a2-weight 2.5 --with-evolve

# Near-prod 64² (previous hires)
OMP_NUM_THREADS=8 python scripts/smoke_field_maps.py --preset smoke --disk-n-pix 64

# Reproduce original 32² dens+⟨v⟩ baseline
OMP_NUM_THREADS=8 python scripts/smoke_field_maps.py --preset baseline32 --moment-set base --arch shallow
```

Requires a Morton corpus manifest at
`runs/ml/smoke_morton_vae_cpu/manifest_mw_morton_corpus_v2.json` and matching
snapshot dumps under `runs/mw_morton_corpus_v2/`.

CPU smoke trains the AE **without** Φ by default; `plummer_potential_multiscale`
still evaluates Φ on each component's native FOV when `--include-potential` is set.
Prefer **CPU + OpenMP** (`OMP_NUM_THREADS≈6`) while a corpus campaign may own the GPU.

### Measured progression (barred `54a8` late + quiet IC)

| Run | Disk | Latent | Loss | Bar A₂ data→resamp | Quiet A₂ resamp | Notes |
|-----|------|--------|------|--------------------|-----------------|-------|
| baseline 32² | 32² | — | dens>mom | 0.37 → 0.12 | 0.036 | washed |
| hires_2026-07-24 | 64² | 64 | dens_w=6 + scalar A₂ | 0.37 → **0.26** | 0.013 | better |
| hifid_2026-07-24 | 96² | 64 | dens + Fourier A₁–₃(R) | 0.37 → 0.26 | **0.29†** | †Fourier invents structure on quiet; mass-starved disk resample |
| **crisp_2026-07-24** | **128²** | **128** | dens_w=4, mom_w=6, **Fourier off** | 0.37 → **0.33** | **0.016** | **best particle A₂**; quiet stays quiet; evolve A₂ 0.33→0.31 |
| crisp_v2 | 160² | 192 | dens=mom=8, Fourier off | 0.37 → 0.30 | 0.014 | dens MSE↓; recon dens map A₂ still ~0.11 |
| crisp_v3 | 128² | 160 | + dens ∇ match | 0.37 → 0.25 | 0.017 | dens-grad **hurt** bars |

Artifacts: `runs/ml/field_maps/crisp_2026-07-24/` (pointer `LATEST`). Prefer visual
field panels + ``A_m(R)`` over a single median. Resample with **count mix**
disk:halo:bulge ≈ 4:2:1 (true masses for particle mass only — mass-weighted N
starves the disk). Shared global COM required.

**Still soft on AE dens maps:** particle A₂ and ``A_m(R)`` improved a lot under
dens+moment / Fourier-off, but midplane dens recon remains blurrier than data
and ``A_m(R)`` peak *heights* under-predict even when peak *radii* match.
Quiet-gated Fourier on linearized dens + scale-free axisym dens residual
(`scripts/smoke_field_maps.py --init-checkpoint …/crisp… --dens-resid-weight
--a2-weight`, artifacts `am_match_ft_*`) targets that amplitude gap without
inventing quiet bars. Avoid dens-grad MSE (hurt bars in crisp_v3).

**Spatial FFT morphology (residual dens):** full-map ``|FFT(R)|`` match on
axisym-subtracted dens ``R=Σ−⟨Σ⟩_φ`` with mid/high-``k`` boost and residual-RMS
quiet gate (`--fft-weight`, optional `--fft-lambda-phase` / `--fft-quiet-gate` /
`--fft-k-floor`; artifacts `fft_morph_ft_*`). Continues:
`fft_morph_ft` (12 ep) → `fft_morph_ft_converged_2026-07-25` (+80 ep, best
~1.08) → **`fft_morph_ft_long_2026-07-25`** (further continue, LR drop
`7.5e-5`, patience 32, until plateau). Exposes a real **visual ↔ metric
tension**: AE / midplane dens look **crisper** than `am_match_ft` / crisp
(best dens MSE; midplane residual ∇ ≫ am_match), but particle A₂ and Am(R)
MSE still trail — FFT restores edge / spiral contrast, Am-match restores bar
amplitude.

**HOLD** on default train weight (`fft_weight=0` for Am-led training), but
**prefer the long FFT ckpt as the sampling / feature-library teacher** (best
dens-map recon for library encoding; user priority: recon sharpness >
Am MSE horse race). Prefer `am_match_ft` only when particle Am(R) MSE is the
explicit goal. Optional blend: low `fft_weight` (~0.3–0.5) + strong
`a2_weight`. Keep `moment_w ≥ dens_w`; leave dens-grad off.

| Role | Prefer |
|------|--------|
| **Sampling teacher / feature library** | `fft_morph_ft_long_2026-07-25` (fallback: `…_converged_…`) |
| Particle Am / Am(R) MSE horse race | `am_match_ft_2026-07-25` |
| Am-led *training* default | `fft_weight=0` |
