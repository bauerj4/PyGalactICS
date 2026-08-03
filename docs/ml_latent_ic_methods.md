# Latent IC methods: algorithmic history, math, and pathways

Math-heavy chronology of approaches tried for **non-equilibrium galaxy IC
sampling** in this repo. Companion narratives:
[`field_maps.md`](field_maps.md) (encode chart),
[`ml_findings.md`](ml_findings.md) (short verdicts + Part~1 scoreboard),
[`morton_generative.md`](morton_generative.md) (particle-set baseline),
[`df_matching_alternatives.md`](df_matching_alternatives.md) (DF match without a 6D mesh).
Marathon numbers: [`runs/ml/field_maps/MARATHON_6H.md`](../runs/ml/field_maps/MARATHON_6H.md),
[`CREATIVE_LATENT_SUMMARY.md`](../runs/ml/field_maps/CREATIVE_LATENT_SUMMARY.md).

**MNRAS Part~1:** [`papers/mnras_noneq_ics/`](../papers/mnras_noneq_ics/)
(`mnras_noneq_ics.tex` / `.pdf`).

**Current best (Part~1 product):** θ-conditioned **library retrieve+decode** —
path-LOO particle remass to GalactICS $f_0(\theta)$; disk-COM $A_2$ evolve
gates. Scoreboard: **906c4 MATCH**, **no-bulge FADE MATCH**, **54a8 demoted**.
Bar sweep + OOD 3/0/0. Scripts: `latent_theta_{gen,evolve_suite,ood,bar_sweep}.py`.
Claim = *slightly non-equilibrium* ICs; cosmo variants = outlook only.

**Demoted vs retrieve+decode:** paint-on-bar / `residual_f0`, FFT dual-product
as primary, free skip-synth, data-disk graft, amplify-as-headline (amplify
remains a strong historical $A_2$ demo: `amplify_knn_hybrid` ≈ 0.418±0.011).

Frozen teacher AE: multitower SliceUNet (FFT morph long / crisp lineage).

### Teacher vs student (terminology)

In this project **teacher** does **not** mean a human or an LLM. It means the
**frozen high-quality crisp multi-tower field autoencoder**
(`runs/ml/field_maps/crisp_2026-07-24/`): its encoder produces bottleneck + U-Net
**skip** tensors (the morphology carriers), and its decoder reconstructs dens /
moment maps. The **feature library** stores those frozen encoder features;
sampling blends / amplifies them and re-decodes with the same frozen AE.

**Students** are models that try to *imitate* those features from a compact code
$z$ (end-to-end field VAEs, skip-synth `FrozenAECodeVAE`, morph-conditioned
flow priors). When a student mean-collapses skips, bars wash out — that is why
the recommended IC path keeps the teacher frozen and morphs library features
instead of training a free skip synthesizer.

---

## 0. Shared notation

| Symbol | Meaning |
|--------|---------|
| $N$ | Particle count (train subset or resample size) |
| $c\in\{0,1,2\}$ | Component: disk / halo / bulge |
| $\theta$ | Structural conditioning vector (shared keys in `ml/conditioning.py`) |
| $z$ | Global latent (VAE code or PCA of pooled AE bottlenecks) |
| $A_m(R)$ | Azimuthal Fourier amplitude $\lvert a_m\rvert/a_0$ in radial bin $R$ |
| $\mathrm{A}_2$ | Median over bins of $A_2(R)$ on disk particles (or dens map) |
| $Q=2K/\lvert W\rvert$ | Softened pairwise virial ratio |

**Particle A₂ (evaluation).** On disk particles with $\lvert z\rvert\le 0.5\,\mathrm{kpc}$,
$R\le 12\,\mathrm{kpc}$,


$$
a_m(R_b)=\sum_{i\in b} m_i\,e^{im\phi_i},\qquad
A_m(R_b)=\frac{\lvert a_m(R_b)\rvert}{\lvert a_0(R_b)\rvert},
\qquad
\mathrm{A}_2=\mathrm{median}_b\,A_2(R_b).
$$


Implemented in `ntropy.analysis.disk_density.disk_azimuthal_fourier`
(default: recenter supplied particles on their mass-weighted COM before annular
Fourier — disk-COM when callers pass disk-only).

**Shared centering (required for all field paths).** One mass-weighted global COM
of *all* particles; never per-component recenter. Optional one common in-plane
rotation $\phi$ after centering (`ml/fields/frame.prepare_shared_frame`).
Particle $A_2(R)$ diagnostics are separate: they recenter on **disk** mass COM
so cylindrical $(R,\phi)$ are about the disk centre (halo-dominated global COM
can leave the disk $\sim$kpc off-origin and inflate $A_2$).

---

## 1. Particle-set Morton VAE (baseline)

**Code:** `ml/models/sequence_vae.py`, `ml/morton/`, `scripts/train_morton_vae.py`.  
**Docs:** [`morton_generative.md`](morton_generative.md), virial
[`runs/ml/vae_virial_ablation/NOTE.md`](../runs/ml/vae_virial_ablation/NOTE.md).

### 1.1 Tokenization

From a snapshot, draw a stratified subset of size $N$, Morton-sort, form tokens
$(c,\Delta m,x,v)$ where $x\in\mathbb{R}^3$ is absolute position in Morton
order and $\Delta m$ is the Morton-key increment (AR feature).

### 1.2 Encoder / decoder

- **Encoder:** phase-space MLP → per-component mean/max pools → $(\mu,\log\sigma^2)$;
  $z=\mu+\sigma\odot\varepsilon$, $\varepsilon\sim\mathcal{N}(0,I)$.
- **Classifier:** MLP on $(x,v)$ → per-particle logits (hard CE vs $c$).
- **Decoder:** sample stratified $c\sim\mathrm{Mix}(\theta)$, then geometric
  bases $x,v\mid c,\theta,z$ + residual. Mix prior supervised by **fraction MSE
  only** (no global soft CE).

### 1.3 Losses (detailed)

**Per-particle CE**


$$
\mathcal{L}_{\mathrm{CE}}=
\mathrm{CE}(\mathrm{cls}(x,v),c)
+\tfrac12\mathrm{CE}(\mathrm{cls}(\hat x,\hat v),c).
$$


**Mix MSE** (easy prior)


$$
\mathcal{L}_{\mathrm{mix}}=\bigl\|
\mathrm{softmax}(\ell_{\mathrm{mix}})-\overline{\mathrm{onehot}(c)}
\bigr\|_2^2.
$$


**Chamfer** (permutation-invariant recon; optionally subsampled to $N_{\max}$)


$$
\mathrm{Ch}(A,B)=\frac1{\lvert A\rvert}\sum_{a\in A}\min_{b\in B}\lVert a-b\rVert_2
+\frac1{\lvert B\rvert}\sum_{b\in B}\min_{a\in A}\lVert a-b\rVert_2.
$$


Total recon blends Chamfer on positions (+ $0.25\times$ on velocities) with a
weak index-aligned MSE and per-component Chamfer.

**KL**


$$
\mathrm{KL}=\tfrac12\sum_j\bigl(
\mu_j^2+\sigma_j^2-\log\sigma_j^2-1
\bigr)
\quad\text{(mean over batch / dims as in code)}.
$$


**Soft profiles** (`ml/profiles.py`): $\Sigma(R)$, $\rho(r)$, $\langle v_\phi\rangle(R)$,
soft maps, and soft Fourier $A_m/A_0$ on the reconstruction (weights
$\lambda_\Sigma,\lambda_\rho,\ldots$).

**Virial** (stratified Plummer subsample of size $n_{\mathrm{sub}}$)


$$
K=\tfrac12\sum_i m_i\lVert v_i\rVert^2,\qquad
W=-\tfrac12\sum_{i\neq j}\frac{m_i m_j}{\sqrt{r_{ij}^2+\varepsilon^2}},\qquad
Q=\frac{2K}{\lvert W\rvert}.
$$


Prefer soft pull $Q_{\mathrm{pred}}\to 1$ (+ KE / COM match) over matching
equal-mass data ratios ($\sim 10$–$20$), which teach unbound BH clouds.

### 1.4 What worked / failed

| Verdict | Detail |
|---------|--------|
| **Failed** | Index-wise MSE on Morton-ordered decode → collapse. |
| **Failed** | Global `mix_head(z,θ)` soft CE → stuck near-uniform ($\approx\log 28$ nats). |
| **Helped** | Chamfer + geometric profiles/maps/Fourier → morphology appears. |
| **Helped** | Per-particle CE $\to$ $\sim 0.23$–$0.32$, $\sim 91$–$95\%$ component acc. |
| **Mixed** | Virial$\to 1$: better COM / outer $r_{90}$ on evolve; can **suppress** strong bars. |

Particle-set VAE remains a **phase-space / mix / virial** baseline; face-on bars
are weaker than the field path.

---

## 2. Multi-scale field maps + U-Net AE (recon teacher AE)

**Code:** `ml/fields/{frame,binning,normalize,autoencoder,resample}.py`,
`scripts/smoke_field_maps.py`.  
**Best recon:** `runs/ml/field_maps/crisp_2026-07-24/`.

### 2.1 Binning / channels

Per component $c$, deposit particles onto a native FOV (shared origin):

| Component | Typical FOV | Grid (crisp) |
|-----------|-------------|--------------|
| bulge | $4\times 4\,\mathrm{kpc}$ | $84^2\times 12\,z$ |
| disk | $12\times 1.5\,\mathrm{kpc}$ | $128^2\times 14\,z$ (midplane $\sinh$ $z$) |
| halo | $40\times 40\,\mathrm{kpc}$ | $42^2\times 8\,z$ |

Default `moment_set="disp"` — $n_{\mathrm{mom}}=7$ channels per slab:


$$
\bigl(\rho,\;\langle v_x\rangle,\;\langle v_y\rangle,\;\langle v_z\rangle,\;
\sigma_x,\;\sigma_y,\;\sigma_z\bigr),
$$


with mass-weighted means/dispersions in each pixel. Slice stack shape
$(n_z\cdot n_{\mathrm{mom}},\,n_{\mathrm{pix}},\,n_{\mathrm{pix}})$.

**Normalization:** dens/$\sigma\to\log(1+\cdot)$; mean vel $\to\tanh$; optional $\Phi$ MAD-affine.

### 2.2 U-Net multi-tower AE

`MultiTowerSliceAE`: one U-Net (or shallow CNN) per component; optional
cross-tower attention on pooled encoder cues. Separate dens / moment heads.
Bottleneck spatial feature maps + **skip** tensors are the morphology carriers.

### 2.3 Reconstruction loss

Let $D$ = dens channel indices, $M$ = remaining moment channels. Per tower:


$$
\mathcal{L}_{\mathrm{tower}}=
w_\rho\,\mathrm{MSE}(\hat\rho,\rho)
+w_m\,\mathrm{MSE}(\hat M,M)
+w_{\nabla}\,\mathrm{MSE}(\nabla\hat\rho,\nabla\rho)
+w_F\sum_{m\in\{1,2,3\}}
\Bigl(
\lVert\hat A_m-A_m\rVert^2
+\lambda_\phi\bigl(\lVert\hat c_m-c_m\rVert^2+\lVert\hat s_m-s_m\rVert^2\bigr)
\Bigr),
$$


with soft radial Fourier from dens maps (`soft_am_radial_from_dens_maps`):


$$
a_0(R_b)=\sum_{xy} \rho_{xy}\,w_{xy,b},\quad
c_m=\frac{\sum\rho\,w\cos(m\phi)}{a_0},\quad
s_m=\frac{\sum\rho\,w\sin(m\phi)}{a_0},\quad
A_m=\sqrt{c_m^2+s_m^2}.
$$


Multi-tower sum with component weights (disk heavy, e.g. $2.5:1.0:0.4$).

**Crisp recipe:** $w_\rho=4$, $w_m=6$, $w_F=0$, $w_\nabla=0$, disk $128^2$,
latent channels $128$.

### 2.4 Resample

Draw particles from reconstructed dens (+ cell $\langle v\rangle$, optional
$\langle v\rangle+\sigma\cdot\mathcal{N}(0,1)$). **Count mix**
disk∶halo∶bulge $\approx 4:2:1$ (not mass-weighted $N$ — that starves the disk).

### 2.5 Progression (barred data $\mathrm{A}_2\approx 0.37$)

| Track | Disk | Bar resamp $\mathrm{A}_2$ | Quiet | Note |
|-------|------|----------------------------|-------|------|
| baseline 32² | 32² | 0.12 | 0.036 | washed |
| hires | 64² | 0.26 | 0.013 | U-Net + $\sigma$ |
| hifid | 96² + $A_m(R)$ | 0.26 | up to ~0.29† | †Fourier invents quiet structure |
| **crisp** | **128²**, Fourier off | **0.33** | **0.016** | best recon teacher |
| crisp_v2 160² | 160² | 0.30 | 0.014 | dens MSE↓, map A₂ still soft |
| crisp_v3 + dens ∇ | 128² | 0.25 | 0.017 | dens-grad **hurt** bars |

**Still soft:** recon *dens-map* median $A_2\sim 0.08$–$0.11$ vs particle $0.33$.

---

## 3. End-to-end conditional field VAE

**Code:** `ml/fields/vae.py`, `scripts/smoke_field_vae.py`.  
**Overnight:** [`OVERNIGHT_SUMMARY.md`](../runs/ml/field_maps/OVERNIGHT_SUMMARY.md).

### 3.1 Formulation

Encoder towers → fused $\mu,\log\sigma^2$ → $z$. Decode with FiLM on
$(z,\theta)$. Training loss:


$$
\mathcal{L}=\mathcal{L}_{\mathrm{recon}}^{\text{prior-path}}(\mathrm{decode}(z,\theta);\,x)
+\beta\,\mathrm{KL}(q(z\mid x)\Vert\mathcal{N}(0,I)),
$$


where prior-path decode **drops encoder skips** (skip-dropout in train). Optional
mild $A_m(R)$ via `a2_weight` $\sim 2$–$3$ ($\ge 5$ can invent quiet structure).

### 3.2 Sampling API

```python
z = torch.randn(1, latent_dim)
fields = model.sample(theta, z=z)  # normalized stacks
# denormalize → resample_particles_from_multiscale (4:2:1) → evolve
```

### 3.3 Verdict

| Lever | Bar μ | Quiet μ |
|-------|-------|---------|
| Best overnight (z256 + mild A₂ FT) | **0.058** | 0.015 |
| Larger $z$ alone (192–384) | dens MSE↓ | bars still washed |
| Stronger A₂=5 | **regressed** (~0.042) | — |

**Root cause:** bars live in U-Net **skips**; pure $\mathrm{decode}(z)$ erases them.
Not scientifically usable for barred ICs vs crisp AE $\mathrm{A}_2\sim 0.33$.

---

## 4. Skip-synth / FrozenAECodeVAE (research)

**Code:** `ml/fields/latent_code.py`, `scripts/train_latent_code_prior.py`.

Idea: freeze **teacher** AE; train a **student** code model to synthesize
**skips** from a single $z$ (+ optional morph $A_2$ / RealNVP prior), distill via
skip-L2.

**Verdict:** long skip-L2 / det+morph+flow washes bars (marathon skip-synth
$\mathrm{A}_2\lesssim 0.08$). Bars are high-frequency skip residual that a
global code mean-collapses. Kept as research; not the recommended IC path.

---

## 5. Frozen teacher feature library (recommended generative path)

**Code:** `ml/fields/feature_library.py`,
`scripts/sample_latent_ic.py`, `scripts/sample_feature_library_v2.py`.  
**Summary:** [`CREATIVE_LATENT_SUMMARY.md`](../runs/ml/field_maps/CREATIVE_LATENT_SUMMARY.md).

The library is built from the **teacher AE** (frozen crisp multi-tower field
autoencoder): encode corpus snaps → store bottleneck+skips → PCA-pool
bottlenecks to $z$ → sample by retrieve / amplify → **teacher decode**. No
student skip-synth is required for the recommended recipes.

### 5.0 Dimensional walkthrough (crisp defaults)

Object shapes for the production **frozen teacher** path
(`crisp_2026-07-24`, `SliceUNet`, `base_channels` $c=48$, three stride-2
downs → $H/8$). Disk stack $C_{\mathrm{disk}}=n_z\,n_{\mathrm{mom}}=14\times 7=98$.
Tower index $k\in\{\mathrm{disk},\mathrm{bulge},\mathrm{halo}\}$.

**Compact flow**


$$
X_k
\xrightarrow{\mathrm{Enc}}
\bigl(B_k,\,S_k\bigr)
\xrightarrow[\mathrm{pool}\,g\times g]{\mathrm{flatten+concat}}
u
\xrightarrow{\mathrm{PCA}}
z
\quad\big\Vert\quad
F\!\leftarrow\!\mathrm{retrieve/amplify}
\xrightarrow{\mathrm{Dec}}
\hat X_k
\xrightarrow{\mathrm{denorm}}
\mathrm{resample}
\xrightarrow{\mathrm{evolve}}.
$$


$z$ indexes the library; the working generative path is **retrieve / amplify
$F$ + frozen decode**, not end-to-end skip synthesis from $z$.

| Step | Object | Typical shape (disk; $c=48$) |
|------|--------|------------------------------|
| 0 | Input $X_k$ | $\mathbb{R}^{C_k\times H_k\times W_k}$; disk $\mathbb{R}^{98\times 128\times 128}$ |
| 1 | Bottleneck $B_k$, skips $S_k=(E_1,E_2,E_3)$ | see below |
| 2 | Pooled concat $u$ | $\mathbb{R}^{d_u}$, $d_u=\sum_k 4c\,g^2$ ($g=4$ → $3\times 192\times 16=9216$) |
| 3 | PCA code $z$; stored $F=(B,S)$; $F_{\mathrm{quiet}}$ | $z\in\mathbb{R}^{n_{\mathrm{pc}}}$ (default $n_{\mathrm{pc}}=64$) |
| 4 | Blended / amplified $F(\alpha)$ | same tensor layout as $F$ |
| 5 | Decode $\hat X_k$ | same as $X_k$ |
| 6 | Denorm → particles → evolve | count mix ≈ 4∶2∶1 |

#### 0. Inputs

Per tower, normalized dens+moment slice stack:


$$
X_k\in\mathbb{R}^{C_k\times H_k\times W_k},\qquad
C_k=n_{z,k}\,n_{\mathrm{mom}},\quad n_{\mathrm{mom}}=7.
$$


| Tower | $n_z$ | $H=W$ | $C_k$ |
|-------|-------|-------|-------|
| disk | 14 | 128 | **98** |
| bulge | 12 | 84 | 84 |
| halo | 8 | 42 | 56 |

#### 1. Teacher encode → $B_k$ + skips $S_k$

Frozen **full** teacher AE (Enc+Dec; not encoder-only). `encode_with_skips`:


$$
\begin{aligned}
E_1&=\mathrm{enc}_1(X_k)\in\mathbb{R}^{c\times H\times W},\\
E_2&=\mathrm{enc}_2(\downarrow E_1)\in\mathbb{R}^{2c\times H/2\times W/2},\\
E_3&=\mathrm{enc}_3(\downarrow E_2)\in\mathbb{R}^{4c\times H/4\times W/4},\\
B_k&=\mathrm{bottleneck}(\downarrow E_3)\in\mathbb{R}^{4c\times H/8\times W/8},\\
S_k&=(E_1,E_2,E_3).
\end{aligned}
$$


Disk ($H=W=128$, $c=48$): $E_1\in\mathbb{R}^{48\times 128\times 128}$,
$E_2\in\mathbb{R}^{96\times 64\times 64}$, $E_3\in\mathbb{R}^{192\times 32\times 32}$,
$B\in\mathbb{R}^{192\times 16\times 16}$.

Bars live **mostly in the skips** $S_k$ (high-frequency residual); the bottleneck
is a coarse morphology cue used for retrieval.

#### 2. AdaptiveAvgPool → flatten → concat towers

Only bottlenecks enter the continuous index (default $g=4$):


$$
\tilde B_k=\mathrm{AdaptiveAvgPool}_{g\times g}(B_k)
\in\mathbb{R}^{4c\times g\times g},\qquad
u=\bigoplus_k\mathrm{vec}(\tilde B_k)\in\mathbb{R}^{d_u}.
$$


With three towers, $c=48$, $g=4$: $d_u=3\cdot(4c)\,g^2=9216$.

#### 3. PCA index; store full features; quiet mean

$$
z=(u-\bar u)\,W\in\mathbb{R}^{n_{\mathrm{pc}}},\qquad
W=\text{top $n_{\mathrm{pc}}$ right singular vectors of centered }\{u_i\}.
$$


**Important:** $z$ is PCA of **pooled bottlenecks**, **not** flattened dens /
moment profiles. Per library member store the full feature pack
$F_i=\{(B_k,S_k)\}_k$. Quiet-mean features:


$$
F_{\mathrm{quiet}}=\frac1{\lvert\mathcal{Q}\rvert}\sum_{i\in\mathcal{Q}} F_i.
$$


#### 4. Sample: blend / residual amplify

Linear ops on every tensor in $F$ (bottleneck **and** each skip):


$$
F=\sum_i w_i F_i
\quad(\textstyle\sum w=1),\qquad
F(\alpha)=F_{\mathrm{base}}+\alpha\,(F_{\mathrm{target}}-F_{\mathrm{base}}).
$$


Recommended recipes (`amplify_knn_hybrid`, `z_amplify`, …) retrieve in $z$-space
then amplify skip residual vs $F_{\mathrm{quiet}}$ (or a knn blend). Quiet draws
never use $\alpha>1$.

#### 5. Frozen teacher decode → $\hat X_k$

$$
\hat X_k=\mathrm{Dec}_k\bigl(B_k^{(\mathrm{samp})},\,S_k^{(\mathrm{samp})}\bigr)
\in\mathbb{R}^{C_k\times H_k\times W_k}.
$$


Same frozen weights as encode — full Enc+Dec teacher, not a student skip-synth
from $z$.

#### 6. Denormalize → resample → evolve

Invert dens/$\sigma$ / $\langle v\rangle$ normalizations →
`resample_particles_from_multiscale` (count mix disk∶halo∶bulge $\approx 4:2:1$) →
$N$-body evolve.

**Working path reminder.** Continuous $z$ is only a **retrieval / jitter index**
over stored $(B,S)$. Do **not** treat $z\to\mathrm{Dec}$ or end-to-end
skip-synth students as the production IC path — those wash bars
($\mathrm{A}_2\lesssim 0.08$).

### 5.1 Library construction

1. Rank corpus dumps by **particle** $\mathrm{A}_2$ (`corpus_particle_a2_rank.json`).
2. Stratified pick: strong bars ($\mathrm{A}_2\ge$ `bar_floor`) + quiet
   ($\le$ `quiet_ceil`) + mid; rotate barred members by $\phi_k=2\pi k/n_{\mathrm{rot}}$.
3. Encode each with frozen **teacher** (crisp AE) → store bottleneck + skips per tower.
4. Pool bottlenecks (adaptive avg pool $g\times g$, default $g=4$), concatenate
   towers → raw vector $u\in\mathbb{R}^{d_u}$.
5. PCA: $z=(u-\bar u)\,W$, $W$ top $n_{\mathrm{pc}}$ right singular vectors.

Quiet-mean features:


$$
F_{\mathrm{quiet}}=\frac1{\lvert\mathcal{Q}\rvert}\sum_{i\in\mathcal{Q}} F_i
\quad\text{(blend of bottleneck+skips)}.
$$


### 5.2 Blend and residual amplify

**Blend** weights $w$ with $\sum w=1$:


$$
F=\sum_i w_i F_i
\quad\text{(linear on bottleneck and each skip tensor)}.
$$


**Residual amplify** ($\alpha>1$ strengthens non-axisymmetric residual):


$$
F(\alpha)=F_{\mathrm{base}}+\alpha\,(F_{\mathrm{target}}-F_{\mathrm{base}}).
$$


Quiet samples **never** use $\alpha>1$; they take exact low-$\mathrm{A}_2$ members.

### 5.3 Sampling methods (math)

| Method | Formula / rule | Typical bar $\mathrm{A}_2$ |
|--------|----------------|------------------------------|
| `uniform_knn` | Seed $i_0$ in pool; nearest $i_1$; $F=(1-\alpha)F_{i_0}+\alpha F_{i_1}$, $\alpha\sim U[0,\alpha_{\max}]$ | ~0.29–0.32 |
| `strong_knn` | Same but only among $\mathrm{A}_2\ge$ strong_floor; A₂-biased seed | ~0.32–0.33 |
| `a2_weighted_knn` | Softmax on $-\mathrm{dist}/T$ × $\mathrm{A}_2^{p}$ | mid |
| `exact` / `topk_exact` | Exact library member (ceiling / retrieval check) | near data |
| `amplify_residual` | $F(\alpha)=F_{\mathrm{quiet}}+\alpha(F_i-F_{\mathrm{quiet}})$, $\alpha\sim U[\alpha_{\mathrm{lo}},\alpha_{\mathrm{hi}}]$, $i\sim\mathrm{A}_2^2$ | ~0.39–0.41 |
| **`amplify_knn_hybrid`** | strong_knn blend → amplify vs quiet mean | **0.418±0.011** |
| `disk_only_amplify` | Amplify disk tower only; mild BH blend | ~0.39–0.43 |
| `local_pca` / `kde_retrieve` | Local PCA or diagonal KDE in $z$, then retrieve | weaker unless amplified |
| **`z_amplify`** | Local-jitter $z$ → strong retrieve → amplify | **0.410±0.026** |

**Continuous $z$ draw** (`sample_z_kde`): pick A₂-biased library code $i_0$,
jitter in local neighbor std with small global mix:


$$
z=(1-\gamma)\,(z_{i_0}+\varepsilon\odot\sigma_{\mathrm{local}})
+\gamma\,(\mu_{\mathrm{pool}}+\eta\odot\sigma_{\mathrm{pool}}).
$$


Global diagonal KDE alone washes bars under retrieve ($\sim 0.17$–$0.23$).

**Morph knob (α sweep):** $\alpha\approx 1.1\to\mathrm{A}_2\sim 0.35$;
$\alpha\approx 1.2\to\sim 0.42$; $\alpha\approx 1.3\to\sim 0.46$.

### 5.4 Sampling API

```python
from galacticsics.ml.fields.feature_library import TeacherFeatureLibrary
fields, meta = lib.sample_fields(kind="barred", method="amplify_knn_hybrid")
feat, meta = lib.sample_features_z_amplify(kind="barred")  # continuous z
fields = lib.decode_features(feat)
# denormalize → resample_particles_from_multiscale (count 4:2:1) → evolve
```

```bash
OMP_NUM_THREADS=6 python scripts/sample_latent_ic.py \
  --method amplify_knn_hybrid --n-samples 10 \
  --n-bar 36 --n-quiet 20 --n-mid 12 --n-rot-bar 6 \
  --bar-floor 0.25 --quiet-ceil 0.05 \
  --also-z-amplify --z-alpha-lo 1.20 --z-alpha-hi 1.50
```

### 5.5 Chronology (what closed the gap)

1. Overnight field VAE → bar **0.06**
2. Feature library v1 (stride sampling) → **0.15**
3. v2 stratified + `uniform_knn` → **0.29–0.32**
4. `amplify_residual` / hybrids → **0.36–0.42**
5. Marathon multiseed / final robust → **`amplify_knn_hybrid` 0.418±0.011**

Dead ends in the same period: skip-synth code VAE, flow-on-PCA codes alone,
α-from-$z$ ridge (overfits maps), global KDE without amplify.

---

## 6. What worked vs failed (summary table)

| Approach | Bar $\mathrm{A}_2$ | Why |
|----------|---------------------|-----|
| Morton index MSE / global mix CE | collapse / uniform | wrong inductive bias |
| Morton Chamfer+profiles+CE | morphology starts | still weak face-on bars |
| Soft virial→1 (particle) | better evolve; bars mixed | fights non-axisym residual |
| Field AE 32² dens-heavy | 0.12 | under-resolved |
| Field AE + Fourier force | quiet invents structure | $w_F$ too strong |
| Field AE dens-grad | bars hurt (0.25) | smooths high-freq residual |
| **Field AE crisp dens+moment** | **0.33** recon | good teacher |
| Field VAE `decode(z)` | ≲0.06 | skips discarded |
| Skip-synth from $z$ | ≲0.08 | mean-collapse |
| Library stride / weak knn | ~0.15–0.32 | under-samples peak bars |
| **Library amplify / hybrid** | **~0.41** | restores skip residual |

---

## 7. Post-resample evolution consistency

### 7.1 Existing smoke evidence (thin / bad dt)

Crisp recon evolve (`crisp_2026-07-24/verdict.json`):

| Quantity | Value |
|----------|-------|
| Settings | `end_gyr=0.02`, **`dt=0.05` → $n_{\mathrm{steps}}=1$** leapfrog |
| $N$ | 40 000, OpenMP `bh_c` |
| Bar $\mathrm{A}_2$ IC → final | **0.326 → 0.307** |
| COM drift | **0.0087 kpc** |
| $\Delta E/E$ | NaN (pairwise energy skipped) |

The smoke script itself notes this is not a meaningful short evolve. Quiet IC was
not evolved in that panel. Marathon “evolve panel” only recorded **IC** particle
$\mathrm{A}_2$ (bar μ≈0.41), not post-evolve metrics.

### 7.2 Proper short compare (this writeup)

Script: `scripts/evolve_resample_compare.py` →
`runs/ml/field_maps/evolve_compare_proper_2026-07-25/verdict.json`.

Settings: `dt=0.01`, `end_gyr=0.20` (**20** leapfrog steps), $N=3\times 10^4$,
OpenMP `bh_c`, shared COM, count mix 4:2:1. Generated:
`amplify_knn_hybrid` from a stratified crisp-AE library. Data: top-ranked
`54a8` dump (`step_001800.npz`), same $N$ stratified subsample.

| System | $\mathrm{A}_2$ pre → post | $\Delta\mathrm{A}_2$ | COM drift | $Q_{\mathrm{sub}}$ (soft) |
|--------|----------------------------|------------------------|-----------|------------------------------|
| **generated** (`amplify_knn_hybrid`) | **0.439 → 0.412** | **−0.027** | **0.0086 kpc** | 19.13 → 19.12 |
| **data dump** (barred peak) | **0.493 → 0.469** | **−0.024** | **0.0010 kpc** | pathological\* |

\*Soft pairwise $Q$ on the data subsample is not comparable (mass/softening
scale vs dens-integrated resample masses); treat COM + $\mathrm{A}_2$ as the
meaningful checks.

**Verdict:** over a proper short self-gravity evolve, the generative bar does
**not** wash out or explode. $\mathrm{A}_2$ drifts by roughly the same absolute
amount as the data subsample ($\sim -0.025$), and COM drift stays $\lesssim 0.01\,\mathrm{kpc}$
(comparable to the old 1-step crisp smoke). This is evidence of **consistent
short-term evolution**, not of long-term pattern-speed fidelity. Longer
($\gtrsim 0.5\,\mathrm{Gyr}$) suite still needed for production claims.

Contrast with thin smoke: crisp recon 1-step panel had $\mathrm{A}_2$ 0.326→0.307
and COM 0.0087 kpc — directionally similar $\Delta\mathrm{A}_2$ but **not** a
valid integrator test.

### 7.3 Evolve-gated multi-recipe suite (50 steps)

Script: `scripts/evolve_gate_recipes.py` →
[`evolve_gate_2026-07-25/`](../runs/ml/field_maps/evolve_gate_2026-07-25/).

Settings: `dt=0.01`, `end_gyr=0.50` (**50** leapfrog steps), $N=3\times 10^4$,
OpenMP=6, **soft global COM/VCOM recenter** before evolve, stratified teacher
library (36/20/12 + 6 bar rotations). Amplify recipes have high IC variance —
gate samples with IC $\mathrm{A}_2\ge 0.30$ (retry up to 8 draws; keep best).

| Recipe | $\mathrm{A}_2$ pre→post | $\Delta\mathrm{A}_2$ | COM drift | Gate |
|--------|-------------------------|----------------------|-----------|------|
| data dump (ref) | 0.546→0.543 | −0.003 | ~0 | ref |
| **`amplify_knn_hybrid`** | **0.714→0.700** | **−0.014** | ~0 | **PASS** |
| `z_amplify` | 0.485→0.444 | −0.041 | ~0 | **PASS** |
| `disk_only_amplify` | 0.498→0.488 | −0.010 | ~0 | **PASS** |
| `skip_residual_pca` (+ mild amplify) | 0.356→0.304 | −0.052 | ~0 | **PASS** |

**Measurable wins vs §7.2:** soft COM drops COM drift from $\sim 0.01\,\mathrm{kpc}$
to $\sim 0$; hybrid bars that clear the IC floor **survive 50 steps** with
$\lvert\Delta\mathrm{A}_2\rvert$ comparable to or smaller than data. Continuous
`skip_residual_pca` (disk residual PCA morph + quiet-mean amplify) clears the
evolve gate at $\mathrm{A}_2\sim 0.30$ post — still below hybrid, but far above
old skip-synth students ($\lesssim 0.08$).

Pilot without IC floor (same script, smaller library): many amplify draws land
weak ($\mathrm{A}_2\sim 0.1$–$0.2$); always report IC-gated or multi-seed means.

### 7.4 Qualitative expectation

Resample from dens+$\langle v\rangle$ (optional $\sigma$) **does not** recover
full DF equilibrium. Expect:

- Mild $\mathrm{A}_2$ drift over tens of steps if the bar is real structure
  (not Poisson noise).
- COM drift small if centering was shared, soft COM is applied, and forces are
  consistent.
- Soft pairwise $Q$ often remains **hot** (same caveat as GalactICS dumps under
  self-gravity only) — not a hard fail by itself.
- Longer evolves ($\gtrsim 0.5\,\mathrm{Gyr}$) needed for spiral winding / bar
  pattern-speed claims; the 50-step panel tests “does it explode / wash
  instantly?” under a proper integrator.

---

## 8. Forward pathways (from $\mathrm{A}_2\sim 0.41$ generative)

Ordered by expected ROI given current bottlenecks (soft dens maps, library
retrieval, evolve variance):

1. **Proper evolve suite on generative ICs** — **done (short gate)**  
   `evolve_gate_recipes.py` with `dt=0.01`, 50 steps, soft COM, IC floor.
   Next: $\gtrsim 0.5$–$1\,\mathrm{Gyr}$ pattern-speed / spiral suite; tree Φ
   energy; more seeds without IC cherry-picking for yield stats.

2. **Sharper teacher dens maps without quiet invention**  
   Quiet-gated Fourier on *linearized* dens + log-space axisym dens residual
   (`am_match_ft_*`): particle $A_m(R)$ MSE improved (bar m1 $0.013\to0.007$,
   m2 $0.007\to0.0065$) and quiet stayed quiet, but dens-map $A_2$ median is
   still $\sim0.12$ vs data $\sim0.37$ — peak *heights* under-predict. **Spatial
   FFT morphology** on residual dens (`spatial_fft_morphology_loss`,
   `--fft-weight`, `fft_morph_ft_*`): short 12-ep run was under-trained; converged
   continue (`fft_morph_ft_converged_*`, +80 ep) still **NO-GO** vs `am_match_ft`
   on bar $A_m(R)$ / map $A_2$ (dens MSE improves) — keep `fft_weight=0`. Next:
   barred-heavy train mix; residual / high-pass dens head; keep dens-grad MSE
   off. Light Fourier **only** after moments converge and only on disk.

3. **True continuous latent without skip wash** — **partial**  
   Disk skip-residual PCA + morph + mild amplify passes evolve gate (~0.30 post)
   but trails hybrid (~0.70 gated). Next: morph-calibrated $\beta\to\mathrm{A}_2$
   map; hierarchical $z_{\mathrm{coarse}}+z_{\mathrm{bar}}$; avoid full skip-L2
   students.

4. **α / morph control as a calibrated knob**  
   Map $\alpha$ (or morph scalar) → particle $\mathrm{A}_2$ with hold-out
   seeds; expose as conditioning rather than ad-hoc ranges. Couple with IC
   floor for production sampling yield.

5. **Disk-only amplify + BH consistency**  
   Evolve-gate PASS; couple with soft COM (done) and optional soft virial on
   resampled particles before evolve (particle-path lessons).

6. **Scale resample $N$** toward corpus counts for production evolves; keep
   count mix 4:2:1; **stratified** downsample when $N_{\mathrm{evolve}}<N_{\mathrm{resample}}$.

7. **Do not prioritize** end-to-end field VAE prior-path or global KDE-$z$
   alone until skip residual is modeled.

---

## 9. Code / artifact map

| Path | Role |
|------|------|
| `ml/models/sequence_vae.py` | Particle set VAE + Chamfer/CE/KL/virial |
| `ml/profiles.py` | Soft $\Sigma,\rho,v_\phi,A_m$, virial |
| `ml/fields/binning.py` | Multi-scale dens+moments deposit |
| `ml/fields/autoencoder.py` | U-Net AE + soft $A_m(R)$ losses |
| `ml/fields/vae.py` | End-to-end field VAE |
| `ml/fields/latent_code.py` | Skip-synth research prior |
| `ml/fields/feature_library.py` | Teacher library + amplify / $z$ APIs |
| `ml/fields/resample.py` | Dens → particles (count mix) |
| `scripts/sample_latent_ic.py` | Recommended IC sampling CLI |
| `scripts/evolve_resample_compare.py` | Proper short gen-vs-data evolve |
| `scripts/evolve_gate_recipes.py` | Evolve-gate multi-recipe suite (`dt≈0.01`, soft COM) |
| `scripts/sample_skip_residual_pca.py` | Continuous morph via disk skip-residual PCA |
| `runs/ml/field_maps/crisp_2026-07-24/` | Best recon **teacher** AE |
| `runs/ml/field_maps/marathon_final_robust_2026-07-25/` | Best generative robust numbers |
| `runs/ml/field_maps/evolve_gate_2026-07-25/` | Evolve-gated recipe comparison |
