# Morton generative galaxies

Learned generative models of **Milky Way–like disk–bulge–halo** systems as
**Morton-ordered particle sequences**, trained on GalactICS ICs and their
evolved snapshots.

**Morphology recommendation:** for bars / spirals / non-equilibrium IC sampling,
prefer the **conditional field VAE**
([`field_maps.md`](field_maps.md), `galacticsics.ml.fields.vae`,
`scripts/smoke_field_vae.py`).  This particle-set track remains a useful
**phase-space baseline** (mix fractions, virial, token-level ablations) but is
weaker on face-on morphology than multi-scale field maps.

Also related: archived field-encoder notes in
[`archive/ml_encoder_strategy.md`](archive/ml_encoder_strategy.md); findings in
[`ml_findings.md`](ml_findings.md); full latent-IC math chronology in
[`ml_latent_ic_methods.md`](ml_latent_ic_methods.md).  Shared structural ``θ`` keys:
`galacticsics.ml.conditioning`.

## Corpus

Config: [`campaigns/mw_morton_corpus.json`](../campaigns/mw_morton_corpus.json)
(grid: [`campaigns/mw_morton_grid.json`](../campaigns/mw_morton_grid.json)).

| Knob | Value |
|------|--------|
| Particles | disk **1e6**, halo **5e5**, bulge **2.5e5** |
| Duration | **2 Gyr** (override with `--end-time-gyr`) |
| Force | `gpu_bh` |
| Concurrent evolves | `run.gpu_batch_size` (default 4) |
| Dumps | float32 `step_*.npz`, ~every 100 Myr |
| Models | up to **48** MW-like axis combinations |

### Storage estimate (float32 compressed)

| Scope | Approx size |
|-------|-------------|
| One snapshot (1.75M particles) | ~55–70 MB |
| One run (~22 snaps) | ~1.3–1.6 GB |
| Full 48-run corpus | **~60–80 GB** (plan ~100 GB free) |

### Launch

```bash
galacticsics-campaign run campaigns/mw_morton_corpus.json \
  --work-root runs/mw_morton_corpus \
  --end-time-gyr 2.0 \
  --gpu-batch-size 4
```

Solve/sample run serially; evolve is dispatched in GPU process-pool batches.

Grid subsampling is **deterministic** (sha256 of the campaign name). With
``skip_done: true``, a relaunch **resumes existing sampled model dirs** under the
work root so a prior run is not discarded.

## Snapshot index (no sequence cache)

```bash
galacticsics-morton-index runs/mw_morton_corpus -o runs/mw_morton_corpus/snapshot_manifest.json
```

Training loads a full snapshot each step, draws a random stratified subset
(`N` default 4096), Morton-sorts, and builds tokens `(c, Δm, x, v)` where `x`
is absolute position in Morton order and `Δm` is the Morton-key increment
(ordering feature for the AR model).

## Models

| Model | Decode | Loss |
|-------|--------|------|
| **Set VAE** (`SequenceVAE`) | Parallel noise queries + `(z, θ)` → `N` tokens in one shot (chunked for larger `N`) | β-VAE + soft `Σ(R)`, `ρ(r)`, `⟨v_φ⟩(R)` + Fourier `A_m/A_0` (m=1,2) + soft maps + stratified Plummer virial (`2K/|W|` → 1) |
| **AR transformer** (`MortonTransformer`) | Left-to-right next-token | CE(`c`) + MSE(`Δm`, `x`, `v`) |

**Findings (short).** Index-wise MSE / Morton-ordered decode collapsed samples;
Chamfer + geometric bases helped morphology. Global `mix_head` soft CE stuck
~uniform (~3.33); per-particle CE + stratified `c → x,v|c` reached ~0.23–0.32 CE
(~91–95% acc). Soft virial→1 improves evolve stability but can under-produce
strong bars — keep Fourier/maps strong. Target `N` 1e5–1e6 with DeepSets +
subsampled Chamfer. Details: [`ml_findings.md`](ml_findings.md),
`runs/ml/vae_virial_ablation/NOTE.md`.

While the corpus evolve is using the GPU, keep VAE training on **CPU** with small
`N` / `d_model` / batch (script defaults: `N=128`, `d_model=64`, `device=cpu`).
Interactive walkthrough: [`notebooks/morton_vae_small.ipynb`](../notebooks/morton_vae_small.ipynb).
Loss ablation (sample $10^5$, short evolve): [`notebooks/vae_loss_ablation.ipynb`](../notebooks/vae_loss_ablation.ipynb).

## Training / generation

```bash
pip install -e '.[ml]'

# Prefer CPU + small N while gpu_bh corpus runs
python scripts/train_morton_vae.py runs/mw_morton_corpus_v2/snapshot_manifest.json \
  --n-particles 128 --epochs 5 --device cpu --out runs/ml/morton_vae

python scripts/train_morton_transformer.py runs/mw_morton_corpus_v2/snapshot_manifest.json \
  --n-particles 256 --epochs 5 --order morton --device cpu --out runs/ml/morton_transformer

python scripts/generate_morton_galaxy.py runs/ml/morton_vae/sequence_vae.pt \
  --model vae --out generated_ic.npz --evolve-gyr 0.05
```

Ablation: train the transformer with `--order random` vs `--order morton` on the same manifest.

## Code map

| Path | Role |
|------|------|
| `galacticsics.ml.morton` | tokenize, index, on-the-fly particle `Dataset` |
| `galacticsics.ml.models` | particle baseline: `SequenceVAE` (set), `MortonTransformer` (AR) |
| `galacticsics.ml.fields` | **morphology path**: maps + recon AE + field VAE |
| `galacticsics.ml.conditioning` | shared ``θ`` keys (field + particle) |
| `galacticsics.ml.profiles` | Soft `Σ` / `ρ` / `v_φ`, Fourier `A_m`, maps, Plummer virial |
| `campaigns/mw_morton_*.json` | corpus configs |
