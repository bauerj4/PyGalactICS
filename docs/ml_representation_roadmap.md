# ML / transformer representations for particle data

Learned models complement parametric, harmonic, and tabulated galaxy
representations: instead of fitting a potential basis or exporting Gadget
blocks, a neural encoder maps an N-body snapshot to a fixed-size latent
vector (and optionally per-particle embeddings).

**For the full phased strategy, architecture comparison, and training plan,
see [`ml_encoder_strategy.md`](ml_encoder_strategy.md).**

## Representation taxonomy

| Kind | Example | Use case |
|------|---------|----------|
| `parametric` | `GalaxyModel` | Fitting, DBH solve |
| `harmonic` | `HarmonicPotential` | Fast evaluation, galpy/agama/gala export |
| `particle` | `ParticleState` | Self-gravitating evolution in ntropy |
| `tabulated` | spline grids | Interpolation, I/O |
| **`learned`** | field / graph / transformer latent | Surrogate potentials, similarity search, hybrid IC generation |

## Code entry points

| Module | Purpose |
|--------|---------|
| `galacticsics.representations.particle_features` | Tokenize snapshots (9-column export) |
| `galacticsics.representations.preprocess` | COM / PCA; 8-column encoder features |
| `galacticsics.representations.learned` | `EncoderBackend`, `build_encoder()`, protocol |
| `galacticsics.integrations.field_encoder` | **Recommended** scalable encoder stub |
| `galacticsics.integrations.graph_encoder` | kNN graph encoder stub |
| `galacticsics.integrations.torch_encoder` | Transformer stub (not for production N) |
| `galacticsics.ml` | Training export, multi-task heads, IC decoder stubs |
| `galacticsics.campaign.ml_hooks` | Per-run / per-checkpoint encoding |

Install ML stack: `pip install galacticsics[ml]`

## Quick start

```python
from galacticsics.representations.learned import EncoderBackend, LearnedEncoderConfig, build_encoder
from galacticsics.representations.particle_features import particle_state_to_batch

batch = particle_state_to_batch(state)
enc = build_encoder(LearnedEncoderConfig(backend=EncoderBackend.FIELD))
rep = enc.encode(batch)
print(rep.latent.shape, rep.metadata.trained)  # (128,) False
```

Export single snapshot:

```bash
python examples/export_particle_features.py --output features.npz
```

Export campaign training bundle:

```bash
python examples/export_campaign_training_data.py path/to/campaign --backend field
```

## Particle tokens

Each particle exports as a 9-column row:

`(x, y, z, vx, vy, vz, log10(mass), log10(eps), type_id)`

Neural encoders use **8 columns** for linear layers; `type_id` feeds
`nn.Embedding` only.  See `ENCODER_FEATURE_NAMES` in `preprocess.py`.

## Encoder backends

| Backend | Scales to 200k? | Status |
|---------|-----------------|--------|
| `mean_pool` | Yes | Baseline |
| `field` | Yes | **Recommended stub** |
| `graph` | With subsample | Stub |
| `transformer` | No (O(N²)) | Untrained scaffold |
| `perceiver` | Yes (planned) | Not implemented |

## Campaign integration

| Stage | ML hook | Status |
|-------|---------|--------|
| `sample` | `features.npz` via export script | Available |
| `evolve` | `encode_evolution_checkpoints()` | Stub hook |
| post-run | `export_campaign_training_bundle()` | Available |
| `compare` | Latent distance vs harmonic RMS | Planned |

## Training objectives (stubs in `galacticsics.ml.heads`)

1. **Contrastive** — same run across timesteps / augmentations
2. **Quality regression** — predict ΔE/E₀, bin stats from IC latent
3. **Param inverse** — recover halo.v0, disk.mass
4. **Field reconstruction** — decode latent → binned ρ, |v|
5. **Potential correction** — latent + r → ΔΦ (Phase 3)

## IC generation (Phase 4 stub)

`ConditionalICDecoder` predicts harmonic coefficients + low-dimensional
non-axisymmetric perturbation—not raw particle coordinates.  See
`ml_encoder_strategy.md` Phase 4.

## Dependencies

| Extra | Packages | Required for |
|-------|----------|--------------|
| `ml` | `torch` | Transformer, field MLP, training |
| (future) `ml-geo` | `e3nn`, `torch-geometric` | Equivariant graph encoder |

Core `galacticsics` stays numpy-only; field binning works without PyTorch.
