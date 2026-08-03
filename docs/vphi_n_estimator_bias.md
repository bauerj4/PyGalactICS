# ⟨v_φ⟩(R) drop-off at low N — estimator bias (not dynamics)

**Date:** 2026-07-27  
**System:** `GalaxyModel.milky_way_disk_halo()` (bulge=`None`); campaign analog
[`campaigns/bauer_morphology.json`](../campaigns/bauer_morphology.json)
(`particles.bulge: 0`).

## Symptom

Disk ⟨v_φ⟩(R) (and sometimes σ) appeared to **fall in the outer disk** when
particle count dropped (e.g. \(N\sim10^4\) vs \(10^6\)), even though Σ(R) agreed.
That is not a physical N-body effect on a fixed DF sample.

## Root cause

Three estimator traps, all of the form “empty → 0”:

1. **Particle radial means** — `numpy_mass_weighted_profile` filled empty
   annuli with `0` instead of `NaN`. Sparse outer bins at low \(N\) then look
   like a falling rotation curve. (`particle_rotation_curve` already used NaN
   + `min_count`; OOD `_disk_kinematics` did not gate on `min_count`.)
2. **Deposit map reduction** — `_deposit_moments_2d` correctly stores `⟨v⟩=0`
   in empty pixels (AE needs finite tensors), but **equal-pixel** annular means
   of those maps average empty cells as \(v=0\). At \(N=10^4\), ~85% of pixels
   in \(R\in[8,12]\) are empty on the smoke disk grid → ⟨v_φ⟩→0. Fix: dens-weight
   occupied pixels only (`radial_vphi_from_deposit_slab`).
3. **Map v_φ orientation (DF loss)** — `dens_weighted_vphi_field_loss` used
   `yy, xx = meshgrid(..., indexing="ij")` which **swaps** axes relative to
   `histogram2d` layout (`arr[i,j]↔(x_i,y_j)`). For a rotating disk the wrong
   combination averages ~0, so the “⟨v_φ⟩ phys” term barely supervised rotation.
   Fixed to `xx, yy = meshgrid(..., ij)` with \(v_\phi=(-y\,v_x+x\,v_y)/R\).

Soft Morton `soft_mean_vphi` had the same empty→0 pattern via
`den.clamp_min(1e-30)`; empty soft bins are now NaN and masked in the profile loss.

## What is *not* broken

- Midplane Σ(R) converges across \(N\in\{10^4,10^5,10^6\}\).
- Mass-weighted particle ⟨v_φ⟩ with `min_count≥20` agrees where bins are
  populated (outer bins simply become NaN / omitted at low \(N\)).

## Fixes shipped

| Location | Change |
|----------|--------|
| `ml/profiles.py` | empty means → NaN; `min_count`; soft vφ mass floor + masked MSE |
| `diagnostics/rotation_curve.py` | mass-weighted means (still NaN + `min_count`) |
| `ml/fields/binning.py` | `radial_vphi_from_deposit_slab`; deposit docstring |
| `ml/fields/autoencoder.py` | correct vφ map axes for dens-weighted loss |
| `scripts/ood_theta_df_compare.py` | `min_count=20` on disk kinematics |
| `scripts/diagnose_vphi_n_convergence.py` | N-convergence diagnostic |

## Figures / artefacts

Run:

```bash
python scripts/diagnose_vphi_n_convergence.py \
  --out runs/ml/field_maps/vphi_n_converg_2026-07-27
```

- Panels: `runs/ml/field_maps/vphi_n_converg_2026-07-27/fig_vphi_N_before_after.png`
- Paper pair: `.../fig_vphi_N_paper.png`
- Metrics: `.../metrics.json`
- Paper copy: `papers/mnras_noneq_ics/results/vphi_N_estimator_SUMMARY.md`
- Tests: `tests/test_vphi_n_estimators.py`

## Retrain after axis fix (2026-07-27)

Prior joint dens+⟨v_φ⟩ (**A**) FT in `df_match_BA_2026-07-27/joint_df_fftlong_ft`
used the **swapped-axis** `dens_weighted_vphi_field_loss`. Retrained with the
corrected loss under `runs/ml/field_maps/vphi_fixed_A_ft_2026-07-27/`:

| Warm start | ckpt |
|------------|------|
| preferred (buggy-A teacher) | `joint_a_ft/multitower_slice_ae.pt` |
| cleaner (`fft_morph_ft_long`) | `joint_a_from_fftlong/multitower_slice_ae.pt` |

**OOD ⟨v_φ⟩ MSE** (fft_recon, particle profiles with `min_count=20`; B on):

| Case | buggy A+B | fixed ← buggy | fixed ← fftlong |
|------|-----------|---------------|-----------------|
| heavy_ext_quiet | 0.180 | 0.180 | 0.179 |
| heavy_bar_forming | 0.231 | 0.230 | 0.240 |
| thick_stable | 0.474 | 0.475 | 0.477 |

**Verdict:** axis fix was required for correctness, but **correct A does not
improve OOD ⟨v_φ⟩** beyond the buggy-A checkpoint. Train `vphi_phys` stays
~10⁻³ on normalized stacks (≪ other loss terms). Prior ~6–11% gains vs
fft_long are attributable to **moment_phys + dens FT**, not the vφ map term.
Do **not** claim equal-pixel zero-filled ⟨v_φ⟩. Summary:
[`papers/mnras_noneq_ics/results/vphi_fixed_A_SUMMARY.md`](../papers/mnras_noneq_ics/results/vphi_fixed_A_SUMMARY.md).
