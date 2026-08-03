# Closest GalactICS \(f_0\) + full dynamical residual

**Goal (user clarification, 2026-08-02):** find the GalactICS equilibrium DF
\(f_0(\theta)\) that most closely matches barred *data* as a starting point
(disk ± bulge ± halo), then reconstruct **all** dynamics of the dump — not
m2 paint on a locked quiet \(\Sigma(R)\).

This note demotes the paint-on-bar recipe as the primary path and documents
the replacement workflow.

> **2026-08 Part~1 update:** primary product is **θ-conditioned latent
> retrieve+decode** (path-LOO particle remass to \(f_0(\theta)\); disk-COM
> \(A_2\)). Scoreboard: 906c4 MATCH, no-bulge FADE MATCH, 54a8 demoted.
> Cosmo variants = outlook. `full_dyn_replace` (data-disk graft) and
> `residual_f0` / paint-on-bar remain **demoted** — not generative headlines.
> Closest-\(f_0\) search is still useful to pick the quiet remass base.
> See [`ml_latent_ic_methods.md`](ml_latent_ic_methods.md),
> [`ml_findings.md`](ml_findings.md),
> [`papers/mnras_noneq_ics/`](../papers/mnras_noneq_ics/).

## Why paint-on-bar was the wrong primary

Phase A `residual_f0_m2` does:

\[
\Sigma'(R,\phi) = \mathrm{axisym}(\Sigma_{f_0})\times\bigl(1+\alpha\,m_2[\Sigma_{\mathrm{morph}}]\bigr),
\]

then optionally kNN-transplants morph velocities, while **radial mass is locked
to quiet \(f_0\)** (`preserve_axisym` + CDF lock to \(f_0\)).

Failures (54a8 fade; OOD radial mismatch) are not fixed by α dials alone when
the target dump’s axisym dens+kin already differ from the locked quiet
profile. The residual must carry **non-axisym dens and the full velocity DF**,
not a bar painted on the wrong radial scaffold.

Honest addendum from the corpus search (below): for 906c4 / 54a8 the
**same-campaign** quiet IC *is* the closest discrete \(\theta\) in
`mw_morton_corpus_v2`. So the bug was not “wrong campaign hash” — it was
**locking the residual to quiet axisym** while the dump has bar-driven
\(\Sigma(R)\) and \(\langle v_\varphi\rangle\) redistribution. Closest-\(f_0\)
still matters for true OOD dumps / unknown \(\theta\).

## A. Closest \(f_0(\theta)\) search

Script: [`scripts/closest_galactics_f0.py`](../scripts/closest_galactics_f0.py)

Scores every corpus `ic_state.npz` (+ optional extras) against a barred dump
on **axisymmetric** observables only:

| Term | Metric |
|------|--------|
| Disk dens | med\(\lvert\log_{10}\Sigma_{\mathrm{disk}}\rvert\) |
| Bulge dens | med\(\lvert\log_{10}\rho_{\mathrm{bulge}}\rvert\) |
| Halo dens | med\(\lvert\log_{10}\rho_{\mathrm{halo}}\rvert\) (small weight) |
| Kinematics | MSE of \(\langle v_\varphi\rangle,\sigma_R,\sigma_\varphi,\sigma_z\) |
| Masses | relative disk/bulge/total mismatch |

```bash
.venv/bin/python scripts/closest_galactics_f0.py \
  --dump runs/mw_morton_corpus_v2/906c4af73543/evolution/particles/step_003200.npz \
  --same-campaign 906c4af73543 \
  --out runs/ml/field_maps/closest_f0_full_dyn_2026-08-02/search_906c4 \
  --label 906c4
```

**2026-08-02 corpus result**

| Dump | Best hash | dens med\(\lvert\log\rvert\) disk | kin MSE mean | vs same-campaign |
|------|-----------|-----------------------------------|--------------|------------------|
| 906c4 `step_003200` | **`906c4af73543`** (rank 1/48) | 0.117 | 0.086 | same is best |
| 54a8 `step_002200` | **`54a8faf836a0`** (rank 1/48) | 0.152 | 0.097 | same is best; `d9cd` nearly tied |

Best θ (906c4): \(M_d=6\), \(R_d=2\), \(z_d=0.25\), \(Q=1.7\), \(v_{0,h}=3.2\), \(a_h=25\), \(v_{0,b}=1.5\), \(a_b=0.4\).

Best θ (54a8): \(M_d=6\), \(R_d=2\), \(z_d=0.2\), \(Q=1.7\), \(v_{0,h}=4.0\), \(a_h=25\), \(v_{0,b}=2.0\), \(a_b=0.4\).

Remaining quiet→barred mismatch at best \(f_0\) is still \(\mathcal{O}(0.1)\) in
med\(\lvert\log\Sigma\rvert\) and \(\langle v_\varphi\rangle\) MSE \(\sim0.25\) —
that gap is what the **full-dyn residual** must close (not another corpus θ).

## B. Full dynamical residual (Phase 1)

`--recipe` on [`scripts/residual_galactics_ic.py`](../scripts/residual_galactics_ic.py):

| Recipe | Behaviour |
|--------|-----------|
| `paint` (default) | Legacy m2 on \(f_0\) axisym + vel dial |
| **`full_dyn_replace`** | Disk phase-space from **data dump**; halo/bulge from \(f_0(\theta^*)\) |
| `full_dyn_ot` | OT-lite radial+\(\langle v_\varphi\rangle\) transport \(f_0\to\)data; **keeps quiet azimuth** (A₂≈0) — axisym helper only |

Primary Phase-1 choice: **`full_dyn_replace`**.

```bash
.venv/bin/python scripts/residual_galactics_ic.py \
  --recipe full_dyn_replace \
  --ic-path runs/mw_morton_corpus_v2/906c4af73543/ic_state.npz \
  --morph-path runs/mw_morton_corpus_v2/906c4af73543/evolution/particles/step_003200.npz \
  --out runs/ml/field_maps/closest_f0_full_dyn_2026-08-02/evolve_2gyr_906c4_full_dyn_replace \
  --n-disk 1000000 --evolve-gyr 2.0 --force gpu_bh --a2-r-eval 2.0 \
  --data-path runs/mw_morton_corpus_v2/906c4af73543/evolution/particles/step_003200.npz \
  --data-arm-label "data dump" \
  --paper-figures papers/mnras_noneq_ics/figures \
  --paper-prefix fig_closest_f0_906c4_full_dyn_replace
```

Gate vs **data dump** (not quiet control). Primary metric: \(A_2(R_d)\).

### t=0 dens/kin vs data (2026-08-02)

| System | Method | \(A_2(R_d)\) | kinetic mean MSE |
|--------|--------|-------------|------------------|
| 906c4 | full_dyn_replace | **0.493** (data≈0.499) | **0.0013** |
| 906c4 | paint + morph-vel | 0.450 | 0.0158 |
| 906c4 | full_dyn_ot | 0.002 | 0.022 |
| 906c4 | quiet \(f_0\) alone | ~0 | dens med\(\lvert\log\rvert\) 0.12; kin MSE 0.086 |
| 54a8 | full_dyn_replace | **0.517** (data≈0.523) | **0.0044** |
| 54a8 | paint + morph-vel | 0.492 | 0.0183 |
| 54a8 | quiet \(f_0\) alone | ~0 | dens 0.15; kin MSE 0.097 |

Replace ≈ **12×** better kinetic match than paint+morph-vel at t=0 on 906c4;
A₂ matches data instead of under-shooting via matchRd paint.

## C. Relation to mid-flight paint improvements

`runs/ml/field_maps/residual_improvements_2026-08-02/` (vel dials, Phase B+morphvel,
OOD paint) remains useful ablation evidence. This goal **supersedes** further
paint-on-bar tuning as the main path: keep hybrid R≤2 / morph-vel as controls;
ship closest-\(f_0\) + `full_dyn_replace` as the intended IC construction.

## Remaining gaps (honest)

1. **`full_dyn_replace` is not a learned residual** — it is data-disk grafting
   onto \(f_0\) halo/bulge. Generative / OOD invention still needs a residual
   operator that *creates* bar dens+kin without the dump particles.
2. **OT-lite does not make bars** (azimuth of source preserved).
3. **Continuous θ re-solve** (GalactICS dbh+sample fit to evolved axisym) is
   not yet automated; corpus discrete search found same-campaign already best.
4. **2 Gyr \(A_2(R_d)\) gates** for full_dyn_replace: see journal under
   `runs/ml/field_maps/closest_f0_full_dyn_2026-08-02/` (queued behind GPU
   improvements suite).

## Artefacts

- Journal / scoreboard: `runs/ml/field_maps/closest_f0_full_dyn_2026-08-02/`
- Search: `search_{906c4,54a8}/`
- Paper figs (kin t=0): `papers/mnras_noneq_ics/figures/fig_closest_f0_*`