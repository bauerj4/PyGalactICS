#!/usr/bin/env bash
# Overnight creative latent sweep: frozen AE + skip distill hybrids.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

OUT_ROOT="runs/ml/field_maps"
LOG="${OUT_ROOT}/creative_overnight_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$OUT_ROOT"
exec > >(tee -a "$LOG") 2>&1
echo "=== creative latent overnight $(date -Is) OMP=${OMP_NUM_THREADS} ==="

run_one () {
  local name="$1"; shift
  local out="${OUT_ROOT}/${name}"
  echo
  echo "######## ${name} ########"
  python scripts/train_latent_code_prior.py --out "$out" "$@"
}

# 1) Base skip-distill (no morph, no flow) — longer train
run_one "creative_latent_base_$(date +%Y-%m-%d)" \
  --epochs 50 --max-snap 36 --latent-dim 128 --lr 2e-4 --min-lr 2e-5 \
  --barred-weight 3 --beta 5e-4 --skip-weight 1.5 \
  --n-resample 80000 --n-prior-samples 4 --n-interp 5 \
  --note "base skip-distill z128 barred-heavy"

# 2) Morphology-conditioned θ (A₂ summary) — controllable bar knob
run_one "creative_latent_morph_$(date +%Y-%m-%d)" \
  --epochs 50 --max-snap 36 --latent-dim 128 --lr 2e-4 --min-lr 2e-5 \
  --with-morph --barred-weight 3 --beta 5e-4 --skip-weight 1.5 \
  --n-resample 80000 --n-prior-samples 4 --n-interp 5 \
  --note "morph A2 conditioning + skip-distill"

# 3) Larger z + RealNVP prior on codes
run_one "creative_latent_morph_flow_$(date +%Y-%m-%d)" \
  --epochs 45 --max-snap 36 --latent-dim 192 --lr 1.5e-4 --min-lr 2e-5 \
  --with-morph --use-flow --flow-nll-weight 0.08 --barred-weight 3.5 \
  --beta 3e-4 --skip-weight 2.0 --synth-grid 10 \
  --n-resample 80000 --n-prior-samples 4 --n-interp 5 \
  --note "morph+flow z192 denser synth grid"

# 4) Deterministic codes + morph + flow (no KL fight — key fix)
run_one "creative_latent_det_morph_flow_$(date +%Y-%m-%d)" \
  --epochs 45 --max-snap 40 --latent-dim 128 --lr 2e-4 --min-lr 2e-5 \
  --deterministic --with-morph --use-flow --flow-nll-weight 0.1 \
  --barred-weight 3.5 --skip-weight 2.0 --synth-grid 10 --beta 0 \
  --n-resample 80000 --n-prior-samples 4 --n-interp 5 --patience 14 \
  --note "det codes + morph + RealNVP (no KL)"

# 5) Barred-heavier morph VAE (legacy hybrid)
run_one "creative_latent_ft_morph_$(date +%Y-%m-%d)" \
  --epochs 35 --max-snap 40 --latent-dim 128 --lr 8e-5 --min-lr 1.5e-5 \
  --with-morph --barred-weight 4 --beta 1e-4 --skip-weight 2.0 \
  --a2-weight 0.0 \
  --n-resample 80000 --n-prior-samples 4 --n-interp 5 \
  --note "barred-heavier morph weak-KL"
python - <<'PY'
import json
from pathlib import Path
root = Path("runs/ml/field_maps")
rows = []
for d in sorted(root.glob("creative_latent_*")):
    v = d / "verdict.json"
    if not v.is_file():
        continue
    j = json.loads(v.read_text())
    bar = next((e for e in j.get("examples", []) if e["name"].startswith("bar")), {})
    quiet = next((e for e in j.get("examples", []) if e["name"].startswith("quiet")), {})
    rows.append({
        "dir": d.name,
        "better": j.get("better_than_overnight_vae"),
        "bar_part_mu": bar.get("a2_part_synth_mu"),
        "bar_map_mu": bar.get("a2_map_synth_mu"),
        "bar_prior_part": bar.get("a2_part_prior0"),
        "bar_prior_map": bar.get("a2_map_prior_mean"),
        "quiet_prior_map": quiet.get("a2_map_prior_mean"),
        "interp_gap": j.get("interp_a2_gap"),
        "teacher_feat": j.get("teacher_feature_interp"),
    })
(root / "creative_latent_sweep_index.json").write_text(json.dumps(rows, indent=2))
best = None
for r in rows:
    score = r.get("bar_part_mu") or r.get("bar_prior_part") or r.get("bar_map_mu") or 0
    if best is None or (score or 0) > (best[1] or 0):
        best = (r["dir"], score)
print("sweep rows", len(rows), "best", best)
if best:
    (root / "LATEST").write_text(str((root / best[0]).resolve()) + "\n")
PY

echo "=== done $(date -Is) log=${LOG} ==="
