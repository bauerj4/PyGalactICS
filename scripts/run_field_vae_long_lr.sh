#!/usr/bin/env bash
# Longer + lower-LR field VAE fallback (primary hypothesis after latent sweep stalls).
# Also bumps enc_grid / bottleneck slightly so z can carry more spatial morphology —
# still a single global z bottleneck (does not abandon VAE).
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"

Z="${1:-256}"
OUT="runs/ml/field_maps/vae_latent_z${Z}_long_lr3e5_enc8_2026-07-25"
mkdir -p "$OUT"
LOG="$OUT/train.log"
echo "=== long/low-LR fallback z=$Z → $OUT $(date -Is) ===" | tee "$LOG"

CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS="$OMP_NUM_THREADS" \
  python scripts/smoke_field_vae.py \
  --out "$OUT" \
  --preset progressive --disk-n-pix 128 --moment-set disp \
  --batch 1 --max-snap 28 \
  --dens-weight 4 --moment-weight 6 --a2-weight 0 \
  --beta 1e-4 \
  --latent-dim "$Z" \
  --base-channels 32 --bottleneck-channels 96 --enc-grid 8 \
  --skip-dropout 1.0 --skip-recon-weight 0.0 \
  --lr 3e-5 --min-lr 1e-5 --warmup-epochs 5 \
  --epochs 100 --patience 30 \
  --n-resample 80000 --n-prior-samples 2 \
  --omp-threads "$OMP_NUM_THREADS" \
  --note "fallback: longer+lower-LR (3e-5→1e-5), enc_grid=8, bottleneck=96, β=1e-4, skip_recon=0; latent=$Z" \
  2>&1 | tee -a "$LOG"

echo "=== done $(date -Is) ===" | tee -a "$LOG"
# refresh sweep index if present
if [[ -f scripts/overnight_field_vae.sh ]]; then
  python - <<'PY'
import json
from pathlib import Path
root = Path("runs/ml/field_maps")
rows = []
for p in sorted(root.glob("vae_latent_*/verdict.json")):
    v = json.loads(p.read_text())
    rows.append({
        "dir": p.parent.name,
        "latent_dim": v.get("latent_dim"),
        "disk_n_pix": v.get("disk_n_pix"),
        "lr": v.get("lr"),
        "epochs_ran": v.get("epochs_ran"),
        "best_loss": v.get("best_loss"),
        "bar_recon_a2": next((r["a2_resampled"] for r in v.get("recon", []) if str(r.get("name","")).startswith("bar")), None),
        "quiet_recon_a2": next((r["a2_resampled"] for r in v.get("recon", []) if str(r.get("name","")).startswith("quiet")), None),
        "morphology_mu_separated": v.get("morphology_mu_separated"),
        "morphology_prior_separated": v.get("morphology_prior_separated"),
        "bar_recon_competitive": v.get("bar_recon_competitive"),
        "peak_rss_mb": v.get("peak_rss_mb"),
        "train_wall_s": v.get("train_wall_s"),
        "a2_weight": v.get("a2_weight"),
        "note": (v.get("user_note") or "")[:80],
    })
(root / "vae_latent_sweep_index.json").write_text(json.dumps(rows, indent=2))
md = ["# Field VAE latent sweep", ""]
md.append("| dir | z | disk | lr | epochs | best_loss | bar A₂ | quiet A₂ | μ-sep | prior-sep | RSS |")
md.append("|-----|---|------|----|--------|-----------|--------|----------|-------|-----------|-----|")
for r in rows:
    md.append(
        f"| `{r['dir']}` | {r['latent_dim']} | {r['disk_n_pix']} | {r['lr']} | "
        f"{r['epochs_ran']} | {None if r['best_loss'] is None else round(r['best_loss'],3)} | "
        f"{None if r['bar_recon_a2'] is None else round(r['bar_recon_a2'],3)} | "
        f"{None if r['quiet_recon_a2'] is None else round(r['quiet_recon_a2'],3)} | "
        f"{r['morphology_mu_separated']} | {r['morphology_prior_separated']} | "
        f"{None if r['peak_rss_mb'] is None else round(r['peak_rss_mb'])} |"
    )
(root / "vae_latent_sweep.md").write_text("\n".join(md) + "\n")
print(f"updated sweep index ({len(rows)} runs)")
PY
fi
