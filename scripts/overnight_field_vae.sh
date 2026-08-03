#!/usr/bin/env bash
# Overnight field-conditional VAE sweep: larger latents, then longer/lower-LR fallback.
# CPU + OpenMP; batch=1. Does not commit.
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"

ROOT="runs/ml/field_maps"
LOG="${ROOT}/overnight_vae_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ROOT"
exec > >(tee -a "$LOG") 2>&1
echo "=== overnight field VAE start $(date -Is) OMP=$OMP_NUM_THREADS ==="
echo "log → $LOG"

run_one() {
  local name="$1"; shift
  local out="${ROOT}/${name}"
  mkdir -p "$out"
  echo ""
  echo "######## RUN $name $(date -Is) ########"
  echo "cmd: $*"
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    python scripts/smoke_field_vae.py --out "$out" "$@" \
    || { echo "FAIL $name exit=$?"; return 0; }
  echo "######## DONE $name $(date -Is) ########"
  if [[ -f "$out/verdict.json" ]]; then
    python - <<PY
import json
from pathlib import Path
v = json.loads(Path("$out/verdict.json").read_text())
print("summary:", {
  "latent": v.get("latent_dim"),
  "best_loss": round(v.get("best_loss", -1), 4),
  "epochs": v.get("epochs_ran"),
  "bar_recon": next((r["a2_resampled"] for r in v.get("recon",[]) if r["name"].startswith("bar")), None),
  "mu_sep": v.get("morphology_mu_separated"),
  "prior_sep": v.get("morphology_prior_separated"),
  "bar_ok": v.get("bar_recon_competitive"),
  "rss_mb": v.get("peak_rss_mb"),
})
PY
  fi
}

# Shared crisp-track knobs (Fourier off, moment ≥ dens).
COMMON=(
  --preset progressive --disk-n-pix 128 --moment-set disp
  --batch 1 --max-snap 28
  --dens-weight 4 --moment-weight 6 --a2-weight 0
  --beta 5e-4 --base-channels 32 --bottleneck-channels 64 --enc-grid 4
  --skip-dropout 1.0 --skip-recon-weight 0.1
  --n-resample 80000 --n-prior-samples 2
  --omp-threads "$OMP_NUM_THREADS"
)

# 1) Push larger latents at matched LR / epochs.
run_one "vae_latent_z192_2026-07-25" "${COMMON[@]}" \
  --latent-dim 192 --lr 1e-4 --min-lr 3e-5 --warmup-epochs 3 \
  --epochs 45 --patience 16 \
  --note "overnight: larger latent 192 @128² Fourier-off"

run_one "vae_latent_z256_2026-07-25" "${COMMON[@]}" \
  --latent-dim 256 --lr 1e-4 --min-lr 3e-5 --warmup-epochs 3 \
  --epochs 45 --patience 16 \
  --note "overnight: larger latent 256 @128² Fourier-off"

run_one "vae_latent_z384_2026-07-25" "${COMMON[@]}" \
  --latent-dim 384 --lr 1e-4 --min-lr 3e-5 --warmup-epochs 3 \
  --epochs 40 --patience 14 \
  --note "overnight: larger latent 384 @128² Fourier-off"

# 2) Longer + lower LR fallback (primary hypothesis if quality stalls).
#    Use best latent from above if verdict looks promising; else 256.
BEST_Z=256
for cand in z384 z256 z192; do
  d="${ROOT}/vae_latent_${cand}_2026-07-25/verdict.json"
  if [[ -f "$d" ]]; then
    ok=$(python -c "import json; v=json.load(open('$d')); print(int(bool(v.get('bar_recon_competitive') or v.get('morphology_mu_separated'))))")
    if [[ "$ok" == "1" ]]; then
      BEST_Z=${cand#z}
      break
    fi
  fi
done
echo "longer/lower-LR fallback using latent_dim=$BEST_Z"

run_one "vae_latent_z${BEST_Z}_long_lr3e5_2026-07-25" "${COMMON[@]}" \
  --latent-dim "$BEST_Z" --lr 3e-5 --min-lr 1e-5 --warmup-epochs 5 \
  --epochs 80 --patience 25 \
  --note "overnight fallback: longer train + lower LR (latent=$BEST_Z)"

# Write a sweep index.
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
    })
out = root / "vae_latent_sweep_index.json"
out.write_text(json.dumps(rows, indent=2))
print(f"wrote {out} ({len(rows)} runs)")
# brief markdown
md = ["# Field VAE latent sweep", ""]
md.append("| dir | z | disk | lr | epochs | best_loss | bar A₂ | quiet A₂ | μ-sep | prior-sep | RSS MB |")
md.append("|-----|---|------|----|--------|-----------|--------|----------|-------|-----------|--------|")
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
print("wrote", root / "vae_latent_sweep.md")
PY

echo "=== overnight field VAE done $(date -Is) ==="
