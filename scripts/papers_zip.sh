#!/usr/bin/env bash
# Bundle MNRAS Part~1 manuscript artefacts for Google Drive (no secrets).
# Usage: from repo root — make papers-zip
#        or: bash scripts/papers_zip.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PAPER_DIR="${ROOT}/papers/mnras_noneq_ics"
OUT_DIR="${ROOT}/dist"
STAMP="$(date -u +%Y%m%d)"
ZIP_NAME="mnras_noneq_ics_bundle_${STAMP}.zip"
ZIP_PATH="${OUT_DIR}/${ZIP_NAME}"
STAGE="${OUT_DIR}/.papers_zip_stage_$$"

cleanup() { rm -rf "${STAGE}"; }
trap cleanup EXIT

if [[ ! -d "${PAPER_DIR}" ]]; then
  echo "ERROR: missing ${PAPER_DIR}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
rm -rf "${STAGE}"
mkdir -p "${STAGE}/mnras_noneq_ics"

# --- Manuscript sources (no secrets) ---
# Copy only top-level manuscript files (avoid recursing into huge trees).
shopt -s nullglob
for f in \
  "${PAPER_DIR}"/*.{tex,bib,bst,cls,pdf,md,bbl,blg} \
  "${PAPER_DIR}"/MANIFEST.txt
do
  [[ -f "$f" ]] || continue
  cp -a "$f" "${STAGE}/mnras_noneq_ics/"
done
shopt -u nullglob

# Live figures referenced by the .tex (extension-less includegraphics names)
# plus print PDF/EPS under figures_eps/. Do NOT ship figures/archive/ or
# unrelated PNG dumps sitting beside live story figures.
mkdir -p "${STAGE}/mnras_noneq_ics/figures" "${STAGE}/mnras_noneq_ics/figures_eps"
TEX="${PAPER_DIR}/mnras_noneq_ics.tex"
if [[ -f "${TEX}" ]]; then
  mapfile -t BASES < <(grep -oE '\\includegraphics(\[[^]]*\])?\{[^}]+\}' "${TEX}" \
    | sed -E 's/.*\{([^}]+)\}/\1/' | sed -E 's|^figures(_eps)?/||; s/\.(png|pdf|eps)$//' | sort -u)
else
  BASES=()
fi

copied_src=0
for base in "${BASES[@]:-}"; do
  [[ -z "${base}" ]] && continue
  for ext in png pdf eps; do
    src="${PAPER_DIR}/figures/${base}.${ext}"
    if [[ -f "${src}" ]]; then
      cp -a "${src}" "${STAGE}/mnras_noneq_ics/figures/"
      copied_src=$((copied_src + 1))
    fi
  done
  for ext in pdf eps; do
    src="${PAPER_DIR}/figures_eps/${base}.${ext}"
    if [[ -f "${src}" ]]; then
      cp -a "${src}" "${STAGE}/mnras_noneq_ics/figures_eps/"
    fi
  done
done

# Small scoreboard / journal text (no large npz dumps)
if [[ -d "${PAPER_DIR}/results" ]]; then
  mkdir -p "${STAGE}/mnras_noneq_ics/results"
  find "${PAPER_DIR}/results" -maxdepth 1 -type f \
    \( -name '*.md' -o -name '*.txt' -o -name '*.json' -o -name '*.csv' \) \
    -exec cp -a {} "${STAGE}/mnras_noneq_ics/results/" \;
fi

# Local MNRAS class extras (cls/bst already copied at top level)
if [[ -d "${PAPER_DIR}/latex/mnras" ]]; then
  mkdir -p "${STAGE}/mnras_noneq_ics/latex/mnras"
  for f in README.txt mnras.cls mnras.bst mnras_guide.tex example.bib; do
    [[ -f "${PAPER_DIR}/latex/mnras/${f}" ]] && cp -a "${PAPER_DIR}/latex/mnras/${f}" "${STAGE}/mnras_noneq_ics/latex/mnras/"
  done
fi

# Manifest: what is in the zip vs what lives only on disk under runs/
cat > "${STAGE}/mnras_noneq_ics/BUNDLE_CONTENTS.md" <<EOF
# MNRAS Part~1 Drive bundle

Built by \`make papers-zip\` / \`scripts/papers_zip.sh\`.

## Included

| Path | Contents |
|------|----------|
| \`*.tex\`, \`*.bib\`, \`mnras.cls\`, \`mnras.bst\` | Manuscript + BibTeX |
| \`mnras_noneq_ics.pdf\` | Built PDF (if present) |
| \`README.md\` | Build + BibTeX + figure notes |
| \`figures/\` | **Only** live story figures cited in the \`.tex\` (PNG/PDF) |
| \`figures_eps/\` | Matching print PDF + EPS for those figures |
| \`results/*.{md,json,csv,txt}\` | Scoreboards / journals (text only) |
| \`latex/mnras/\` | CTAN class essentials (if present locally) |

Live figure basenames in this build (${#BASES[@]}):
$(printf -- '- %s\n' "${BASES[@]:-}")

Source figure files copied: ${copied_src}

## Not included (regenerate or sync separately)

| Path | Why |
|------|-----|
| \`figures/archive/\` and unused PNGs under \`figures/\` | Demoted / bulk (~hundreds of MB) |
| \`runs/ml/field_maps/latent_theta_*\` | Large evolve dumps / checkpoints |
| \`runs/mw_morton_corpus*\` | Campaign particle dumps |
| \`.tools/\`, \`.venv/\`, credentials, \`.env\` | Tooling / secrets |

Primary evidence runs (on the machine that produced the paper):

\`\`\`
runs/ml/field_maps/latent_theta_gen_*
runs/ml/field_maps/latent_theta_evolve_match_*
runs/ml/field_maps/latent_theta_ood_*
runs/ml/field_maps/latent_theta_bar_sweep_*
\`\`\`

Teacher checkpoint (encode library; not in zip):

\`\`\`
runs/ml/field_maps/fft_morph_ft_long_*/multitower_slice_ae.pt
\`\`\`

No API keys, tokens, or \`.env\` files are packaged.
EOF

# Explicit secret scrub
find "${STAGE}" \( -name '.env' -o -name '*.pem' -o -name '*credentials*' \
  -o -name '*secret*' -o -name '.git' \) -prune -exec rm -rf {} + 2>/dev/null || true

rm -f "${ZIP_PATH}" "${OUT_DIR}/mnras_noneq_ics_bundle.zip"
(
  cd "${STAGE}"
  zip -rq "${ZIP_PATH}" mnras_noneq_ics
)
ln -sfn "${ZIP_NAME}" "${OUT_DIR}/mnras_noneq_ics_bundle.zip"

BYTES="$(wc -c < "${ZIP_PATH}" | tr -d ' ')"
echo "Wrote ${ZIP_PATH} (${BYTES} bytes)"
echo "Alias: ${OUT_DIR}/mnras_noneq_ics_bundle.zip -> ${ZIP_NAME}"
du -sh "${STAGE}/mnras_noneq_ics" "${STAGE}/mnras_noneq_ics/figures" "${STAGE}/mnras_noneq_ics/figures_eps" 2>/dev/null || true
