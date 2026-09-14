#!/bin/bash
#SBATCH --job-name=tl-audit
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
: "${TINYLIBRARY_CODE_DIR:?Set TINYLIBRARY_CODE_DIR to the absolute release-repository path}"
source "$TINYLIBRARY_CODE_DIR/training/slurm/common.sh"

# Age-label stability audit: out-of-family classifier vs Mistral bands.
: "${TINYLIBRARY_OCR_DIR:?Set TINYLIBRARY_OCR_DIR to the cleaned book-text directory}"
TINYLIBRARY_PROMPT_DIR="${TINYLIBRARY_PROMPT_DIR:-$TINYLIBRARY_CODE_DIR/playground/prompts/age-group}"
TINYLIBRARY_LABELS="${TINYLIBRARY_LABELS:-$BASE/new4-age-category-library.jsonl}"
TINYLIBRARY_AUDIT_OUT="${TINYLIBRARY_AUDIT_OUT:-$BASE/audit_qwen.jsonl}"

cd "$BASE"
python "$TINYLIBRARY_CODE_DIR/training/audit_age_labels.py" \
    --library_dir "$TINYLIBRARY_OCR_DIR" \
    --prompt_dir "$TINYLIBRARY_PROMPT_DIR" \
    --labels "$TINYLIBRARY_LABELS" \
    --out "$TINYLIBRARY_AUDIT_OUT" \
    --model "${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
