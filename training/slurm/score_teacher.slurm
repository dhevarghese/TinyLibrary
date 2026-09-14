#!/bin/bash
#SBATCH --job-name=tl-score
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
: "${TINYLIBRARY_CODE_DIR:?Set TINYLIBRARY_CODE_DIR to the absolute release-repository path}"
source "$TINYLIBRARY_CODE_DIR/training/slurm/common.sh"

# Score the manifest with an alternate teacher LM (teacher-robustness check).
# Usage: sbatch --export=ALL,SCORER=Qwen/Qwen3-0.6B,OUT=manifest_qwen_raw.jsonl,BS=8 score_teacher.slurm


cd "$BASE"
python "$TINYLIBRARY_CODE_DIR/training/compute_perplexity.py" \
    --manifest manifest_scored.jsonl \
    --out "$BASE/${OUT:?set OUT}" \
    --scorer "${SCORER:?set SCORER}" \
    --batch_size "${BS:-8}"
# Verify that scoring produced one row per TinyLibrary sample.
LINES=$(wc -l < "$BASE/$OUT")
[ "$LINES" -eq 128823 ] || { echo "INCOMPLETE OUTPUT: $LINES lines"; exit 1; }
echo "OK: $OUT complete ($LINES lines)"
