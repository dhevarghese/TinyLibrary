#!/bin/bash
#SBATCH --job-name=tl-merge
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
: "${TINYLIBRARY_CODE_DIR:?Set TINYLIBRARY_CODE_DIR to the absolute release-repository path}"
source "$TINYLIBRARY_CODE_DIR/training/slurm/common.sh"

# Merge alternate-teacher scores into manifest_scored_multi.jsonl and print
# teacher-agreement stats. Includes gemma column only if that scoring ran.


cd "$BASE"
ALTS="--alt qwen=manifest_qwen_raw.jsonl"
if [ -f manifest_gemma_raw.jsonl ] && [ "$(wc -l < manifest_gemma_raw.jsonl)" -eq 128823 ]; then
    ALTS="$ALTS --alt gemma=manifest_gemma_raw.jsonl"
fi
python "$TINYLIBRARY_CODE_DIR/training/merge_teacher_scores.py" \
    --base manifest_scored.jsonl $ALTS --out manifest_scored_multi.jsonl
LINES=$(wc -l < manifest_scored_multi.jsonl)
[ "$LINES" -eq 128823 ] || { echo "INCOMPLETE MERGE: $LINES"; exit 1; }
echo "OK: manifest_scored_multi.jsonl complete"
