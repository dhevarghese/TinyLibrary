#!/bin/bash
#SBATCH --job-name=tl-curr
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
: "${TINYLIBRARY_CODE_DIR:?Set TINYLIBRARY_CODE_DIR to the absolute release-repository path}"
source "$TINYLIBRARY_CODE_DIR/training/slurm/common.sh"

# Usage: sbatch train_curriculum.slurm --order developmental --schedule staged --epochs 5 --seed 0

# Configure W&B with WANDB_PROJECT/WANDB_ENTITY, or pass --no_log_wandb.

cd "$NANOVLM_DIR"
PYTHONPATH=.:"$TINYLIBRARY_CODE_DIR/training:${PYTHONPATH:-}" python "$TINYLIBRARY_CODE_DIR/training/train_curriculum.py" \
    --manifest "$BASE/manifest_scored.jsonl" \
    --image_root "$BASE/images" \
    --checkpoint_path "$BASE/checkpoints" \
    "$@"
