#!/bin/bash
#SBATCH --job-name=tl-txt
#SBATCH --array=0-8%6
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
: "${TINYLIBRARY_CODE_DIR:?Set TINYLIBRARY_CODE_DIR to the absolute release-repository path}"
source "$TINYLIBRARY_CODE_DIR/training/slurm/common.sh"

# Text-only control grid: 3 conditions x 3 seeds, decoder trained on the SAME
# text with NO image tokens (modality control). 10 epochs, from-scratch LM.

# Configure W&B with WANDB_PROJECT/WANDB_ENTITY, or add --no_log_wandb below.
mkdir -p "$BASE/checkpoints_textonly"

ORDERS=(random random random perplexity perplexity perplexity developmental developmental developmental)
SEEDS=(0 1 2 0 1 2 0 1 2)
ORDER=${ORDERS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "text-only train: order=$ORDER schedule=staged seed=$SEED ep10"

cd "$NANOVLM_DIR"
PYTHONPATH=.:"$TINYLIBRARY_CODE_DIR/training:${PYTHONPATH:-}" python "$TINYLIBRARY_CODE_DIR/training/train_curriculum.py" \
    --manifest "$BASE/manifest_scored.jsonl" \
    --image_root "$BASE/images" \
    --checkpoint_path "$BASE/checkpoints_textonly" \
    --order "$ORDER" --schedule staged --epochs 10 --seed "$SEED" \
    --lm_init scratch --text_only
