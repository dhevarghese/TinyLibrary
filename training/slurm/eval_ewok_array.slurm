#!/bin/bash
#SBATCH --job-name=tl-ewok
#SBATCH --array=0-26%8
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
: "${TINYLIBRARY_CODE_DIR:?Set TINYLIBRARY_CODE_DIR to the absolute release-repository path}"
source "$TINYLIBRARY_CODE_DIR/training/slurm/common.sh"

# EWoK full zero-shot for one exported decoder (run after eval_zeroshot_array).


EXPORT_GLOB=${EXPORT_GLOB:-tl_*ep5*}
MODEL=$(ls -d "$BASE/hf_exports"/$EXPORT_GLOB | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
[ -z "$MODEL" ] && { echo "no model for index $SLURM_ARRAY_TASK_ID"; exit 0; }

cd "$BABYLM_EVAL_DIR/strict"
python -m evaluation_pipeline.sentence_zero_shot.run \
    --model_path_or_name "$MODEL" --backend causal \
    --task ewok --data_path evaluation_data/full_eval/ewok_filtered --save_predictions
