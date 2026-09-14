#!/bin/bash
#SBATCH --job-name=tl-untrained
#SBATCH --array=0-2
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
: "${TINYLIBRARY_CODE_DIR:?Set TINYLIBRARY_CODE_DIR to the absolute release-repository path}"
source "$TINYLIBRARY_CODE_DIR/training/slurm/common.sh"

# Full-suite zero-shot eval of UNTRAINED decoders (step_0 checkpoints), one
# per seed. Gives the Table 2 "untrained" reference row under the same full
# protocol as the trained models (review fix: fast-vs-full comparability).


SEED=$SLURM_ARRAY_TASK_ID
RUN=$(ls -d "$BASE/checkpoints"/tl_random_staged_ep10_*_seed${SEED}_* | head -1)
[ -z "$RUN" ] && { echo "no run for seed $SEED"; exit 1; }
MODEL="$BASE/hf_exports/untrained_seed${SEED}"
if [ ! -f "$MODEL/model.safetensors" ]; then
    cd "$NANOVLM_DIR"
    PYTHONPATH=.:"$TINYLIBRARY_CODE_DIR/training" python "$TINYLIBRARY_CODE_DIR/training/export_decoder_hf.py" \
        --checkpoint "$RUN/step_0" --out "$MODEL"
fi

cd "$BABYLM_EVAL_DIR/strict"
echo "evaluating untrained $MODEL (from $RUN/step_0)"
EVAL_DIR="evaluation_data/full_eval"
for task_data in "blimp blimp_filtered" "blimp supplement_filtered" \
                 "entity_tracking entity_tracking" "comps comps" \
                 "ewok ewok_filtered"; do
    set -- $task_data
    python -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$MODEL" --backend causal \
        --task "$1" --data_path "$EVAL_DIR/$2" --save_predictions
done
