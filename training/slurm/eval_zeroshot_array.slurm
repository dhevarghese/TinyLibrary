#!/bin/bash
#SBATCH --job-name=tl-zeroshot
#SBATCH --array=0-26%8
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
: "${TINYLIBRARY_CODE_DIR:?Set TINYLIBRARY_CODE_DIR to the absolute release-repository path}"
source "$TINYLIBRARY_CODE_DIR/training/slurm/common.sh"

# BabyLM 2026 strict zero-shot eval (minus gated EWoK) on one exported decoder.


# Pick the i-th run, export its final decoder if not done yet, then eval.
# Override RUN_GLOB to target other grids (e.g. RUN_GLOB="tl_*ep10*").
RUN_GLOB=${RUN_GLOB:-tl_*ep5*}
RUN=$(ls -d "$BASE/checkpoints"/$RUN_GLOB | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
[ -z "$RUN" ] && { echo "no run for index $SLURM_ARRAY_TASK_ID"; exit 0; }
NAME=$(basename "$RUN")
LAST=$(ls "$RUN" | sort -t_ -k2 -n | tail -1)
MODEL="$BASE/hf_exports/$NAME"
if [ ! -f "$MODEL/model.safetensors" ]; then
    mkdir -p "$BASE/hf_exports"
    cd "$NANOVLM_DIR"
    PYTHONPATH=.:"$TINYLIBRARY_CODE_DIR/training" python "$TINYLIBRARY_CODE_DIR/training/export_decoder_hf.py" \
        --checkpoint "$RUN/$LAST" --out "$MODEL"
fi

cd "$BABYLM_EVAL_DIR/strict"
echo "evaluating $MODEL (from $NAME/$LAST)"

EVAL_DIR="evaluation_data/full_eval"
for task_data in "blimp blimp_filtered" "blimp supplement_filtered" \
                 "entity_tracking entity_tracking" "comps comps"; do
    set -- $task_data
    python -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$MODEL" --backend causal \
        --task "$1" --data_path "$EVAL_DIR/$2" --save_predictions
done
python -m evaluation_pipeline.reading.run \
    --model_path_or_name "$MODEL" --backend causal \
    --data_path "$EVAL_DIR/reading/reading_data.csv"
