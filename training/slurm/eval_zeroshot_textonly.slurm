#!/bin/bash
#SBATCH --job-name=tl-zs-txt
#SBATCH --array=0-8%6
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
: "${TINYLIBRARY_CODE_DIR:?Set TINYLIBRARY_CODE_DIR to the absolute release-repository path}"
source "$TINYLIBRARY_CODE_DIR/training/slurm/common.sh"

# Text-only-control zero-shot eval: export each text-only decoder, run the BabyLM
# text suite. Separate hf_exports / results dirs from the multimodal grid so the
# two never collide during collation.


RUN=$(ls -d "$BASE/checkpoints_textonly"/tl_*ep10* | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
[ -z "$RUN" ] && { echo "no run for index $SLURM_ARRAY_TASK_ID"; exit 0; }
NAME=$(basename "$RUN")
LAST=$(ls "$RUN" | sort -t_ -k2 -n | tail -1)
MODEL="$BASE/hf_exports_textonly/$NAME"
if [ ! -f "$MODEL/model.safetensors" ]; then
    mkdir -p "$BASE/hf_exports_textonly"
    cd "$NANOVLM_DIR"
    PYTHONPATH=.:"$TINYLIBRARY_CODE_DIR/training" python "$TINYLIBRARY_CODE_DIR/training/export_decoder_hf.py" \
        --checkpoint "$RUN/$LAST" --out "$MODEL"
fi

cd "$BABYLM_EVAL_DIR/strict"
echo "text-only eval $MODEL (from $NAME/$LAST)"
EVAL_DIR="evaluation_data/full_eval"
OUT="results_textonly"
for task_data in "blimp blimp_filtered" "blimp supplement_filtered" \
                 "ewok ewok_filtered" "entity_tracking entity_tracking" "comps comps"; do
    set -- $task_data
    python -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$MODEL" --backend causal \
        --task "$1" --data_path "$EVAL_DIR/$2" --output_dir "$OUT" --save_predictions
done
