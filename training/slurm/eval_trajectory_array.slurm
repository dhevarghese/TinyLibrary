#!/bin/bash
#SBATCH --job-name=tl-traj
#SBATCH --array=0-26%8
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
: "${TINYLIBRARY_CODE_DIR:?Set TINYLIBRARY_CODE_DIR to the absolute release-repository path}"
source "$TINYLIBRARY_CODE_DIR/training/slurm/common.sh"

# Learning-trajectory fast evals: for one grid run, export EVERY saved
# checkpoint's decoder and run the fast zero-shot suite, labelling results
# with --revision_name step_N -> results/<run_name>/step_N/...
# Override RUN_GLOB to target other grids (e.g. tl_*ep10*).
RUN_GLOB=${RUN_GLOB:-tl_*ep5*}


RUN=$(ls -d "$BASE/checkpoints"/$RUN_GLOB | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
[ -z "$RUN" ] && { echo "no run for index $SLURM_ARRAY_TASK_ID"; exit 0; }
NAME=$(basename "$RUN")
TMP="$BASE/hf_exports_traj/$NAME"
EVAL_DIR="evaluation_data/fast_eval"

for ckpt in $(ls "$RUN" | sort -t_ -k2 -n); do
    step=${ckpt#step_}
    marker="$BABYLM_EVAL_DIR/strict/results/$NAME/$ckpt/zero_shot/causal/entity_tracking/entity_tracking_fast/best_temperature_report.txt"
    if [ -f "$marker" ]; then
        echo "skip $NAME/$ckpt (done)"
        continue
    fi
    echo "=== $NAME/$ckpt ==="
    rm -rf "$TMP"; mkdir -p "$TMP"
    cd "$NANOVLM_DIR"
    PYTHONPATH=.:"$TINYLIBRARY_CODE_DIR/training" python "$TINYLIBRARY_CODE_DIR/training/export_decoder_hf.py" \
        --checkpoint "$RUN/$ckpt" --out "$TMP" || { echo "export failed for $ckpt"; continue; }
    cd "$BABYLM_EVAL_DIR/strict"
    for task_data in "blimp blimp_fast" "blimp supplement_fast" \
                     "ewok ewok_fast" "entity_tracking entity_tracking_fast"; do
        set -- $task_data
        python -m evaluation_pipeline.sentence_zero_shot.run \
            --model_path_or_name "$TMP" --backend causal \
            --task "$1" --data_path "$EVAL_DIR/$2" \
            --save_predictions --revision_name "$ckpt"
    done
done
rm -rf "$TMP"
echo "trajectory done for $NAME"
