#!/bin/bash
#SBATCH --job-name=tl-mmeval
#SBATCH --array=0-35%8
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
: "${TINYLIBRARY_CODE_DIR:?Set TINYLIBRARY_CODE_DIR to the absolute release-repository path}"
source "$TINYLIBRARY_CODE_DIR/training/slurm/common.sh"

# Multimodal zero-shot eval (VQA / Winoground) on one nanoVLM VLM checkpoint,
# via the official completion-ranking scorer + our nanoVLM wrapper
# (training/run_multimodal.py). Uses the FULL VLM (vision + projector + decoder),
# not the exported text-only decoder.
#
#   RUN_GLOB  : which grid to target (default all ep10 conditions)
#   MM_TASKS  : space-separated subset of {vqa, winoground} (default both)

export HF_DATASETS_TRUST_REMOTE_CODE=1   # VQAv2/winoground are script-based datasets (datasets 3.6.0)

RUN_GLOB=${RUN_GLOB:-tl_*ep10*}
MM_TASKS=${MM_TASKS:-"vqa winoground"}
RUN=$(ls -d "$BASE/checkpoints"/$RUN_GLOB | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
[ -z "$RUN" ] && { echo "no run for index $SLURM_ARRAY_TASK_ID"; exit 0; }
NAME=$(basename "$RUN")
LAST=$(ls "$RUN" | sort -t_ -k2 -n | tail -1)   # true final step (bare step_N names)
MODEL="$RUN/$LAST"

cd "$BABYLM_EVAL_DIR/strict"
echo "multimodal eval: $NAME/$LAST  tasks=[$MM_TASKS]"

# run_multimodal.py imports the official pipeline (cwd = strict) + nanoVLM (NANOVLM_DIR).
export PYTHONPATH="$NANOVLM_DIR:${PYTHONPATH:-}"

for task in $MM_TASKS; do
    if [ "$task" = "vqa" ]; then
        # Local COCO val2014 images (by image_id), bypassing the ~40GB HF VQAv2 script.
        LOCAL_COCO_DIR="${LOCAL_COCO_DIR:-$BASE/coco/val2014}" \
        python "$TINYLIBRARY_CODE_DIR/training/run_multimodal.py" \
            --model_path_or_name "$MODEL" --backend causal \
            --task vqa --data_path "evaluation_data/full_eval/vqa_filtered" \
            --images_path=local-coco --image_split=validation \
            --batch_size=1 --save_predictions
    elif [ "$task" = "winoground" ]; then
        python "$TINYLIBRARY_CODE_DIR/training/run_multimodal.py" \
            --model_path_or_name "$MODEL" --backend causal \
            --task winoground --data_path "evaluation_data/full_eval/winoground_filtered" \
            --images_path=facebook/winoground --image_split=test \
            --batch_size=1 --save_predictions
    fi
done
