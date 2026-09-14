#!/bin/bash
#SBATCH --job-name=tl-export
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
: "${TINYLIBRARY_CODE_DIR:?Set TINYLIBRARY_CODE_DIR to the absolute release-repository path}"
source "$TINYLIBRARY_CODE_DIR/training/slurm/common.sh"

# Export the final-step decoder of every grid run to HF LlamaForCausalLM
# format under $BASE/hf_exports/<run_name>, for babylm-eval.

mkdir -p "$BASE/hf_exports"

cd "$NANOVLM_DIR"
for run in "$BASE/checkpoints"/tl_*ep5*; do
    name=$(basename "$run")
    last=$(ls "$run" | sort -t_ -k2 -n | tail -1)
    out="$BASE/hf_exports/$name"
    if [ -f "$out/model.safetensors" ]; then
        echo "skip $name (already exported)"
        continue
    fi
    echo "exporting $name/$last"
    PYTHONPATH=.:"$TINYLIBRARY_CODE_DIR/training" python "$TINYLIBRARY_CODE_DIR/training/export_decoder_hf.py" \
        --checkpoint "$run/$last" --out "$out" || echo "FAILED: $name"
done
echo "done: $(ls "$BASE/hf_exports" | wc -l) exports"
