#!/usr/bin/env bash

# Shared, site-independent environment setup for the Slurm templates.
set -euo pipefail

: "${TINYLIBRARY_CODE_DIR:?Set TINYLIBRARY_CODE_DIR to the absolute release-repository path}"
case "$TINYLIBRARY_CODE_DIR" in
    /*) ;;
    *) echo "TINYLIBRARY_CODE_DIR must be an absolute path" >&2; exit 2 ;;
esac
: "${TINYLIBRARY_BASE:?Set TINYLIBRARY_BASE to the experiment workspace}"
BASE="$TINYLIBRARY_BASE"

if [ -n "${TINYLIBRARY_ACTIVATE:-}" ]; then
    # shellcheck disable=SC1090
    source "$TINYLIBRARY_ACTIVATE"
fi

export HF_HOME="${HF_HOME:-$BASE/hf_cache}"
export NANOVLM_DIR="${NANOVLM_DIR:-$BASE/nanoVLM}"
export BABYLM_EVAL_DIR="${BABYLM_EVAL_DIR:-$BASE/babylm-eval}"
export WANDB_DIR="${WANDB_DIR:-$BASE/wandb}"

mkdir -p "$BASE/logs" "$WANDB_DIR"
