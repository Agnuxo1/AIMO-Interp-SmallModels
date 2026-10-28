#!/usr/bin/env bash
# Run from the src/ directory. Set PY to the runtime-matching interpreter (see setup_env.sh).
set -euo pipefail
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore
export AIMO_FEATURES=${AIMO_FEATURES:-features} OMP_NUM_THREADS=4
${PY:-python} ablate.py "$@" 2>&1 | grep -v -E "Loading weights|Warning"
