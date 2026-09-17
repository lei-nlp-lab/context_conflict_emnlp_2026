#!/bin/bash
# Response generation for ContextConflict (step 1 of the evaluation pipeline).
#
# Queries one model on every sample of the selected conflict types and writes
#   results/responses/<MODEL_NAME>/<conflict_type>/[<subfolder>/]<id>.json
# with the model output stored under the key "<MODEL_NAME>_response". Files that
# already hold a valid response are skipped, so re-running only fills gaps.
#
# Usage (from anywhere; the script cd's to the repository root):
#   # API model through an OpenAI-compatible endpoint
#   export OPENAI_API_KEY=...            # optionally: export OPENAI_BASE_URL=...
#   bash evaluation/scripts/run_generate_responses.sh api gpt-5
#   API_KEY_ENV=DEEPSEEK_API_KEY OPENAI_BASE_URL=https://api.deepseek.com \
#     bash evaluation/scripts/run_generate_responses.sh api deepseek-chat
#   # local Hugging Face model: <backend> <model id or path> <short name>
#   bash evaluation/scripts/run_generate_responses.sh local meta-llama/Llama-3.1-8B-Instruct llama-3.1-8b-instruct
#   # restrict to some conflict types (default: all six)
#   CONFLICT_TYPES="temporal_conflict misinformation_conflict" \
#     bash evaluation/scripts/run_generate_responses.sh api gpt-5
#
# Example SLURM header (uncomment and adapt if you submit this file with sbatch;
# when using sbatch, submit from the repository root or add --chdir=<repo root>):
# #SBATCH --job-name=gen-responses
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=64gb
# #SBATCH --gpus=2                 # local models only
# #SBATCH --time=2-00:00:00
# #SBATCH --output=logs/%j.out
# #SBATCH --error=logs/%j.err

set -euo pipefail

# activate your environment (e.g. conda activate <env>)

cd "$(dirname "$0")/../.."
mkdir -p logs

export HF_HOME=${HF_HOME:-~/.cache/huggingface}
# export HF_TOKEN=...   # only needed to download gated models such as meta-llama/*

BACKEND=${1:-api}                    # api | local
CONFLICT_TYPES=${CONFLICT_TYPES:-}   # space-separated list; empty = all six conflict types

CONFLICT_ARGS=""
if [ -n "${CONFLICT_TYPES}" ]; then
  CONFLICT_ARGS="--conflict-types ${CONFLICT_TYPES}"
fi

case "${BACKEND}" in
  api)
    MODEL=${2:-gpt-5}
    API_KEY_ENV=${API_KEY_ENV:-OPENAI_API_KEY}
    python -u evaluation/generate_responses_api.py \
      --model "${MODEL}" \
      --api-key-env "${API_KEY_ENV}" \
      ${CONFLICT_ARGS}
    ;;
  local)
    MODEL_PATH=${2:?"usage: $0 local <hf-model-id-or-path> <model-name>"}
    MODEL_NAME=${3:?"usage: $0 local <hf-model-id-or-path> <model-name>"}
    python -u evaluation/generate_responses_local.py \
      --model-path "${MODEL_PATH}" \
      --model-name "${MODEL_NAME}" \
      --max-new-tokens "${MAX_NEW_TOKENS:-512}" \
      ${CONFLICT_ARGS}
    ;;
  *)
    echo "unknown backend '${BACKEND}' (expected: api | local)" >&2
    exit 1
    ;;
esac

echo "Done: $(date)"
