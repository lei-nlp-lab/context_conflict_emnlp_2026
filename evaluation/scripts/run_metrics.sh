#!/bin/bash
# Accuracy (reasoning conflict types) and Balance score (summarization conflict
# types), paper Section 3.1.
#
# accuracy.py reads the response JSON files produced by the generation scripts and
# scores inferential, misinformation and temporal conflicts. balance_score.py reads
# the Shapley contribution shares that evaluation/scripts/run_batch_evaluate.sh wrote
# under the "eval" key and scores ambiguity, granularity and perspective conflicts.
#
# Usage (from anywhere; the script cd's to the repository root):
#   bash evaluation/scripts/run_metrics.sh
#   MODELS="gpt-5 llama-3.1-8b-instruct" RESPONSE_KEY=response bash evaluation/scripts/run_metrics.sh
#   EVAL_KEY=eval_answer_only bash evaluation/scripts/run_metrics.sh   # Balance of the <answer>-only attribution
#
# Outputs:
#   results/evaluation/accuracy/accuracy.json (+ wrong_samples.csv)
#   results/evaluation/balance_scores.json (balance_scores_<EVAL_KEY>.json when EVAL_KEY != eval)

set -euo pipefail

# activate your environment (e.g. conda activate <env>)

cd "$(dirname "$0")/../.."

RESULT_DIR=${RESULT_DIR:-results/responses}
RESPONSE_KEY=${RESPONSE_KEY:-"{model}_response"}   # key written by generate_responses_*.py; use "response" for the paper result files
EVAL_KEY=${EVAL_KEY:-eval}               # key written by batch_evaluate.py --output-key
MODELS=${MODELS:-}                       # space-separated list; empty = the seven models of the paper

MODEL_ARGS=""
if [ -n "${MODELS}" ]; then
  MODEL_ARGS="--models ${MODELS}"
fi

# 1) Accuracy: inferential, misinformation and temporal conflicts
python evaluation/accuracy.py \
  --result-dir "${RESULT_DIR}" \
  --response-key "${RESPONSE_KEY}" \
  --output-dir results/evaluation/accuracy \
  --csv \
  ${MODEL_ARGS}

# 2) Balance score: ambiguity, granularity and perspective conflicts
BALANCE_OUT=results/evaluation/balance_scores.json
if [ "${EVAL_KEY}" != "eval" ]; then
  BALANCE_OUT="results/evaluation/balance_scores_${EVAL_KEY}.json"
fi
python evaluation/balance_score.py \
  --result-dir "${RESULT_DIR}" \
  --eval-key "${EVAL_KEY}" \
  --output "${BALANCE_OUT}" \
  ${MODEL_ARGS}

echo "Done: $(date)"
