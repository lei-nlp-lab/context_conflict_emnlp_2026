#!/bin/bash
# Shapley-based evidence attribution (Evidence Balance, paper Section 3.1).
#
# Runs evaluation/batch_evaluate.py over one model's response directory with the
# frozen meta-llama/Llama-3.2-1B-Instruct scorer. Each response JSON gets an
# in-place "eval" key with "shapley_values" and "prob_distribution"; compute the
# Balance score afterwards with evaluation/balance_score.py.
#
# Usage:
#   bash evaluation/scripts/run_batch_evaluate.sh                       # MODEL=gpt-5
#   MODEL=llama-3.1-8b-instruct bash evaluation/scripts/run_batch_evaluate.sh
#   MODEL=results/responses/my_model RESPONSE_KEY=response bash evaluation/scripts/run_batch_evaluate.sh
#
# Example SLURM header (uncomment and adapt if you submit this file with sbatch;
# when using sbatch, submit from the repository root or add --chdir=<repo root>):
# #SBATCH --job-name=batch-eval
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=32gb
# #SBATCH --gpus=1
# #SBATCH --time=1-00:00:00
# #SBATCH --output=logs/%j.out
# #SBATCH --error=logs/%j.err

set -euo pipefail

# activate your environment (e.g. conda activate <env>)

cd "$(dirname "$0")/../.."
mkdir -p logs

export HF_HOME=${HF_HOME:-~/.cache/huggingface}
# export HF_TOKEN=...   # only needed to download gated models such as meta-llama/*

# Sub-directory of results/responses (or a path to a directory of response JSON files).
MODEL=${MODEL:-gpt-5}
# Key holding the model output in each JSON. generate_responses_api.py and
# generate_responses_local.py write "<model>_response"; "{model}" is expanded to $MODEL.
RESPONSE_KEY=${RESPONSE_KEY:-"{model}_response"}
# Frozen scorer for the value function (robustness runs: meta-llama/Llama-3.1-8B-Instruct, google/gemma-2b).
EVAL_MODEL=${EVAL_MODEL:-meta-llama/Llama-3.2-1B-Instruct}

echo "=================================================="
echo "Batch evaluation: model=${MODEL} scorer=${EVAL_MODEL}"
echo "Time: $(date)"
echo "=================================================="

# 1) Full response: Shapley values over the complete model output -> key "eval"
python -u evaluation/batch_evaluate.py \
  --model "${MODEL}" \
  --eval-model "${EVAL_MODEL}" \
  --eval-methods perplexity \
  --response-key "${RESPONSE_KEY}" \
  --output-key eval

# 2) Answer-only variant: score only the text inside <answer></answer> -> key "eval_answer_only"
python -u evaluation/batch_evaluate.py \
  --model "${MODEL}" \
  --eval-model "${EVAL_MODEL}" \
  --eval-methods perplexity \
  --response-key "${RESPONSE_KEY}" \
  --answer-only \
  --output-key eval_answer_only

echo "Done: $(date)"
