#!/bin/bash
# RAGAS faithfulness (paper Section 3.1; reported for all six conflict types).
#
# Scores every response JSON under results/responses/<model>/<conflict_type>/ with
# the RAGAS faithfulness metric (judge LLM: deepseek-chat by default), writes the
# per-sample score into each JSON under "faithfulness" and the per-(model, conflict
# type) mean/std to results/evaluation/faithfulness.tsv. Files that already hold a
# valid score are skipped, so interrupted runs resume.
#
# Usage (from anywhere; the script cd's to the repository root):
#   export DEEPSEEK_API_KEY=...
#   bash evaluation/scripts/run_faithfulness.sh                # sequential, the 7 paper models
#   MODELS="gpt-5 llama-3.1-8b-instruct" bash evaluation/scripts/run_faithfulness.sh
#   RESPONSE_KEY=response bash evaluation/scripts/run_faithfulness.sh   # result files whose output is stored under "response"
#   bash evaluation/scripts/run_faithfulness.sh list           # print the task table
#   TASK_ID=3 bash evaluation/scripts/run_faithfulness.sh single   # one (model, conflict type) pair
#   # another OpenAI-compatible judge
#   JUDGE_MODEL=gpt-4o BASE_URL=https://api.openai.com/v1 API_KEY_ENV=OPENAI_API_KEY \
#     bash evaluation/scripts/run_faithfulness.sh
#
# Job-array example, one task per (model, conflict type) pair. N is the number of
# tasks printed by "list" minus one, K the number of concurrent jobs (30 was used
# for the paper to respect the judge API rate limit). Submit with
#   sbatch evaluation/scripts/run_faithfulness.sh single
# from the repository root (TASK_ID defaults to SLURM_ARRAY_TASK_ID). Use the same
# RESULT_DIR / MODELS / CONFLICT_TYPES for "list" and "single" so task ids agree.
# #SBATCH --job-name=faithfulness
# #SBATCH --array=0-N%K
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=1
# #SBATCH --mem=4gb
# #SBATCH --time=1-00:00:00
# #SBATCH --output=logs/%A_%a.out
# #SBATCH --error=logs/%A_%a.err

set -euo pipefail

# activate your environment (e.g. conda activate <env>)

cd "$(dirname "$0")/../.."
mkdir -p logs

MODE=${1:-sequential}                     # sequential | single | list
RESULT_DIR=${RESULT_DIR:-results/responses}
RESPONSE_KEY=${RESPONSE_KEY:-"{model}_response"}   # key written by generate_responses_*.py; use "response" for the paper result files
OUTPUT=${OUTPUT:-results/evaluation/faithfulness.tsv}
MODELS=${MODELS:-}                        # space-separated list; empty = the seven models of the paper
CONFLICT_TYPES=${CONFLICT_TYPES:-}        # space-separated list; empty = all six conflict types
JUDGE_MODEL=${JUDGE_MODEL:-deepseek-chat}
API_KEY_ENV=${API_KEY_ENV:-DEEPSEEK_API_KEY}
BASE_URL=${BASE_URL:-}                    # empty = $DEEPSEEK_BASE_URL or https://api.deepseek.com

COMMON_ARGS="--result-dir ${RESULT_DIR} --response-key ${RESPONSE_KEY} --output ${OUTPUT}"
COMMON_ARGS="${COMMON_ARGS} --judge-model ${JUDGE_MODEL} --api-key-env ${API_KEY_ENV}"
if [ -n "${MODELS}" ]; then
  COMMON_ARGS="${COMMON_ARGS} --models ${MODELS}"
fi
if [ -n "${CONFLICT_TYPES}" ]; then
  COMMON_ARGS="${COMMON_ARGS} --conflict-types ${CONFLICT_TYPES}"
fi
if [ -n "${BASE_URL}" ]; then
  COMMON_ARGS="${COMMON_ARGS} --base-url ${BASE_URL}"
fi

case "${MODE}" in
  sequential)
    python -u evaluation/faithfulness_ragas.py ${COMMON_ARGS} --mode sequential
    ;;
  list)
    python evaluation/faithfulness_ragas.py ${COMMON_ARGS} --mode list
    ;;
  single)
    TASK_ID=${TASK_ID:-${SLURM_ARRAY_TASK_ID:?"set TASK_ID (or run as a SLURM job array)"}}
    python -u evaluation/faithfulness_ragas.py ${COMMON_ARGS} --mode single --task-id "${TASK_ID}"
    ;;
  *)
    echo "unknown mode '${MODE}' (expected: sequential | single | list)" >&2
    exit 1
    ;;
esac

echo "Done: $(date)"
