#!/bin/bash
# =============================================================================
# Directional evidence position-bias measurement (Section 4.3, "Evidence
# Position Bias in Internal Representations"; appendix "Bias Measurement
# Implementation Notes") for the two models shown in the paper:
#   meta-llama/Llama-3.1-8B-Instruct  with the simple system prompt
#       -> bias_simple_prompt_all_conflicts.png in the paper
#   openai/gpt-oss-20b                with the detailed system prompt
#       -> bias_stacked_area_all_conflicts_gpt20b.png in the paper
#
# For every model, analysis/position_bias/run_directional_bias.py
#   1. collects final-token activations of all layers for one combined-evidence
#      prompt and K single-evidence prompts per sample (ambiguity, granularity
#      and perspective conflicts; 20% split, seed 42),
#   2. computes b_i = cos(c - mu, a_i - mu) per sample and layer and the
#      per-layer fraction of samples favoring each evidence,
#   3. writes activations (.pkl), metrics (.json) and the stacked-area figures.
#
# Usage (from anywhere; the script changes to the repository root):
#   bash analysis/scripts/run_directional_bias.sh
# For SLURM, adapt the example header below and submit from the repository
# root, or replace the `cd` line with the absolute path of your checkout.
#
# Example SLURM header (uncomment and adapt to your cluster):
# #SBATCH --job-name=position_bias
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=64G
# #SBATCH --gpus=1
# #SBATCH --partition=gpu
# #SBATCH --time=2-00:00:00
# #SBATCH --output=logs/%j.out
# #SBATCH --error=logs/%j.err
# =============================================================================

set -e

# activate your environment
export HF_HOME=${HF_HOME:-~/.cache/huggingface}
# Gated models (Llama-3.1) need a HuggingFace token: export HF_TOKEN=<your token>
# in your shell before running. Do not write the token into this file.

# Move to the repository root
cd "$(dirname "$0")/../.."

mkdir -p logs

PYTHON=${PYTHON:-python}

# Data root and output base directory (relative to the repository root)
DATA_ROOT="ContextConflict_Dataset/data"
OUTPUT_BASE="results/analysis/position_bias"

# Conflict types used for the measurement (summarization tasks)
CONFLICT_TYPES="ambiguity_conflict granularity_conflict perspective_conflict"

# Sample selection (paper setting)
TRAIN_RATIO=0.2
MAX_SAMPLES=10000
SEED=42

# Extra arguments passed to the runner (e.g. --layers 0 1 2 ... or --device cuda)
EXTRA_ARGS=""

# =============================================================================
# [1/2] Llama-3.1-8B-Instruct, simple prompt
# =============================================================================
echo "=============================================="
echo "[1/2] Llama-3.1-8B-Instruct - simple prompt"
echo "=============================================="
$PYTHON -u analysis/position_bias/run_directional_bias.py \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --data_dir "$DATA_ROOT" \
    --conflict_types $CONFLICT_TYPES \
    --output_dir "$OUTPUT_BASE/llama_8b" \
    --prompt_type simple \
    --train_ratio $TRAIN_RATIO \
    --max_samples $MAX_SAMPLES \
    --seed $SEED \
    --activation_mode last_token \
    $EXTRA_ARGS
echo "[done] Llama-3.1-8B-Instruct (simple prompt)"

# =============================================================================
# [2/2] GPT-OSS-20B, detailed prompt
# =============================================================================
echo ""
echo "=============================================="
echo "[2/2] GPT-OSS-20B - detailed prompt"
echo "=============================================="
$PYTHON -u analysis/position_bias/run_directional_bias.py \
    --model_name "openai/gpt-oss-20b" \
    --data_dir "$DATA_ROOT" \
    --conflict_types $CONFLICT_TYPES \
    --output_dir "$OUTPUT_BASE/gpt_20b" \
    --prompt_type detailed \
    --train_ratio $TRAIN_RATIO \
    --max_samples $MAX_SAMPLES \
    --seed $SEED \
    --activation_mode last_token \
    $EXTRA_ARGS
echo "[done] GPT-OSS-20B (detailed prompt)"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "POSITION-BIAS MEASUREMENT COMPLETED"
echo "=============================================="
echo ""
echo "Results saved to: $OUTPUT_BASE/"
echo ""
echo "  llama_8b/"
echo "    evidence_activations_simple.pkl"
echo "    collection_summary_simple.json"
echo "    bias_metrics_per_layer_simple.json"
echo "    directional_bias_ratios_simple.json"
echo "    bias_stacked_area_all_conflicts_simple.png   # paper: bias_simple_prompt_all_conflicts.png"
echo "    bias_stacked_area_averaged_simple.png"
echo ""
echo "  gpt_20b/                                        # same files with suffix _detailed"
echo "    bias_stacked_area_all_conflicts_detailed.png # paper: bias_stacked_area_all_conflicts_gpt20b.png"
