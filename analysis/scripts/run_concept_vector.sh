#!/bin/bash
# =============================================================================
# Concept vector analysis (Section 4.1, "Conflict Awareness via Concept
# Vectors") for the three models used in the paper:
#   meta-llama/Llama-3.1-8B-Instruct   (main figures)
#   meta-llama/Llama-3.1-70B-Instruct  (appendix)
#   openai/gpt-oss-20b                 (appendix)
#
# For every model this script runs
#   1. the cross-conflict analysis over all six conflict types
#      (analysis/run_cross_conflict_analysis.py),
#   2. the subfolder analysis for inferential_conflict
#      (entailment_bank, folio, medical_qa),
#   3. the subfolder analysis for perspective_conflict
#      (allsides, perspectrum)
#      (both with analysis/run_subfolder_analysis.py).
#
# Evaluation protocol (analysis/concept_vector/cross_conflict_concept_vector.py):
#   - 5-fold stratified cross-validation for the layer-wise AUC
#   - concept vectors fitted on the training folds only, AUC on the held-out fold
#   - TF-IDF lexical baseline to check for shallow cues
#
# Usage (from anywhere; the script changes to the repository root):
#   bash analysis/scripts/run_concept_vector.sh
# For SLURM, adapt the example header below and submit from the repository
# root, or replace the `cd` line with the absolute path of your checkout.
#
# Example SLURM header (uncomment and adapt to your cluster):
# #SBATCH --job-name=concept_vec
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=128G
# #SBATCH --gpus=2
# #SBATCH --partition=gpu
# #SBATCH --time=7-00:00:00
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
OUTPUT_BASE="results/analysis"

# Models to analyze
MODEL_1="meta-llama/Llama-3.1-8B-Instruct"
MODEL_2="meta-llama/Llama-3.1-70B-Instruct"
MODEL_3="openai/gpt-oss-20b"

# Sample limits
CROSS_CONFLICT_SAMPLES=200
SUBFOLDER_SAMPLES=100

# Extra arguments passed to both runners.
# Add --skip_sea to run only the concept vector part.
EXTRA_ARGS=""

# =============================================================================
# PART 1: Cross-Conflict Analysis (All 6 Conflict Types)
# =============================================================================

echo "=============================================="
echo "PART 1: Cross-Conflict Analysis"
echo "=============================================="

# ===== Model 1: Llama-8B =====
echo ""
echo "[1/3] Llama-3.1-8B-Instruct - Cross-Conflict..."
$PYTHON analysis/run_cross_conflict_analysis.py \
    --model_path "$MODEL_1" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_BASE/llama_8b/" \
    --sample_limit $CROSS_CONFLICT_SAMPLES \
    $EXTRA_ARGS
echo "[done] Llama-8B cross-conflict completed"

# ===== Model 2: Llama-70B =====
echo ""
echo "[2/3] Llama-3.1-70B-Instruct - Cross-Conflict..."
$PYTHON analysis/run_cross_conflict_analysis.py \
    --model_path "$MODEL_2" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_BASE/llama_70b/" \
    --sample_limit $CROSS_CONFLICT_SAMPLES \
    $EXTRA_ARGS
echo "[done] Llama-70B cross-conflict completed"

# ===== Model 3: GPT-OSS-20B =====
echo ""
echo "[3/3] GPT-OSS-20B - Cross-Conflict..."
$PYTHON analysis/run_cross_conflict_analysis.py \
    --model_path "$MODEL_3" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_BASE/gpt_20b/" \
    --sample_limit $CROSS_CONFLICT_SAMPLES \
    $EXTRA_ARGS
echo "[done] GPT-OSS-20B cross-conflict completed"

# =============================================================================
# PART 2: Subfolder Analysis - Inferential Conflict
# (entailment_bank, folio, medical_qa)
# =============================================================================

echo ""
echo "=============================================="
echo "PART 2: Subfolder Analysis - Inferential Conflict"
echo "  Subfolders: entailment_bank, folio, medical_qa"
echo "=============================================="

# ===== Model 1: Llama-8B =====
echo ""
echo "[1/3] Llama-8B - Inferential Subfolder..."
$PYTHON analysis/run_subfolder_analysis.py \
    --model_path "$MODEL_1" \
    --data_root "$DATA_ROOT" \
    --conflict_type "inferential_conflict" \
    --output_dir "$OUTPUT_BASE/llama_8b/subfolder/" \
    --sample_limit $SUBFOLDER_SAMPLES \
    $EXTRA_ARGS
echo "[done] Llama-8B inferential subfolder completed"

# ===== Model 2: Llama-70B =====
echo ""
echo "[2/3] Llama-70B - Inferential Subfolder..."
$PYTHON analysis/run_subfolder_analysis.py \
    --model_path "$MODEL_2" \
    --data_root "$DATA_ROOT" \
    --conflict_type "inferential_conflict" \
    --output_dir "$OUTPUT_BASE/llama_70b/subfolder/" \
    --sample_limit $SUBFOLDER_SAMPLES \
    $EXTRA_ARGS
echo "[done] Llama-70B inferential subfolder completed"

# ===== Model 3: GPT-OSS-20B =====
echo ""
echo "[3/3] GPT-OSS-20B - Inferential Subfolder..."
$PYTHON analysis/run_subfolder_analysis.py \
    --model_path "$MODEL_3" \
    --data_root "$DATA_ROOT" \
    --conflict_type "inferential_conflict" \
    --output_dir "$OUTPUT_BASE/gpt_20b/subfolder/" \
    --sample_limit $SUBFOLDER_SAMPLES \
    $EXTRA_ARGS
echo "[done] GPT-OSS-20B inferential subfolder completed"

# =============================================================================
# PART 3: Subfolder Analysis - Perspective Conflict
# (allsides, perspectrum)
# =============================================================================

echo ""
echo "=============================================="
echo "PART 3: Subfolder Analysis - Perspective Conflict"
echo "  Subfolders: allsides, perspectrum"
echo "=============================================="

# ===== Model 1: Llama-8B =====
echo ""
echo "[1/3] Llama-8B - Perspective Subfolder..."
$PYTHON analysis/run_subfolder_analysis.py \
    --model_path "$MODEL_1" \
    --data_root "$DATA_ROOT" \
    --conflict_type "perspective_conflict" \
    --output_dir "$OUTPUT_BASE/llama_8b/subfolder/" \
    --sample_limit $SUBFOLDER_SAMPLES \
    $EXTRA_ARGS
echo "[done] Llama-8B perspective subfolder completed"

# ===== Model 2: Llama-70B =====
echo ""
echo "[2/3] Llama-70B - Perspective Subfolder..."
$PYTHON analysis/run_subfolder_analysis.py \
    --model_path "$MODEL_2" \
    --data_root "$DATA_ROOT" \
    --conflict_type "perspective_conflict" \
    --output_dir "$OUTPUT_BASE/llama_70b/subfolder/" \
    --sample_limit $SUBFOLDER_SAMPLES \
    $EXTRA_ARGS
echo "[done] Llama-70B perspective subfolder completed"

# ===== Model 3: GPT-OSS-20B =====
echo ""
echo "[3/3] GPT-OSS-20B - Perspective Subfolder..."
$PYTHON analysis/run_subfolder_analysis.py \
    --model_path "$MODEL_3" \
    --data_root "$DATA_ROOT" \
    --conflict_type "perspective_conflict" \
    --output_dir "$OUTPUT_BASE/gpt_20b/subfolder/" \
    --sample_limit $SUBFOLDER_SAMPLES \
    $EXTRA_ARGS
echo "[done] GPT-OSS-20B perspective subfolder completed"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "ALL ANALYSES COMPLETED"
echo "=============================================="
echo ""
echo "Results saved to: $OUTPUT_BASE/"
echo ""
echo "Structure:"
echo "  llama_8b/"
echo "    concept_vector/                  # Cross-conflict (all 6 types)"
echo "    sea/                             # Cross-conflict spectral energy"
echo "    subfolder/"
echo "      inferential_conflict/          # entailment_bank, folio, medical_qa"
echo "      perspective_conflict/          # allsides, perspectrum"
echo ""
echo "  llama_70b/                         # Same structure"
echo "  gpt_20b/                           # Same structure"
echo ""
echo "Key output files per concept_vector/ directory:"
echo "  - auc_comparison.png"
echo "  - best_layers_comparison.png"
echo "  - vector_similarity_heatmap.png"
echo "  - auc_scores.json"
echo "  - auc_details_cv.json (mean and std over CV folds)"
echo "  - lexical_baseline.json"
echo ""
echo "Note: all AUC values use 5-fold CV to avoid optimistic estimates."
