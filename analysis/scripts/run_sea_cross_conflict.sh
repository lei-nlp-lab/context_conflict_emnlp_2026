#!/bin/bash
# Cross-Conflict Spectral Energy Analysis (SEA), Section 4.2 of
# "Large Language Models in Resolving Contextual Knowledge Conflicts".
# Computes per-layer top-k energy ratios of conflict vs. consistent prompts
# for all six conflict types and plots the Delta-ER curves
# (Figures delta_er for Llama-3.1-8B and delta_er_comparison for GPT-OSS-20B).
#
# Example SLURM header (uncomment and adapt to your cluster):
# #SBATCH --job-name=sea_cross
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=128G
# #SBATCH --gpus=2
# #SBATCH --partition=gpu
# #SBATCH --time=1-00:00:00
# #SBATCH --output=logs/%j.out
# #SBATCH --error=logs/%j.err

# activate your environment
export HF_HOME=${HF_HOME:-~/.cache/huggingface}
# Gated models (e.g. meta-llama/*) additionally need: export HF_TOKEN=<your token>

# Move to the repository root (this script lives in analysis/scripts/).
# Under sbatch, $0 is a spooled copy of the script, so submit it from the
# repository root and set the working directory explicitly, e.g.
#   sbatch --chdir="$PWD" analysis/scripts/run_sea_cross_conflict.sh
cd "$(dirname "$0")/../.."

mkdir -p logs

# Model to analyze (override with MODEL_PATH=... ./analysis/scripts/run_sea_cross_conflict.sh).
# The paper reports meta-llama/Llama-3.1-8B-Instruct and openai/gpt-oss-20b.
MODEL_PATH="${MODEL_PATH:-openai/gpt-oss-20b}"
OUTPUT_DIR="${OUTPUT_DIR:-results/analysis/sea}"

# Run cross-conflict SEA analysis with the paper hyper-parameters
python -u analysis/spectral_energy/run_sea_cross_conflict.py \
    --model_path "${MODEL_PATH}" \
    --data_root ContextConflict_Dataset/data \
    --conflict_types inferential_conflict misinformation_conflict temporal_conflict \
                     ambiguity_conflict granularity_conflict perspective_conflict \
    --sample_limit 400 \
    --top_k 10 \
    --compute_ci \
    --n_bootstrap 500 \
    --output_dir "${OUTPUT_DIR}" \
    --seed 42

echo "SEA Cross-Conflict Analysis Complete!"
