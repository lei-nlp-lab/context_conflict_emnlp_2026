#!/bin/bash
# Subfolder-Level Spectral Energy Analysis (SEA), Section 4.2 of
# "Large Language Models in Resolving Contextual Knowledge Conflicts".
# Runs the energy-ratio / Delta-ER analysis separately for each source
# subfolder of the two conflict types that have several sources
# (per-subfolder energy comparison figure):
#   perspective_conflict : allsides, perspectrum
#   inferential_conflict : entailment_bank, folio, medical_qa
#
# Example SLURM header (uncomment and adapt to your cluster):
# #SBATCH --job-name=sea_subfolder
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
#   sbatch --chdir="$PWD" analysis/scripts/run_sea_subfolder.sh
cd "$(dirname "$0")/../.."

mkdir -p logs

# Model to analyze (override with MODEL_PATH=... ./analysis/scripts/run_sea_subfolder.sh).
MODEL_PATH="${MODEL_PATH:-openai/gpt-oss-20b}"
OUTPUT_DIR="${OUTPUT_DIR:-results/analysis/sea}"

echo "===== Analyzing perspective_conflict ====="
python -u analysis/spectral_energy/run_sea_subfolder.py \
    --model_path "${MODEL_PATH}" \
    --data_root ContextConflict_Dataset/data \
    --conflict_type perspective_conflict \
    --sample_limit 200 \
    --top_k 10 \
    --compute_ci \
    --n_bootstrap 500 \
    --output_dir "${OUTPUT_DIR}" \
    --seed 42

echo "===== Analyzing inferential_conflict ====="
python -u analysis/spectral_energy/run_sea_subfolder.py \
    --model_path "${MODEL_PATH}" \
    --data_root ContextConflict_Dataset/data \
    --conflict_type inferential_conflict \
    --sample_limit 200 \
    --top_k 10 \
    --compute_ci \
    --n_bootstrap 500 \
    --output_dir "${OUTPUT_DIR}" \
    --seed 42

echo "SEA Subfolder Analysis Complete for both conflict types!"
