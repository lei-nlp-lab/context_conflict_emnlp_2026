#!/bin/bash
# Position shuffling experiment (appendix "Position Shuffling Experiment" of
# "Large Language Models in Resolving Contextual Knowledge Conflicts").
# Samples 33 items per summarization conflict type (ambiguity, granularity,
# perspective), permutes the evidence order, generates responses with
# Llama-3.1-8B-Instruct and GPT-OSS-20B and scores them with the
# Llama-3.2-1B perplexity scorer (full response and <answer> span only).
# Afterwards run
#   python analysis/position_bias/shuffle_pie_charts.py --shift-dir <OUTPUT_DIR>
# to print the mean share per evidence position (paper tables) and draw the pies.
#
# Example SLURM header (uncomment and adapt to your cluster):
# #SBATCH --job-name=shuffle
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=64G
# #SBATCH --gpus=2
# #SBATCH --partition=gpu
# #SBATCH --time=1-00:00:00
# #SBATCH --output=logs/%j.out
# #SBATCH --error=logs/%j.err

# activate your environment
export HF_HOME=${HF_HOME:-~/.cache/huggingface}
# Gated models (meta-llama/*) additionally need: export HF_TOKEN=<your token>
mkdir -p "${HF_HOME}"

# Move to the repository root (this script lives in analysis/scripts/).
# Under sbatch, $0 is a spooled copy of the script, so submit it from the
# repository root and set the working directory explicitly, e.g.
#   sbatch --chdir="$PWD" analysis/scripts/run_shuffle_experiment.sh
cd "$(dirname "$0")/../.."

mkdir -p logs

# Override with DATA_ROOT=... OUTPUT_DIR=... ./analysis/scripts/run_shuffle_experiment.sh
DATA_ROOT="${DATA_ROOT:-ContextConflict_Dataset/data}"
OUTPUT_DIR="${OUTPUT_DIR:-results/analysis/position_bias/shift}"

echo "=================================================="
echo "Generating shuffled responses + evaluating"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Time: $(date)"
echo "=================================================="
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi
echo "=================================================="

python -u analysis/position_bias/shuffle_experiment.py \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed 42 \
    --max-new-tokens 512 \
    --eval-model meta-llama/Llama-3.2-1B-Instruct

echo "=================================================="
echo "All done! $(date)"
echo "=================================================="
