#!/usr/bin/env python3
"""
Cross-Conflict SEA (Spectral Energy Analysis) Runner

Reproduces the layer-wise Delta-ER curves of Section 4.2 of
"Large Language Models in Resolving Contextual Knowledge Conflicts"
(EMNLP 2026): Figure delta_er (Llama-3.1-8B) and Figure delta_er_comparison
(GPT-OSS-20B). For each of the six conflict types it loads paired
consistent / conflict prompts with ConflictAwareDataLoader, extracts the
last non-padding-token hidden state of every layer in a single forward pass,
computes the top-k energy ratio of the centered activation matrices with
torch.svd_lowrank, and reports Delta-ER = ER_conflict - ER_consistent with
bootstrap confidence intervals (see cross_conflict_sea.py).

Outputs (under --output_dir/<model_name>/cross_conflict/):
    <conflict_type>/spectrum_layer<L>.npz   per-layer ER, Delta-ER, singular values, CIs
    energy_summary.json                     max / mean Delta-ER and best layer per type
    plots/delta_er_comparison.png           Delta-ER and ER curves by layer
    plots/energy_comparison.png             conflict ER curves by layer
    plots/summary_bar_chart.png             max Delta-ER per conflict type

Example (paper setting; run from the repository root):
    python analysis/spectral_energy/run_sea_cross_conflict.py \
        --model_path meta-llama/Llama-3.1-8B-Instruct \
        --sample_limit 400 --top_k 10 --compute_ci --n_bootstrap 500 --seed 42
"""

import argparse
import logging
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM

from analysis.conflict_aware_data_loader import ConflictAwareDataLoader
from analysis.spectral_energy.cross_conflict_sea import CrossConflictSEAAnalyzer

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cross-Conflict SEA Analysis (Spectral Energy Only)"
    )

    # Model parameters
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Hugging Face model id or local path (gated models read HF_TOKEN from the environment)"
    )

    # Data parameters
    parser.add_argument(
        "--data_root",
        type=str,
        default="ContextConflict_Dataset/data",
        help="Root directory of the ContextConflict dataset (default: ContextConflict_Dataset/data)"
    )
    parser.add_argument(
        "--conflict_types",
        type=str,
        nargs="+",
        default=["inferential_conflict", "misinformation_conflict", "temporal_conflict",
                 "ambiguity_conflict", "granularity_conflict", "perspective_conflict"],
        help="Conflict types to analyze"
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=100,
        help="Number of samples per conflict type (paper: 400)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    # Analysis parameters
    parser.add_argument(
        "--target_layers",
        type=int,
        nargs="+",
        default=None,
        help="Layers to analyze (default: all layers)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-k components for energy ratio"
    )
    parser.add_argument(
        "--compute_ci",
        action="store_true",
        default=True,
        help="Compute bootstrap confidence intervals (enabled by default; use --no_ci to disable)"
    )
    parser.add_argument(
        "--no_ci",
        action="store_true",
        help="Disable bootstrap confidence intervals (faster)"
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=100,
        help="Number of bootstrap resamples for the CI (paper: 500)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Reserved; activation extraction currently uses a fixed batch size of 4 inside the analyzer"
    )

    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/analysis/sea",
        help="Output directory; results go to <output_dir>/<model_name>/cross_conflict/"
    )

    return parser.parse_args()


def load_model(model_path: str):
    """Load model and tokenizer."""
    logger.info(f"Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # Get device
    device = next(model.parameters()).device
    logger.info(f"Model loaded on device: {device}")

    return model, tokenizer, device


def load_conflict_data(
    data_root: str,
    conflict_types: list,
    sample_limit: int,
    seed: int
) -> dict:
    """
    Load conflict data for all specified conflict types.

    Returns:
        {conflict_type: (consistent_texts, conflict_texts)}
    """
    logger.info(f"Loading data from {data_root}")

    data_loader = ConflictAwareDataLoader(data_root)
    conflict_samples_dict = {}

    for conflict_type in conflict_types:
        logger.info(f"Loading {conflict_type}...")

        try:
            consistent_texts, conflict_texts = data_loader.load_conflict_type_samples(
                conflict_type,
                limit=sample_limit,
                seed=seed
            )

            if consistent_texts and conflict_texts:
                conflict_samples_dict[conflict_type] = (consistent_texts, conflict_texts)
                logger.info(f"  Loaded {len(consistent_texts)} consistent, {len(conflict_texts)} conflict samples")
            else:
                logger.warning(f"  No samples found for {conflict_type}")

        except Exception as e:
            logger.error(f"  Failed to load {conflict_type}: {e}")

    return conflict_samples_dict


def print_summary(analyzer: CrossConflictSEAAnalyzer):
    """Print analysis summary."""
    summary = analyzer.get_energy_strength_summary()

    print("\n" + "=" * 80)
    print("SEA ANALYSIS SUMMARY")
    print("=" * 80)

    print(f"\n{'Conflict Type':<25} {'Max ΔER':>10} {'Mean ΔER':>10} {'Best Layer':>12} {'Conflict ER':>12} {'Consistent ER':>14}")
    print("-" * 85)

    for conflict_type, stats in sorted(summary.items()):
        print(f"{conflict_type:<25} {stats['max_delta_er']:>10.4f} {stats['mean_delta_er']:>10.4f} "
              f"{stats['highest_layer']:>12} {stats['max_conflict_er']:>12.4f} {stats['max_consistent_er']:>14.4f}")

    print("=" * 80)

    # Rank by max ΔER
    print("\nConflict Types Ranked by Max ΔER (Conflict Signal Strength):")
    ranked = sorted(summary.items(), key=lambda x: x[1]['max_delta_er'], reverse=True)
    for i, (ct, stats) in enumerate(ranked, 1):
        print(f"  {i}. {ct}: ΔER={stats['max_delta_er']:.4f} at Layer {stats['highest_layer']}")


def main():
    args = parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, tokenizer, device = load_model(args.model_path)

    # Get model name for output
    model_name = args.model_path.split("/")[-1]

    # Load data
    conflict_samples_dict = load_conflict_data(
        args.data_root,
        args.conflict_types,
        args.sample_limit,
        args.seed
    )

    if not conflict_samples_dict:
        logger.error("No conflict data loaded. Exiting.")
        return

    # Determine target layers
    target_layers = args.target_layers
    if target_layers is None:
        num_layers = len(model.model.layers)
        target_layers = list(range(num_layers))
        logger.info(f"Analyzing all {num_layers} layers")

    # Initialize analyzer
    logger.info("Initializing SEA Analyzer...")
    analyzer = CrossConflictSEAAnalyzer(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        device=device
    )

    # Determine CI computation
    compute_ci = args.compute_ci and not args.no_ci

    # Run analysis
    logger.info("Starting SEA analysis...")
    analyzer.analyze_all_conflict_types(
        conflict_samples_dict=conflict_samples_dict,
        target_layers=target_layers,
        top_k=args.top_k,
        compute_ci=compute_ci,
        n_bootstrap=args.n_bootstrap
    )

    # Print summary
    print_summary(analyzer)

    # Save results
    results_dir = os.path.join(args.output_dir, model_name, "cross_conflict")
    analyzer.save_results(results_dir)

    # Generate plots
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    logger.info("Generating plots...")

    # Main comparison plot (ΔER)
    analyzer.plot_delta_er_comparison(
        save_path=os.path.join(plots_dir, "delta_er_comparison.png"),
        show_ci=compute_ci
    )

    # Legacy energy comparison
    analyzer.plot_energy_comparison(
        save_path=os.path.join(plots_dir, "energy_comparison.png")
    )

    # Summary bar chart
    analyzer.plot_summary_bar_chart(
        save_path=os.path.join(plots_dir, "summary_bar_chart.png")
    )

    logger.info(f"\nResults saved to {results_dir}")
    logger.info("SEA Cross-Conflict Analysis Complete!")


if __name__ == "__main__":
    main()

