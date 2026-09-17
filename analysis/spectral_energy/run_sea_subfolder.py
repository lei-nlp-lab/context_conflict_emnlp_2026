#!/usr/bin/env python3
"""
Subfolder-Level SEA (Spectral Energy Analysis) Runner

Companion to run_sea_cross_conflict.py for Section 4.2 of
"Large Language Models in Resolving Contextual Knowledge Conflicts"
(EMNLP 2026). Two conflict types of ContextConflict are built from several
source datasets that are stored as subfolders:

    inferential_conflict: entailment_bank, folio, medical_qa
    perspective_conflict: allsides, perspectrum

This script runs the same energy-ratio / Delta-ER analysis separately for
each subfolder, which reproduces the per-subfolder energy comparison figure
and lets source-specific spectral behaviour be compared within one conflict
category.

Outputs (under --output_dir/<model_name>/subfolder/<conflict_type>/):
    <subfolder>/spectrum_layer<L>.npz       per-layer ER, Delta-ER, singular values, CIs
    energy_summary.json                     max / mean Delta-ER and best layer per subfolder
    plots/<conflict_type>_subfolder_delta_er.png, ..._subfolder_energy.png,
    plots/<conflict_type>_subfolder_summary.png, ..._delta_er_heatmap.png

Example (paper setting; run from the repository root):
    python analysis/spectral_energy/run_sea_subfolder.py \
        --model_path openai/gpt-oss-20b --conflict_type perspective_conflict \
        --sample_limit 200 --top_k 10 --compute_ci --n_bootstrap 500 --seed 42
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


# Define subfolders for conflict types that have them
# Only perspective_conflict and inferential_conflict have meaningful subfolders
CONFLICT_SUBFOLDERS = {
    "inferential_conflict": ["entailment_bank", "folio", "medical_qa"],
    "perspective_conflict": ["allsides", "perspectrum"],
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Subfolder-Level SEA Analysis (Spectral Energy Only)"
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
        "--conflict_type",
        type=str,
        required=True,
        choices=list(CONFLICT_SUBFOLDERS.keys()),
        help="Conflict type to analyze in detail"
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=50,
        help="Number of samples per subfolder (paper: 200)"
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

    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/analysis/sea",
        help="Output directory; results go to <output_dir>/<model_name>/subfolder/<conflict_type>/"
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

    device = next(model.parameters()).device
    logger.info(f"Model loaded on device: {device}")

    return model, tokenizer, device


def load_subfolder_data(
    data_root: str,
    conflict_type: str,
    sample_limit: int,
    seed: int
) -> dict:
    """
    Load conflict data for each subfolder within a conflict type.

    Returns:
        {subfolder_name: (consistent_texts, conflict_texts)}
    """
    logger.info(f"Loading subfolder data for {conflict_type}")

    subfolders = CONFLICT_SUBFOLDERS.get(conflict_type, [])
    if not subfolders:
        logger.error(f"No subfolders defined for {conflict_type}")
        return {}

    data_loader = ConflictAwareDataLoader(data_root)
    subfolder_data = {}

    # Map subfolders to their processing methods
    process_methods = {
        # inferential_conflict
        "entailment_bank": data_loader.process_inferential_conflict_entailment_bank,
        "folio": data_loader.process_inferential_conflict_folio,
        "medical_qa": data_loader.process_inferential_conflict_medical_qa,
        # perspective_conflict
        "allsides": data_loader.process_perspective_conflict_allsides,
        "perspectrum": data_loader.process_perspective_conflict_perspectrum,
    }

    for subfolder in subfolders:
        logger.info(f"Loading subfolder: {subfolder}")

        try:
            # Load JSON files for this subfolder
            data_list = data_loader.load_json_files(
                conflict_type,
                subfolder=subfolder,
                limit=sample_limit,
                seed=seed
            )

            if not data_list:
                logger.warning(f"  No data files found for {subfolder}")
                continue

            # Get appropriate processing method
            process_method = process_methods.get(subfolder)
            if process_method is None:
                logger.warning(f"  No processing method for {subfolder}, skipping")
                continue

            # Process data to get consistent/conflict pairs
            consistent_texts, conflict_texts = process_method(data_list)

            if consistent_texts and conflict_texts:
                subfolder_data[subfolder] = (consistent_texts, conflict_texts)
                logger.info(f"  Loaded {len(consistent_texts)} consistent, {len(conflict_texts)} conflict samples")
            else:
                logger.warning(f"  No valid samples generated for {subfolder}")

        except Exception as e:
            logger.error(f"  Failed to load {subfolder}: {e}")
            import traceback
            traceback.print_exc()

    return subfolder_data


def print_subfolder_summary(analyzer: CrossConflictSEAAnalyzer, conflict_type: str):
    """Print subfolder analysis summary."""
    summary = analyzer.get_energy_strength_summary()

    print("\n" + "=" * 80)
    print(f"SEA SUBFOLDER ANALYSIS: {conflict_type.upper()}")
    print("=" * 80)

    print(f"\n{'Subfolder':<20} {'Max ΔER':>10} {'Mean ΔER':>10} {'Best Layer':>12} {'Conflict ER':>12} {'Consistent ER':>14}")
    print("-" * 80)

    for subfolder, stats in sorted(summary.items()):
        print(f"{subfolder:<20} {stats['max_delta_er']:>10.4f} {stats['mean_delta_er']:>10.4f} "
              f"{stats['highest_layer']:>12} {stats['max_conflict_er']:>12.4f} {stats['max_consistent_er']:>14.4f}")

    print("=" * 80)

    # Rank by max ΔER
    print("\nSubfolders Ranked by Max ΔER (Conflict Signal Strength):")
    ranked = sorted(summary.items(), key=lambda x: x[1]['max_delta_er'], reverse=True)
    for i, (sf, stats) in enumerate(ranked, 1):
        print(f"  {i}. {sf}: ΔER={stats['max_delta_er']:.4f} at Layer {stats['highest_layer']}")


def plot_subfolder_comparison(analyzer: CrossConflictSEAAnalyzer, conflict_type: str, save_dir: str):
    """Generate comparison plots for subfolders."""

    os.makedirs(save_dir, exist_ok=True)

    # Use the analyzer's built-in plots
    analyzer.plot_delta_er_comparison(
        save_path=os.path.join(save_dir, f"{conflict_type}_subfolder_delta_er.png"),
        show_ci=True
    )

    analyzer.plot_energy_comparison(
        save_path=os.path.join(save_dir, f"{conflict_type}_subfolder_energy.png")
    )

    analyzer.plot_summary_bar_chart(
        save_path=os.path.join(save_dir, f"{conflict_type}_subfolder_summary.png")
    )

    # Additional: heatmap of ΔER across layers and subfolders
    plot_delta_er_heatmap(analyzer, conflict_type, save_dir)


def plot_delta_er_heatmap(analyzer: CrossConflictSEAAnalyzer, conflict_type: str, save_dir: str):
    """Plot heatmap of ΔER across layers and subfolders."""

    if not analyzer.spectrum_results:
        return

    # Build matrix
    subfolders = list(analyzer.spectrum_results.keys())
    first_subfolder = subfolders[0]
    layers = sorted(analyzer.spectrum_results[first_subfolder].keys())

    delta_er_matrix = np.zeros((len(subfolders), len(layers)))

    for i, sf in enumerate(subfolders):
        for j, layer in enumerate(layers):
            if layer in analyzer.spectrum_results[sf]:
                delta_er_matrix[i, j] = analyzer.spectrum_results[sf][layer]['delta_er']

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    im = ax.imshow(delta_er_matrix, aspect='auto', cmap='RdBu_r', vmin=-0.2, vmax=0.2)

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=8)
    ax.set_yticks(range(len(subfolders)))
    ax.set_yticklabels(subfolders, fontsize=10)

    ax.set_xlabel('Layer Index', fontweight='bold')
    ax.set_ylabel('Subfolder', fontweight='bold')
    ax.set_title(f'ΔER Heatmap: {conflict_type}', fontweight='bold', fontsize=14)

    plt.colorbar(im, ax=ax, label='ΔER')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{conflict_type}_delta_er_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Heatmap saved to {save_dir}")


def main():
    args = parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, tokenizer, device = load_model(args.model_path)

    model_name = args.model_path.split("/")[-1]

    # Load subfolder data
    subfolder_data = load_subfolder_data(
        args.data_root,
        args.conflict_type,
        args.sample_limit,
        args.seed
    )

    if not subfolder_data:
        logger.error("No subfolder data loaded. Exiting.")
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

    # Run analysis for each subfolder
    logger.info(f"Starting SEA analysis for {args.conflict_type} subfolders...")
    analyzer.analyze_all_conflict_types(
        conflict_samples_dict=subfolder_data,
        target_layers=target_layers,
        top_k=args.top_k,
        compute_ci=compute_ci,
        n_bootstrap=args.n_bootstrap
    )

    # Print summary
    print_subfolder_summary(analyzer, args.conflict_type)

    # Save results
    results_dir = os.path.join(args.output_dir, model_name, "subfolder", args.conflict_type)
    analyzer.save_results(results_dir)

    # Generate plots
    plots_dir = os.path.join(results_dir, "plots")
    plot_subfolder_comparison(analyzer, args.conflict_type, plots_dir)

    logger.info(f"\nResults saved to {results_dir}")
    logger.info("SEA Subfolder Analysis Complete!")


if __name__ == "__main__":
    main()

