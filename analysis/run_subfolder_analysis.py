#!/usr/bin/env python3
"""
Subfolder-level analysis runner: concept vectors and spectral energy analysis
for the source datasets that make up one conflict type.

Companion to run_cross_conflict_analysis.py.  Two conflict types of the
ContextConflict dataset are built from several source datasets:
    inferential_conflict: entailment_bank, folio, medical_qa
    perspective_conflict: allsides, perspectrum
This script treats every subfolder as its own group, builds consistent /
conflict prompt pairs with the subfolder-specific rule of
ConflictAwareDataLoader, and reports the layer-wise concept vector AUC
(5-fold stratified CV) and the spectral energy metrics per subfolder.  The
per-subfolder AUC curves are the appendix figures of Section 4.1.

Outputs: <output_dir>/<conflict_type>/concept_vector/ (subfolder_results.json,
subfolder_auc_comparison.png), <output_dir>/<conflict_type>/sea/ and
<output_dir>/<conflict_type>/combined_comparison.png.

Example:
    python analysis/run_subfolder_analysis.py \
        --model_path meta-llama/Llama-3.1-8B-Instruct \
        --conflict_type inferential_conflict \
        --data_root ContextConflict_Dataset/data \
        --output_dir results/analysis/subfolder/llama_8b \
        --sample_limit 100
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM

from analysis.conflict_aware_data_loader import ConflictAwareDataLoader
from analysis.concept_vector.cross_conflict_concept_vector import CrossConflictConceptVectorAnalyzer
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


CONFLICT_SUBFOLDERS = {
    "inferential_conflict": ["entailment_bank", "folio", "medical_qa"],
    "perspective_conflict": ["allsides", "perspectrum"]
}


def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Subfolder-level analysis: concept vectors and spectral energy analysis for one conflict type"
    )

    # model parameters
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="HuggingFace model path or name"
    )

    # data parameters
    parser.add_argument(
        "--data_root",
        type=str,
        default="ContextConflict_Dataset/data",
        help="Root directory of the ContextConflict data"
    )
    parser.add_argument(
        "--conflict_type",
        type=str,
        required=True,
        choices=["inferential_conflict", "perspective_conflict"],
        help="Conflict type to analyze in detail"
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=50,
        help="Number of samples per subfolder"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    # analysis parameters
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
        help="Top-k components for energy ratio (SEA)"
    )

    # output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/analysis/subfolder",
        help="Output directory for results (a <conflict_type>/ subdirectory is created inside it)"
    )

    # control parameters
    parser.add_argument(
        "--skip_concept_vector",
        action="store_true",
        help="Skip Concept Vector analysis"
    )
    parser.add_argument(
        "--skip_sea",
        action="store_true",
        help="Skip SEA analysis"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )

    return parser.parse_args()


def load_model(model_path: str, device: str):
    """
    load model and tokenizer

    Args:
        model_path: model path
        device: device

    Returns:
        model, tokenizer
    """
    logger.info(f"Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        logger.info(f"Using dtype: {torch_dtype}")
    else:
        torch_dtype = torch.float32

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )

    if device != "cuda":
        model = model.to(device)

    model.eval()

    logger.info(f"Model loaded successfully on {device}")
    logger.info(f"Number of layers: {len(model.model.layers)}")

    return model, tokenizer


def load_subfolder_data(data_loader: ConflictAwareDataLoader,
                       conflict_type: str,
                       sample_limit: int,
                       seed: int):
    """Build consistent / conflict prompt pairs for every subfolder of `conflict_type`."""

    subfolders = CONFLICT_SUBFOLDERS[conflict_type]
    subfolder_samples = {}

    for subfolder in subfolders:
        try:
            logger.info(f"Loading data for {conflict_type}/{subfolder}...")

            data_list = data_loader.load_json_files(
                conflict_type,
                subfolder=subfolder,
                limit=sample_limit,
                seed=seed
            )

            if not data_list:
                logger.warning(f"No data found for {subfolder}, skipping")
                continue

            if conflict_type == "inferential_conflict":
                if subfolder == "entailment_bank":
                    consistent, conflict = data_loader.process_inferential_conflict_entailment_bank(data_list)
                elif subfolder == "folio":
                    consistent, conflict = data_loader.process_inferential_conflict_folio(data_list)
                elif subfolder == "medical_qa":
                    consistent, conflict = data_loader.process_inferential_conflict_medical_qa(data_list)
            elif conflict_type == "perspective_conflict":
                if subfolder == "allsides":
                    consistent, conflict = data_loader.process_perspective_conflict_allsides(data_list)
                elif subfolder == "perspectrum":
                    consistent, conflict = data_loader.process_perspective_conflict_perspectrum(data_list)

            if consistent and conflict:
                subfolder_samples[subfolder] = (consistent, conflict)
                logger.info(f"  {subfolder}: {len(consistent)} consistent, {len(conflict)} conflict samples")
            else:
                logger.warning(f"  {subfolder}: No valid samples after processing")

        except Exception as e:
            logger.error(f"Error loading {subfolder}: {e}")

    return subfolder_samples


def plot_subfolder_auc_comparison(subfolder_results: dict,
                                  target_layers: list,
                                  save_path: str,
                                  title_suffix: str = ""):
    """Plot layer-wise concept vector AUC, one line per subfolder."""
    plt.figure(figsize=(12, 6))

    colors = sns.color_palette("husl", n_colors=len(subfolder_results))

    for idx, (subfolder, results) in enumerate(subfolder_results.items()):
        layers = sorted(results.keys())
        auc_scores = [results[layer] for layer in layers]

        plt.plot(layers, auc_scores,
                linewidth=2.5,
                label=subfolder,
                color=colors[idx],
                alpha=0.85)

    plt.xlabel("Layer", fontsize=14)
    plt.ylabel("AUC Score", fontsize=14)
    title = f"Concept Vector: AUC Comparison Across Subfolders {title_suffix}".strip()
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"AUC comparison plot saved to {save_path}")


def plot_subfolder_energy_comparison(subfolder_results: dict,
                                    target_layers: list,
                                    save_path: str,
                                    metric: str = "energy_ratio",
                                    title_suffix: str = ""):
    """Plot a layer-wise SEA metric, one line per subfolder."""
    plt.figure(figsize=(12, 6))

    colors = sns.color_palette("husl", n_colors=len(subfolder_results))

    for idx, (subfolder, results) in enumerate(subfolder_results.items()):
        layers = sorted(results.keys())
        values = [results[layer] for layer in layers]

        plt.plot(layers, values,
                linewidth=2.5,
                label=subfolder,
                color=colors[idx],
                alpha=0.85)

    if metric == "energy_ratio":
        ylabel = "Energy Ratio"
        base_title = "SEA: Energy Ratio Comparison Across Subfolders"
    elif metric == "energy_difference":
        ylabel = "Energy Difference (Conflict - Consistent)"
        base_title = "SEA: Energy Difference Comparison Across Subfolders"
    elif metric == "complexity_score":
        ylabel = "Complexity Score (Entropy Difference)"
        base_title = "SEA: Complexity Score Comparison Across Subfolders"
    else:
        ylabel = "Value"
        base_title = "SEA: Comparison Across Subfolders"

    plt.xlabel("Layer", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    title = f"{base_title} {title_suffix}".strip()
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Energy comparison plot saved to {save_path}")


def plot_combined_comparison(cv_results: dict,
                            sea_results: dict,
                            save_path: str):
    """Side-by-side plot of concept vector AUC and SEA energy ratio per subfolder."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    colors = plt.cm.Set2(np.linspace(0, 1, len(cv_results)))

    # Plot AUC
    for idx, (subfolder, results) in enumerate(cv_results.items()):
        layers = sorted(results.keys())
        auc_scores = [results[layer] for layer in layers]

        ax1.plot(layers, auc_scores,
                linewidth=2,
                label=subfolder,
                color=colors[idx])

    ax1.set_xlabel("Layer", fontsize=14)
    ax1.set_ylabel("AUC Score", fontsize=14)
    ax1.set_title("Concept Vector: AUC Comparison", fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Plot Energy Ratio
    for idx, (subfolder, results) in enumerate(sea_results.items()):
        layers = sorted(results.keys())
        energy_ratios = [results[layer] for layer in layers]

        ax2.plot(layers, energy_ratios,
                linewidth=2,
                label=subfolder,
                color=colors[idx])

    ax2.set_xlabel("Layer", fontsize=14)
    ax2.set_ylabel("Energy Ratio", fontsize=14)
    ax2.set_title("SEA: Energy Ratio Comparison", fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Combined comparison plot saved to {save_path}")


def main():
    """main function"""
    args = parse_args()

    logger.info("="*80)
    logger.info(f"Subfolder-Level Analysis: {args.conflict_type}")
    logger.info("="*80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Sample limit per subfolder: {args.sample_limit}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("="*80)

    subfolders = CONFLICT_SUBFOLDERS[args.conflict_type]
    logger.info(f"Subfolders to analyze: {subfolders}")

    output_dir = os.path.join(args.output_dir, args.conflict_type)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("\n[Step 1] Loading Data...")

    data_loader = ConflictAwareDataLoader(data_root=args.data_root)
    subfolder_samples = load_subfolder_data(
        data_loader,
        args.conflict_type,
        args.sample_limit,
        args.seed
    )

    if not subfolder_samples:
        logger.error("No valid samples loaded. Exiting.")
        return

    logger.info("\n[Step 2] Loading Model...")

    model, tokenizer = load_model(args.model_path, args.device)

    if args.target_layers:
        target_layers = args.target_layers
    else:
        # analyze all layers by default
        target_layers = list(range(len(model.model.layers)))

    logger.info(f"Target layers: {target_layers}")

    cv_linear_subfolder_results = {}  # {subfolder: {layer: auc_score}}
    sea_subfolder_results = {}  # {subfolder: {layer: spectrum result dict}}

    if not args.skip_concept_vector:
        logger.info("\n[Step 3] Running Concept Vector Analysis...")

        cv_analyzer = CrossConflictConceptVectorAnalyzer(
            model,
            tokenizer,
            args.model_path,
            device=args.device
        )

        for subfolder, (consistent, conflict) in subfolder_samples.items():
            logger.info(f"\n  Analyzing {subfolder}...")

            cv_analyzer.analyze_single_conflict_type(
                conflict_type=subfolder,
                conflict_texts=conflict,
                consistent_texts=consistent,
                target_layers=target_layers
            )

            if subfolder in cv_analyzer.auc_scores:
                cv_linear_subfolder_results[subfolder] = cv_analyzer.auc_scores[subfolder]

            cv_analyzer.auc_scores = {}
            cv_analyzer.concept_vectors = {}

        if cv_linear_subfolder_results:
            cv_output_dir = os.path.join(output_dir, "concept_vector")
            os.makedirs(cv_output_dir, exist_ok=True)

            logger.info("\nGenerating Concept Vector comparison visualizations...")

            # AUC comparison
            plot_subfolder_auc_comparison(
                cv_linear_subfolder_results,
                target_layers,
                save_path=os.path.join(cv_output_dir, "subfolder_auc_comparison.png"),
                title_suffix=""
            )

            # Save results
            with open(os.path.join(cv_output_dir, "subfolder_results.json"), 'w') as f:
                serializable_results = {}
                for subfolder, results in cv_linear_subfolder_results.items():
                    serializable_results[subfolder] = {
                        int(layer): float(score) for layer, score in results.items()
                    }
                json.dump(serializable_results, f, indent=2)

            logger.info(f"Concept Vector analysis completed. Results saved to {cv_output_dir}")

    # [Step 4] SEA Analysis
    if not args.skip_sea:
        logger.info("\n[Step 4] Running SEA Analysis...")

        sea_analyzer = CrossConflictSEAAnalyzer(
            model,
            tokenizer,
            args.model_path,
            device=args.device
        )

        for subfolder, (consistent, conflict) in subfolder_samples.items():
            logger.info(f"\n  Analyzing {subfolder}...")

            samples_dict = {subfolder: (consistent, conflict)}

            # bootstrap CI settings use the analyzer defaults;
            # see analysis/spectral_energy/run_sea_subfolder.py for full control
            sea_analyzer.analyze_all_conflict_types(
                samples_dict,
                target_layers,
                top_k=args.top_k
            )


            if subfolder in sea_analyzer.spectrum_results:
                sea_subfolder_results[subfolder] = sea_analyzer.spectrum_results[subfolder]


            sea_analyzer.spectrum_results = {}

        if sea_subfolder_results:
            sea_output_dir = os.path.join(output_dir, "sea")
            os.makedirs(sea_output_dir, exist_ok=True)

            logger.info("\nGenerating SEA comparison visualizations...")

            # Extract metrics for plotting.
            # The plotted "energy ratio" is the top-k energy ratio of the conflict
            # prompts ('conflict_er'), matching CrossConflictSEAAnalyzer.plot_energy_comparison.
            sea_energy_ratios = {}

            for subfolder, results in sea_subfolder_results.items():
                sea_energy_ratios[subfolder] = {
                    layer: data['conflict_er']
                    for layer, data in results.items()
                }

            # Plot energy ratio
            plot_subfolder_energy_comparison(
                sea_energy_ratios,
                target_layers,
                save_path=os.path.join(sea_output_dir, "subfolder_energy_comparison_ratio.png"),
                metric="energy_ratio"
            )

            with open(os.path.join(sea_output_dir, "subfolder_results.json"), 'w') as f:
                serializable_results = {}
                for subfolder, results in sea_subfolder_results.items():
                    serializable_results[subfolder] = {
                        int(layer): {
                            'energy_ratio': float(data['conflict_er']),
                            'singular_values': [float(v) for v in data.get('conflict_singular_values', [])]
                        }
                        for layer, data in results.items()
                    }
                json.dump(serializable_results, f, indent=2)

            logger.info(f"SEA analysis completed. Results saved to {sea_output_dir}")

    if cv_linear_subfolder_results and sea_subfolder_results:
        logger.info("\n[Step 5] Generating Combined Comparison...")

        sea_energy_only = {}
        for subfolder, results in sea_subfolder_results.items():
            sea_energy_only[subfolder] = {
                layer: data['conflict_er']
                for layer, data in results.items()
            }

        plot_combined_comparison(
            cv_linear_subfolder_results,
            sea_energy_only,
            save_path=os.path.join(output_dir, "combined_comparison.png")
        )

    logger.info("\n" + "="*80)
    logger.info(f"Subfolder Analysis Completed for {args.conflict_type}!")
    logger.info(f"All results saved to: {output_dir}")
    logger.info("  - concept_vector/: Concept vector analysis results")
    logger.info("  - sea/: Spectral energy analysis results")
    logger.info("="*80)


if __name__ == "__main__":
    main()
