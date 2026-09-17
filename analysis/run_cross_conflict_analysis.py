#!/usr/bin/env python3
"""
Cross-conflict analysis runner: concept vectors and spectral energy analysis.

Main entry point for the representation-level analyses of the paper
"Large Language Models in Resolving Contextual Knowledge Conflicts"
(Section 4.1, "Conflict Awareness via Concept Vectors", and the spectral
energy analysis).

For every conflict type of the ContextConflict dataset the script
1. builds paired consistent / conflict prompts with ConflictAwareDataLoader
   (Step 1),
2. loads a HuggingFace causal LM (Step 2),
3. runs the concept vector analysis (Step 3): at every layer the residual-stream
   hidden state of the last prompt token is extracted for both prompt sets, the
   concept direction is fitted on the training folds and evaluated by ROC-AUC on
   the held-out fold of a 5-fold stratified CV, together with a TF-IDF lexical
   baseline,
4. runs the spectral energy analysis (SEA) on the same samples (Step 4).

Outputs: <output_dir>/concept_vector/ (auc_scores.json, auc_details_cv.json,
lexical_baseline.json, auc_summary.json, methodology.json, concept_vectors/*.pt
and plots) and <output_dir>/sea/ (per-layer spectra and energy_summary.json).

Example:
    python analysis/run_cross_conflict_analysis.py \
        --model_path meta-llama/Llama-3.1-8B-Instruct \
        --data_root ContextConflict_Dataset/data \
        --output_dir results/analysis/cross_conflict/llama_8b \
        --sample_limit 200
"""
import argparse
import logging
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import torch
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


def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Cross-conflict analysis: concept vectors and spectral energy analysis"
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
        "--conflict_types",
        type=str,
        nargs="+",
        default=None,
        help="Conflict types to analyze (default: all)"
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=100,
        help="Number of samples per conflict type"
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
        default="results/analysis/cross_conflict",
        help="Output directory for results"
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


def main():
    """main function"""
    args = parse_args()

    logger.info("="*80)
    logger.info("Cross-Conflict Mechanism Analysis")
    logger.info("="*80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Sample limit: {args.sample_limit}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("="*80)

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("\n[Step 1] Loading Data...")

    data_loader = ConflictAwareDataLoader(data_root=args.data_root)

    # determine conflict types to analyze
    if args.conflict_types:
        conflict_types_to_analyze = args.conflict_types
    else:
        conflict_types_to_analyze = data_loader.conflict_types

    logger.info(f"Conflict types to analyze: {conflict_types_to_analyze}")

    # load data for all conflict types
    all_samples = {}
    for conflict_type in conflict_types_to_analyze:
        try:
            consistent, conflict = data_loader.load_conflict_type_samples(
                conflict_type,
                limit=args.sample_limit,
                seed=args.seed
            )

            if consistent and conflict:
                all_samples[conflict_type] = (consistent, conflict)
                logger.info(f"  {conflict_type}: {len(consistent)} consistent, {len(conflict)} conflict")
            else:
                logger.warning(f"  {conflict_type}: No valid samples, skipping")

        except Exception as e:
            logger.error(f"  {conflict_type}: Error loading data - {e}")

    if not all_samples:
        logger.error("No valid samples loaded. Exiting.")
        return


    logger.info("\n[Step 2] Loading Model...")

    model, tokenizer = load_model(args.model_path, args.device)

    # determine layers to analyze
    if args.target_layers:
        target_layers = args.target_layers
    else:
        # analyze all layers by default
        target_layers = list(range(len(model.model.layers)))

    logger.info(f"Target layers: {target_layers}")

    if not args.skip_concept_vector:
        logger.info("\n[Step 3] Running Concept Vector Analysis...")

        cv_analyzer = CrossConflictConceptVectorAnalyzer(
            model,
            tokenizer,
            args.model_path,
            device=args.device
        )

        # run analysis
        cv_analyzer.analyze_all_conflict_types(
            all_samples,
            target_layers
        )

        # generate visualizations
        cv_output_dir = os.path.join(args.output_dir, "concept_vector")
        os.makedirs(cv_output_dir, exist_ok=True)

        logger.info("\nGenerating Concept Vector visualizations...")

        # 1. AUC comparison
        cv_analyzer.plot_auc_comparison(
            save_path=os.path.join(cv_output_dir, "auc_comparison.png")
        )

        # 2. Best awareness layer comparison
        cv_analyzer.plot_best_layers_comparison(
            save_path=os.path.join(cv_output_dir, "best_layers_comparison.png")
        )

        # 3. concept vector similarity heatmap
        similarity_matrix = cv_analyzer.compute_concept_vector_similarity()
        cv_analyzer.plot_similarity_heatmap(
            similarity_matrix,
            save_path=os.path.join(cv_output_dir, "vector_similarity_heatmap.png")
        )

        # save results
        cv_analyzer.save_results(cv_output_dir)

        logger.info(f"Concept Vector analysis completed. Results saved to {cv_output_dir}")

    if not args.skip_sea:
        logger.info("\n[Step 4] Running SEA Analysis...")

        sea_analyzer = CrossConflictSEAAnalyzer(
            model,
            tokenizer,
            args.model_path,
            device=args.device
        )

        # run analysis (bootstrap CI settings use the analyzer defaults;
        # see analysis/spectral_energy/run_sea_cross_conflict.py for full control)
        sea_analyzer.analyze_all_conflict_types(
            all_samples,
            target_layers,
            top_k=args.top_k
        )

        # generate visualizations
        sea_output_dir = os.path.join(args.output_dir, "sea")
        os.makedirs(sea_output_dir, exist_ok=True)

        logger.info("\nGenerating SEA visualizations...")

        # 1. energy comparison
        sea_analyzer.plot_energy_comparison(
            save_path=os.path.join(sea_output_dir, "energy_comparison.png")
        )

        # save results
        sea_analyzer.save_results(sea_output_dir)

        logger.info(f"SEA analysis completed. Results saved to {sea_output_dir}")

    logger.info("\n" + "="*80)
    logger.info("Cross-Conflict Analysis Completed!")
    logger.info(f"All results saved to: {args.output_dir}")
    logger.info("  - concept_vector/: Concept vector analysis results")
    logger.info("  - sea/: Spectral energy analysis results")
    logger.info("="*80)


if __name__ == "__main__":
    main()
