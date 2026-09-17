#!/usr/bin/env python3
"""
Balance score for the summarization conflict types of ContextConflict.

Section 3.1 of "Large Language Models in Resolving Contextual Knowledge Conflicts"
(EMNLP 2026) scores the summarization conflict types (ambiguity, granularity,
perspective) by how evenly a response draws on the conflicting evidence pieces.
evaluation/batch_evaluate.py first writes one Shapley-based contribution share per
evidence piece into every response JSON as <eval-key>.prob_distribution (a list that
sums to 1). This script turns those shares into per-sample statistics, all measured
against the uniform distribution over the same number of pieces, and averages them per
model and per conflict type:

- Gini coefficient (the paper's Balance score): 0 = every piece contributes equally,
  larger = the response relies on fewer pieces (lower is more balanced).
- JS divergence to uniform: 0 = identical to uniform, larger = more divergence.
- KL divergence KL(p || uniform): 0 = identical to uniform, larger = more divergence.

Expected layout: <result-dir>/<model>/<conflict_type>/[<subfolder>/]<id>.json

Examples (from the repository root):
    python evaluation/balance_score.py --result-dir results/responses
    python evaluation/balance_score.py --result-dir results/responses \
        --models gpt-5 llama-3.1-8b-instruct --eval-key eval_answer_only \
        --output results/evaluation/balance_scores_answer_only.json

Output: a JSON file (default results/evaluation/balance_scores.json) with n_samples,
mean/std of gini, js and kl per model, plus a by_conflict_type breakdown; the same
numbers are printed as tables.
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr

REPO_ROOT = Path(__file__).resolve().parents[1]

# Key under which batch_evaluate.py stored the attribution result
# (alternatives: "eval_answer_only", or the key given via --output-key).
EVAL_KEY = "eval"

DEFAULT_MODELS = [
    "gpt-5",
    "claude-4.5-sonnet",
    "gemini-2.5-pro",
    "gpt-oss-120b",
    "gpt-oss-20b",
    "llama-3.1-70b-instruct",
    "llama-3.1-8b-instruct",
]

# Summarization conflict types: the ones the paper scores with the Balance score.
DEFAULT_CONFLICT_TYPES = [
    "ambiguity_conflict",
    "granularity_conflict",
    "perspective_conflict",
]

ALL_CONFLICT_TYPES = DEFAULT_CONFLICT_TYPES + [
    "inferential_conflict",
    "misinformation_conflict",
    "temporal_conflict",
]

DEFAULT_OUTPUT = "results/evaluation/balance_scores.json"


def calculate_js_divergence(probs: np.ndarray, uniform: np.ndarray) -> float:
    """Calculate Jensen-Shannon divergence between distribution and uniform."""
    js_distance = jensenshannon(probs, uniform)
    return js_distance ** 2  # JS divergence = JS distance squared


def calculate_kl_divergence(probs: np.ndarray, uniform: np.ndarray) -> float:
    """Calculate KL divergence: KL(probs || uniform)."""
    # rel_entr computes p * log(p/q) for each element
    return np.sum(rel_entr(probs, uniform))


def calculate_gini_coefficient(probs: np.ndarray) -> float:
    """
    Calculate Gini coefficient for a distribution.

    Gini = 0: Perfect equality (uniform distribution)
    Gini -> (n-1)/n: maximal inequality (one element has all the weight). The value is
    the plain Gini estimator and is not rescaled to [0, 1].
    """
    sorted_probs = np.sort(probs)
    n = len(sorted_probs)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_probs)) / (n * np.sum(sorted_probs)) - (n + 1) / n
    return gini


def calculate_all_metrics(probs: list) -> dict:
    """
    Calculate JS, KL, and Gini for a probability distribution.

    Args:
        probs: List of probabilities

    Returns:
        Dict with 'js', 'kl', 'gini' values (or np.nan if invalid)
    """
    result = {'js': np.nan, 'kl': np.nan, 'gini': np.nan}

    if not probs or len(probs) < 2:
        return result

    probs = np.array(probs, dtype=np.float64)

    # Handle all zeros or invalid
    if np.sum(probs) == 0 or np.any(probs < 0):
        return result

    # Normalize to ensure valid probability distribution
    probs = probs / np.sum(probs)

    # Create uniform distribution
    n = len(probs)
    uniform = np.ones(n) / n

    # Add small epsilon to avoid log(0) in KL
    eps = 1e-10
    probs_safe = np.clip(probs, eps, 1.0)
    probs_safe = probs_safe / np.sum(probs_safe)

    try:
        result['js'] = calculate_js_divergence(probs, uniform)
        result['kl'] = calculate_kl_divergence(probs_safe, uniform)
        result['gini'] = calculate_gini_coefficient(probs)
    except Exception:
        pass

    return result


def load_prob_distribution(file_path: Path, eval_key: str = None) -> list:
    """Load prob_distribution (or the legacy shapley_percentage) from a JSON file."""
    if eval_key is None:
        eval_key = EVAL_KEY

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        if eval_key in data:
            if "prob_distribution" in data[eval_key]:
                return data[eval_key]["prob_distribution"]
            elif "shapley_percentage" in data[eval_key]:
                return data[eval_key]["shapley_percentage"]
    except Exception:
        pass

    return None


def calculate_metrics_for_model(
    model_dir: Path,
    conflict_types: list = None,
    eval_key: str = None
) -> dict:
    """Calculate all metrics for all samples in a model directory."""
    if eval_key is None:
        eval_key = EVAL_KEY

    if conflict_types is None:
        conflict_types = list(DEFAULT_CONFLICT_TYPES)

    all_js, all_kl, all_gini = [], [], []
    by_type = defaultdict(lambda: {'js': [], 'kl': [], 'gini': []})

    for ct in conflict_types:
        ct_dir = model_dir / ct
        if not ct_dir.exists():
            continue

        for json_file in ct_dir.rglob("*.json"):
            prob_dist = load_prob_distribution(json_file, eval_key)
            if prob_dist is not None:
                metrics = calculate_all_metrics(prob_dist)
                if not np.isnan(metrics['js']):
                    all_js.append(metrics['js'])
                    by_type[ct]['js'].append(metrics['js'])
                if not np.isnan(metrics['kl']):
                    all_kl.append(metrics['kl'])
                    by_type[ct]['kl'].append(metrics['kl'])
                if not np.isnan(metrics['gini']):
                    all_gini.append(metrics['gini'])
                    by_type[ct]['gini'].append(metrics['gini'])

    result = {
        "n_samples": len(all_js),
        "mean_js": np.mean(all_js) if all_js else np.nan,
        "mean_kl": np.mean(all_kl) if all_kl else np.nan,
        "mean_gini": np.mean(all_gini) if all_gini else np.nan,
        "std_js": np.std(all_js) if all_js else np.nan,
        "std_kl": np.std(all_kl) if all_kl else np.nan,
        "std_gini": np.std(all_gini) if all_gini else np.nan,
        "by_conflict_type": {}
    }

    for ct, vals in by_type.items():
        result["by_conflict_type"][ct] = {
            "n_samples": len(vals['js']),
            "mean_js": np.mean(vals['js']) if vals['js'] else np.nan,
            "mean_kl": np.mean(vals['kl']) if vals['kl'] else np.nan,
            "mean_gini": np.mean(vals['gini']) if vals['gini'] else np.nan,
        }

    return result


def convert_to_serializable(obj):
    """Turn numpy scalars into JSON-friendly values (NaN -> null)."""
    if isinstance(obj, np.floating):
        return float(obj) if not np.isnan(obj) else None
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    return obj


def parse_args():
    parser = argparse.ArgumentParser(
        description="Balance score (Gini of Shapley contribution shares) plus JS/KL divergence to uniform, "
                    "per model and conflict type.")
    parser.add_argument("--result-dir", type=str, required=True,
                        help="Directory containing <model>/<conflict_type>/... JSON files that already hold "
                             "<eval-key>.prob_distribution (e.g. results/responses). "
                             "Relative paths are resolved from the repository root.")
    parser.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS,
                        help="Model sub-directories of --result-dir to score. Default: the seven models of the paper.")
    parser.add_argument("--eval-key", type=str, default=EVAL_KEY,
                        help="JSON key written by batch_evaluate.py --output-key (e.g. eval, eval_answer_only). Default: eval")
    parser.add_argument("--conflict-types", type=str, nargs="+", default=DEFAULT_CONFLICT_TYPES,
                        choices=ALL_CONFLICT_TYPES,
                        help="Conflict types to score. Default: the three summarization types "
                             "(ambiguity_conflict granularity_conflict perspective_conflict).")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help=f"Output JSON file. Default: {DEFAULT_OUTPUT}")
    return parser.parse_args()


def main():
    """Compute Balance / divergence statistics for every requested model."""
    args = parse_args()

    result_dir = Path(args.result_dir if os.path.isabs(args.result_dir)
                      else os.path.join(str(REPO_ROOT), args.result_dir))
    output_file = Path(args.output if os.path.isabs(args.output)
                       else os.path.join(str(REPO_ROOT), args.output))
    conflict_types = list(args.conflict_types)

    print("=" * 72)
    print("Evidence Balance (Gini of contribution shares; lower = more balanced)")
    print(f"  Result dir     : {result_dir}")
    print(f"  eval_key       : {args.eval_key}")
    print(f"  conflict types : {', '.join(conflict_types)}")
    print("=" * 72)
    print(f"{'Model':<28} {'N':>6} {'Gini':>10} {'JS':>10} {'KL':>10}")
    print("-" * 72)

    all_results = {}

    for model_name in args.models:
        model_dir = result_dir / model_name
        if not model_dir.exists():
            print(f"{model_name:<28} {'N/A':>6} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
            continue

        result = calculate_metrics_for_model(model_dir, conflict_types=conflict_types, eval_key=args.eval_key)
        all_results[model_name] = result

        print(f"{model_name:<28} {result['n_samples']:>6} "
              f"{result['mean_gini']:>10.4f} {result['mean_js']:>10.4f} {result['mean_kl']:>10.4f}")

    # Print per conflict type breakdown
    print("\n" + "=" * 72)
    print("Breakdown by Conflict Type")
    print("=" * 72)

    for ct in conflict_types:
        print(f"\n{ct}:")
        print(f"  {'Model':<28} {'N':>6} {'Gini':>10}")
        print(f"  {'-'*50}")
        for model_name, result in all_results.items():
            if ct in result.get("by_conflict_type", {}):
                ct_result = result["by_conflict_type"][ct]
                print(f"  {model_name:<28} {ct_result['n_samples']:>6} "
                      f"{ct_result['mean_gini']:>10.4f}")

    # Save results to JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)

    print(f"\n\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
