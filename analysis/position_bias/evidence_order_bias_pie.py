#!/usr/bin/env python3
"""
Evidence-position pie charts computed from model outputs (paper appendix
"Evidence Position Bias Across Tested Models").

For every model and every summarization conflict type (ambiguity, granularity,
perspective) the script reads the per-sample evidence-contribution
distributions produced by the Llama-3.2-1B perplexity/Shapley scorer, sorts
each sample's distribution in descending order (position 1 = largest
contributor), normalizes it to percentages and averages it position by
position over all samples of the conflict type. The mean share of every
position is drawn as one pie per conflict type (one figure per model) and,
with --combined, as a model x conflict-type grid.

Input layout (response files written by the evaluation pipeline):
    <result_dir>/<model_name>/<conflict_type>/[<subfolder>/]<id>.json
Each JSON is read for two evaluation dicts,
    <eval_key_answer_only>  default "evaluation_with_llama1b_answer_only"
                            (scored on the <answer>...</answer> span only)
    <eval_key_full>         default "evaluation_with_llama1b"
                            (scored on the full response)
each holding a "prob_distribution" list (older files: "shapley_percentage").
When both exist with the same length, the per-sample distribution is
WEIGHT_ANSWER_ONLY * answer_only + WEIGHT_FULL * full (0.6 / 0.4); otherwise
the full-response distribution is used, falling back to answer-only.

Example (run from the repository root):
    python analysis/position_bias/evidence_order_bias_pie.py \
        --result-dir <result_dir> \
        --models gpt-5 claude-4.5-sonnet gemini-2.5-pro gpt-oss-120b \
                 gpt-oss-20b llama-3.1-70b-instruct llama-3.1-8b-instruct \
        --output-dir results/analysis/position_bias/pie_charts \
        --combined
"""

import argparse
import json
import os
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The seven models shown in the appendix figures.
DEFAULT_MODELS = [
    "gpt-5",
    "claude-4.5-sonnet",
    "gemini-2.5-pro",
    "gpt-oss-120b",
    "gpt-oss-20b",
    "llama-3.1-70b-instruct",
    "llama-3.1-8b-instruct",
]

# Summarization conflict types analyzed in the appendix figures.
TARGET_CATEGORIES = [
    "ambiguity_conflict",
    "granularity_conflict",
    "perspective_conflict",
]

# Weights of the two scorer runs when both are available for a sample.
WEIGHT_ANSWER_ONLY = 0.6  # weight of the answer-only evaluation
WEIGHT_FULL = 0.4         # weight of the full-response evaluation

# Keys of the evaluation dicts inside every response JSON.
EVAL_KEY_ANSWER_ONLY = "evaluation_with_llama1b_answer_only"
EVAL_KEY_FULL = "evaluation_with_llama1b"

DEFAULT_OUTPUT_DIR = "results/analysis/position_bias/pie_charts"

# Morandi color palette
MORANDI_COLORS = [
    '#9E8AAD',
    '#B8847C',
    '#7FA37A',
    '#8BA3B8',
    '#C9A07A',
    '#7A9E9E',
    '#B8A08A',
    '#A89888',
    '#8A9E86',
    '#B8909E',
]


def load_all_jsons(folder):
    """Load all JSON file paths from a folder recursively."""
    json_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".json"):
                json_files.append(os.path.join(root, f))
    return json_files


def get_raw_shapley_distributions(model, category, base_dir, use_weighted=True,
                                  weight_answer_only=WEIGHT_ANSWER_ONLY,
                                  weight_full=WEIGHT_FULL,
                                  eval_key_full=EVAL_KEY_FULL,
                                  eval_key_answer_only=EVAL_KEY_ANSWER_ONLY):
    """
    Get raw Shapley distributions (sorted by contribution, descending).
    Returns a list of numpy arrays, one per sample.

    Per sample the distribution is the weighted combination of
    - <eval_key_answer_only> (weight: weight_answer_only)
    - <eval_key_full>        (weight: weight_full)
    when both exist with equal length; otherwise the full-response
    distribution is used, falling back to the answer-only one.

    Args:
        model: Model name (joined with base_dir) or an absolute results directory
        category: Conflict category name (a sub-directory of the model directory)
        base_dir: Root directory of the response files (ignored for absolute model paths)
        use_weighted: If True, use the weighted combination; if False, use only <eval_key_full>
        weight_answer_only, weight_full: Weights of the two evaluation dicts
        eval_key_full, eval_key_answer_only: Names of the evaluation dicts in each JSON
    """
    if os.path.isabs(model):
        path = os.path.join(model, category)
    else:
        path = os.path.join(base_dir, model, category)
    json_files = load_all_jsons(path)

    distributions = []

    for f in json_files:
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)

            # prob_distribution of both evaluations (older files: shapley_percentage)
            eval_ao = data.get(eval_key_answer_only, {})
            eval_full = data.get(eval_key_full, {})
            shapley_answer_only = eval_ao.get("prob_distribution") or eval_ao.get("shapley_percentage")
            shapley_full = eval_full.get("prob_distribution") or eval_full.get("shapley_percentage")

            arr = None

            if use_weighted and shapley_answer_only and shapley_full:
                # Both present: weighted combination
                arr_answer_only = np.array(shapley_answer_only, dtype=float)
                arr_full = np.array(shapley_full, dtype=float)

                # Both arrays must have the same length
                if len(arr_answer_only) == len(arr_full):
                    # Weighted combination
                    arr = weight_answer_only * arr_answer_only + weight_full * arr_full
                elif shapley_full:
                    # Different lengths: use full
                    arr = np.array(shapley_full, dtype=float)
                elif shapley_answer_only:
                    # Fall back to answer_only
                    arr = np.array(shapley_answer_only, dtype=float)
            elif shapley_full:
                # Only full present
                arr = np.array(shapley_full, dtype=float)
            elif shapley_answer_only:
                # Only answer_only present (fallback)
                arr = np.array(shapley_answer_only, dtype=float)

            if arr is not None and len(arr) > 0:
                # Sort in descending order (highest contribution first)
                arr = np.array(sorted(arr, reverse=True), dtype=float)
                distributions.append(arr)
        except Exception:
            pass

    return distributions


def get_raw_shapley_distributions_subcategory(model, category, subcategory, base_dir, **loader_kwargs):
    """
    Get raw Shapley distributions for a specific subcategory (e.g., allsides, perspectrum).

    Args:
        model: Model name or absolute results directory
        category: Conflict category name (e.g., "perspective_conflict")
        subcategory: Subcategory name (e.g., "allsides", "perspectrum")
        base_dir: Root directory of the response files
        loader_kwargs: Forwarded to get_raw_shapley_distributions (weights, eval keys, use_weighted)

    Returns:
        List of numpy arrays, one per sample (sorted descending)
    """
    # Build the path to the subcategory
    if os.path.isabs(model):
        path = os.path.join(model, category, subcategory)
    else:
        path = os.path.join(base_dir, model, category, subcategory)

    if not os.path.exists(path):
        return []

    # Same parsing as the category-level loader, restricted to the subcategory folder
    return get_raw_shapley_distributions(model, os.path.join(category, subcategory), base_dir, **loader_kwargs)


def create_perspective_subcategory_comparison(model_path, base_dir, output_dir, **loader_kwargs):
    """
    Create a comparison chart for perspective_conflict subcategories (allsides vs perspectrum).
    The figure contains two pie charts side by side.

    Args:
        model_path: Model name or absolute results directory
        base_dir: Root directory of the response files
        output_dir: Directory the figure is written to
        loader_kwargs: Forwarded to get_raw_shapley_distributions
    """
    display_name = get_model_display_name(model_path)

    # Get distributions for both subcategories
    # Note: the directory name is "allsides" (with an "s")
    allsides_dist = get_raw_shapley_distributions_subcategory(
        model_path, "perspective_conflict", "allsides", base_dir, **loader_kwargs
    )
    perspectrum_dist = get_raw_shapley_distributions_subcategory(
        model_path, "perspective_conflict", "perspectrum", base_dir, **loader_kwargs
    )

    if not allsides_dist and not perspectrum_dist:
        print(f"No data found for perspective_conflict subcategories: {display_name}")
        return

    # Compute stats
    subcategories_data = {}

    if allsides_dist:
        means, stds, avg_pos = compute_evidence_position_stats(allsides_dist)
        subcategories_data['allsides'] = {
            'means': means,
            'stds': stds,
            'n_samples': len(allsides_dist),
            'avg_positions': avg_pos
        }
        print(f"  allsides: {len(allsides_dist)} samples, avg {avg_pos} evidence positions")

    if perspectrum_dist:
        means, stds, avg_pos = compute_evidence_position_stats(perspectrum_dist)
        subcategories_data['perspectrum'] = {
            'means': means,
            'stds': stds,
            'n_samples': len(perspectrum_dist),
            'avg_positions': avg_pos
        }
        print(f"  perspectrum: {len(perspectrum_dist)} samples, avg {avg_pos} evidence positions")

    # Create figure with two pie charts
    n_subcats = len(subcategories_data)
    if n_subcats == 0:
        return

    fig, axes = plt.subplots(1, n_subcats, figsize=(7 * n_subcats, 6))

    if n_subcats == 1:
        axes = [axes]

    colors = MORANDI_COLORS

    for idx, (subcat_name, result) in enumerate(subcategories_data.items()):
        ax = axes[idx]
        means = result['means']
        n_samples = result['n_samples']

        # Get contribution values and labels
        positions = sorted(means.keys())
        values = [means[pos] for pos in positions]

        # Filter out very small contributions
        threshold = 1.0
        filtered_positions = []
        filtered_values = []
        other_value = 0

        for pos, val in zip(positions, values):
            if val >= threshold:
                filtered_positions.append(pos)
                filtered_values.append(val)
            else:
                other_value += val

        if other_value > 0:
            filtered_positions.append('Others')
            filtered_values.append(other_value)

        # Create legend labels
        legend_labels = []
        for pos, val in zip(filtered_positions, filtered_values):
            if pos == 'Others':
                legend_labels.append(f'Others ({val:.1f}%)')
            else:
                legend_labels.append(f'Evidence {pos} ({val:.1f}%)')

        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            filtered_values,
            labels=None,
            colors=colors[:len(filtered_values)],
            autopct=lambda pct: f'{pct:.1f}%' if pct >= 5 else '',
            startangle=90,
            pctdistance=0.6,
            explode=[0.02] * len(filtered_values),
            textprops={'fontsize': 10, 'fontweight': 'bold'}
        )

        # Add legend
        ax.legend(wedges, legend_labels,
                  title="Evidence",
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1),
                  fontsize=9)

        # Set title with subcategory info
        conflict_strength = "Strong" if subcat_name == "perspectrum" else "Weak"
        ax.set_title(f'{subcat_name}\n({conflict_strength} conflict signal, n={n_samples})',
                     fontsize=12, fontweight='bold', pad=10)

    plt.suptitle(f'Evidence Order Bias: perspective_conflict Subcategories\nModel: {display_name}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    safe_name = display_name.replace("/", "_")
    save_path = os.path.join(output_dir, f'perspective_subcategory_{safe_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Figure saved to: {save_path}")
    plt.close()


def compute_evidence_position_stats(distributions):
    """
    Compute mean and std for each evidence position.
    Uses the average number of evidence positions (rounded up) instead of the max.

    Returns:
        position_means: dict mapping position (1-indexed) to mean percentage
        position_stds: dict mapping position (1-indexed) to std
        avg_positions: average number of evidence positions (rounded up)
    """
    if not distributions:
        return {}, {}, 0

    # Calculate average number of evidence positions (rounded up)
    lengths = [len(d) for d in distributions]
    avg_len = int(np.ceil(np.mean(lengths)))

    # Normalize each distribution and truncate/pad to avg_len
    normalized = []
    for d in distributions:
        # Normalize to percentage (0-100)
        total = np.sum(d)
        if total > 0:
            d_norm = (d / total) * 100
        else:
            d_norm = d

        # Truncate if longer than avg_len, pad if shorter
        if len(d_norm) > avg_len:
            d_final = d_norm[:avg_len]
        elif len(d_norm) < avg_len:
            d_final = np.concatenate([d_norm, np.zeros(avg_len - len(d_norm))])
        else:
            d_final = d_norm

        normalized.append(d_final)

    normalized = np.array(normalized)

    # Compute stats for each position
    position_means = {}
    position_stds = {}

    for pos in range(avg_len):
        position_means[pos + 1] = np.mean(normalized[:, pos])
        position_stds[pos + 1] = np.std(normalized[:, pos])

    return position_means, position_stds, avg_len


def analyze_model(model_name, base_dir, print_tables=True, categories=None, **loader_kwargs):
    """
    Analyze evidence contribution for a single model.

    Args:
        model_name: Name of the model (sub-directory of base_dir) or absolute results directory
        base_dir: Root directory of the response files
        print_tables: Whether to print summary tables
        categories: Conflict types to analyze (default: TARGET_CATEGORIES)
        loader_kwargs: Forwarded to get_raw_shapley_distributions (weights, eval keys, use_weighted)

    Returns:
        all_results: Dictionary containing analysis results
        summary_df: Pandas DataFrame with summary table
    """
    categories = categories or TARGET_CATEGORIES
    weight_answer_only = loader_kwargs.get('weight_answer_only', WEIGHT_ANSWER_ONLY)
    weight_full = loader_kwargs.get('weight_full', WEIGHT_FULL)

    if print_tables:
        print("="*100)
        print(f"Evidence Contribution Analysis by Position (Model: {model_name})")
        print(f"Using weighted Shapley: answer_only={weight_answer_only}, full={weight_full}")
        print("="*100)

    all_results = {}
    max_positions_all = 0

    for category in categories:
        distributions = get_raw_shapley_distributions(model_name, category, base_dir, **loader_kwargs)
        if distributions:
            means, stds, avg_pos = compute_evidence_position_stats(distributions)
            all_results[category] = {
                'means': means,
                'stds': stds,
                'n_samples': len(distributions),
                'avg_positions': avg_pos
            }
            max_positions_all = max(max_positions_all, avg_pos)
            if print_tables:
                print(f"\n{category}: {len(distributions)} samples, avg {avg_pos} evidence positions")

    # Create DataFrame for table display
    table_data = []

    for category in categories:
        if category not in all_results:
            continue

        row = {'Conflict Type': category}
        result = all_results[category]

        for pos in range(1, max_positions_all + 1):
            mean_val = result['means'].get(pos, 0)
            std_val = result['stds'].get(pos, 0)
            row[f'Evidence {pos}'] = f"{mean_val:.2f}% ± {std_val:.2f}%"

        table_data.append(row)

    df = pd.DataFrame(table_data)

    if print_tables:
        print("\n" + "="*100)
        print("Summary Table: Average Evidence Contribution (%) by Position")
        print("(Evidence positions sorted by contribution: Position 1 = highest contributor)")
        print("="*100)
        print(df.to_string(index=False))

    return all_results, df


def create_pie_charts_single_model(model_name, all_results, output_dir, categories=None):
    """
    Create pie charts for a single model showing the evidence contribution distribution
    (one pie per conflict type; the paper appendix figures).

    Args:
        model_name: Display name of the model (used in the title and file name)
        all_results: Dictionary with analysis results from analyze_model()
        output_dir: Directory the figure is written to
        categories: Conflict types in plotting order (default: TARGET_CATEGORIES)
    """
    categories = categories or TARGET_CATEGORIES
    present = [category for category in categories if category in all_results]
    n_categories = len(present)

    if n_categories == 0:
        print(f"No data available for model: {model_name}")
        return

    fig, axes = plt.subplots(1, n_categories, figsize=(7 * n_categories, 6))

    if n_categories == 1:
        axes = [axes]

    # Color palette for evidence positions
    colors = MORANDI_COLORS

    for idx, category in enumerate(present):
        ax = axes[idx]
        result = all_results[category]
        means = result['means']

        # Get contribution values and labels
        positions = sorted(means.keys())
        values = [means[pos] for pos in positions]

        # Filter out very small contributions for cleaner visualization
        threshold = 1.0  # Only show evidence with >= 1% contribution
        filtered_positions = []
        filtered_values = []
        other_value = 0

        for pos, val in zip(positions, values):
            if val >= threshold:
                filtered_positions.append(pos)
                filtered_values.append(val)
            else:
                other_value += val

        if other_value > 0:
            filtered_positions.append('Others')
            filtered_values.append(other_value)

        # Create legend labels
        legend_labels = []
        for pos, val in zip(filtered_positions, filtered_values):
            if pos == 'Others':
                legend_labels.append(f'Others ({val:.1f}%)')
            else:
                legend_labels.append(f'Evidence {pos} ({val:.1f}%)')

        # Create pie chart without labels, show percentage inside
        wedges, texts, autotexts = ax.pie(
            filtered_values,
            labels=None,  # No direct labels
            colors=colors[:len(filtered_values)],
            autopct=lambda pct: f'{pct:.1f}%' if pct >= 5 else '',
            startangle=90,
            pctdistance=0.6,
            explode=[0.02] * len(filtered_values),
            textprops={'fontsize': 10, 'fontweight': 'bold'}
        )

        # Add legend
        ax.legend(wedges, legend_labels,
                  title="Evidence",
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1),
                  fontsize=9)

        ax.set_title(f'{category}',
                     fontsize=12, fontweight='bold', pad=10)

    plt.suptitle(f'Evidence Contribution Distribution by Conflict Type\nModel: {model_name}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'evidence_order_bias_pie_chart_{model_name.replace("/", "_")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {save_path}")
    plt.close()  # Close instead of show to avoid blocking when processing multiple models


def compare_models(model_names, base_dir, output_dir, show_tables=False, categories=None, **loader_kwargs):
    """
    Compare evidence contribution across multiple models with side-by-side pie charts
    (re-analyzes every model; see create_combined_comparison_chart for pre-computed results).

    Args:
        model_names: List of model names to compare
        base_dir: Root directory of the response files
        output_dir: Directory the figure is written to
        show_tables: Whether to print summary tables for each model
        categories: Conflict types to analyze (default: TARGET_CATEGORIES)
        loader_kwargs: Forwarded to analyze_model

    Returns:
        results_dict: Dictionary mapping model names to their analysis results
    """
    categories = categories or TARGET_CATEGORIES
    results_dict = {}

    print("="*100)
    print(f"Comparing {len(model_names)} models:")
    for model in model_names:
        print(f"  - {model}")
    print("="*100)

    # Analyze each model
    for model_name in model_names:
        print(f"\nAnalyzing {model_name}...")
        all_results, summary_df = analyze_model(model_name, base_dir, print_tables=show_tables,
                                                categories=categories, **loader_kwargs)
        results_dict[model_name] = {
            'results': all_results,
            'df': summary_df
        }

    # Create comparison pie charts
    n_models = len(model_names)
    n_categories = len(categories)

    fig, axes = plt.subplots(n_models, n_categories,
                             figsize=(7 * n_categories, 6 * n_models))

    # Handle single model or single category case
    if n_models == 1 and n_categories == 1:
        axes = np.array([[axes]])
    elif n_models == 1:
        axes = axes.reshape(1, -1)
    elif n_categories == 1:
        axes = axes.reshape(-1, 1)

    colors = MORANDI_COLORS

    for model_idx, model_name in enumerate(model_names):
        all_results = results_dict[model_name]['results']

        for cat_idx, category in enumerate(categories):
            if category not in all_results:
                continue

            ax = axes[model_idx, cat_idx]
            result = all_results[category]
            means = result['means']

            # Get contribution values and labels
            positions = sorted(means.keys())
            values = [means[pos] for pos in positions]

            # Filter out very small contributions
            threshold = 1.0
            filtered_positions = []
            filtered_values = []
            other_value = 0

            for pos, val in zip(positions, values):
                if val >= threshold:
                    filtered_positions.append(pos)
                    filtered_values.append(val)
                else:
                    other_value += val

            if other_value > 0:
                filtered_positions.append('Others')
                filtered_values.append(other_value)

            # Create legend labels
            legend_labels = []
            for pos, val in zip(filtered_positions, filtered_values):
                if pos == 'Others':
                    legend_labels.append(f'Others ({val:.1f}%)')
                else:
                    legend_labels.append(f'Evi {pos} ({val:.1f}%)')

            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                filtered_values,
                labels=None,
                colors=colors[:len(filtered_values)],
                autopct=lambda pct: f'{pct:.1f}%' if pct >= 5 else '',
                startangle=90,
                pctdistance=0.6,
                explode=[0.02] * len(filtered_values),
                textprops={'fontsize': 9, 'fontweight': 'bold'}
            )

            # Add legend
            ax.legend(wedges, legend_labels,
                      title="Evidence",
                      loc="center left",
                      bbox_to_anchor=(1, 0, 0.5, 1),
                      fontsize=8)

            # Set title
            title = f'{category}'
            ax.set_title(title, fontsize=10, fontweight='bold', pad=5)

    models_str = ', '.join(model_names)
    plt.suptitle(f'Evidence Contribution Comparison Across Models\nModels: {models_str}',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'compare_models_pie_charts.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {save_path}")
    plt.close()

    return results_dict


def print_detailed_numeric_table(all_results, model_name, categories=None):
    """
    Print a detailed numeric table with separate mean and std columns.
    """
    categories = categories or TARGET_CATEGORIES

    print("\n" + "="*120)
    print(f"Detailed Numeric Table for {model_name}: Mean and Std for Each Evidence Position")
    print("="*120)

    # Find max of avg positions
    max_pos = max(r['avg_positions'] for r in all_results.values())

    # Create header
    header = f"{'Conflict Type':<25}"
    for pos in range(1, min(max_pos + 1, 8)):  # Show first 7 positions
        header += f" | {'Evi ' + str(pos) + ' Mean':>10} {'Std':>8}"
    print(header)
    print("-" * len(header))

    # Print each row
    for category in categories:
        if category not in all_results:
            continue

        result = all_results[category]
        row = f"{category:<25}"

        for pos in range(1, min(result['avg_positions'] + 1, 8)):
            mean_val = result['means'].get(pos, 0)
            std_val = result['stds'].get(pos, 0)
            row += f" | {mean_val:>10.2f} {std_val:>8.2f}"

        print(row)


def create_combined_pie_chart_single_model(model_name, all_results, output_dir, categories=None):
    """
    Create a single combined pie chart aggregating all conflict types.
    Averages the evidence contributions across all conflict types.

    Args:
        model_name: Name of the model
        all_results: Dictionary with analysis results from analyze_model()
        output_dir: Directory the figure is written to
        categories: Conflict types to aggregate (default: TARGET_CATEGORIES)
    """
    categories = categories or TARGET_CATEGORIES

    if not all_results:
        print(f"No data available for model: {model_name}")
        return

    # Aggregate contributions across all conflict types
    # Find max of avg positions across all categories
    max_pos = max(r['avg_positions'] for r in all_results.values())

    # Collect all means for averaging
    all_means = {pos: [] for pos in range(1, max_pos + 1)}
    total_samples = 0

    for category in categories:
        if category not in all_results:
            continue

        result = all_results[category]
        total_samples += result['n_samples']

        for pos in range(1, max_pos + 1):
            mean_val = result['means'].get(pos, 0)
            all_means[pos].append(mean_val)

    # Calculate average contribution for each position
    combined_means = {}
    for pos, values in all_means.items():
        if values:
            combined_means[pos] = np.mean(values)
        else:
            combined_means[pos] = 0

    # Create single pie chart
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = MORANDI_COLORS

    # Get contribution values and labels
    positions = sorted(combined_means.keys())
    values = [combined_means[pos] for pos in positions]

    # Filter out very small contributions
    threshold = 1.0
    filtered_positions = []
    filtered_values = []
    other_value = 0

    for pos, val in zip(positions, values):
        if val >= threshold:
            filtered_positions.append(pos)
            filtered_values.append(val)
        else:
            other_value += val

    if other_value > 0:
        filtered_positions.append('Others')
        filtered_values.append(other_value)

    # Create legend labels
    legend_labels = []
    for pos, val in zip(filtered_positions, filtered_values):
        if pos == 'Others':
            legend_labels.append(f'Others ({val:.1f}%)')
        else:
            legend_labels.append(f'Evidence {pos} ({val:.1f}%)')

    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        filtered_values,
        labels=None,
        colors=colors[:len(filtered_values)],
        autopct=lambda pct: f'{pct:.1f}%' if pct >= 3 else '',
        startangle=90,
        pctdistance=0.65,
        explode=[0.03] * len(filtered_values),
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )

    # Add legend
    ax.legend(wedges, legend_labels,
              title="Evidence Position",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1),
              fontsize=10,
              title_fontsize=11)

    # Set title with info about aggregation
    title = f'Combined Evidence Contribution Distribution\n'
    title += f'Model: {model_name}\n'
    title += f'(Averaged across {len(all_results)} conflict types, total n={total_samples // len(all_results)} samples avg)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'combined_pie_{model_name.replace("/", "_")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {save_path}")
    plt.close()

    return combined_means


def compare_models_combined(model_names, base_dir, output_dir, show_tables=False, categories=None, **loader_kwargs):
    """
    Compare models with combined (aggregated) pie charts - one chart per model.
    Each chart aggregates evidence contributions across all conflict types.

    Args:
        model_names: List of model names to compare
        base_dir: Root directory of the response files
        output_dir: Directory the figure is written to
        show_tables: Whether to print summary tables
        categories: Conflict types to analyze (default: TARGET_CATEGORIES)
        loader_kwargs: Forwarded to analyze_model

    Returns:
        results_dict: Dictionary with analysis results for each model
    """
    categories = categories or TARGET_CATEGORIES
    results_dict = {}

    print("="*100)
    print(f"Creating combined evidence contribution charts for {len(model_names)} models")
    print("="*100)

    # Analyze each model
    for model_name in model_names:
        print(f"\nAnalyzing {model_name}...")
        all_results, summary_df = analyze_model(model_name, base_dir, print_tables=show_tables,
                                                categories=categories, **loader_kwargs)
        results_dict[model_name] = {
            'results': all_results,
            'df': summary_df
        }

    # Create combined comparison - one pie chart per model
    n_models = len(model_names)

    if n_models == 1:
        # Single model - just create one chart
        model_name = model_names[0]
        combined_means = create_combined_pie_chart_single_model(
            model_name,
            results_dict[model_name]['results'],
            output_dir,
            categories=categories
        )
        results_dict[model_name]['combined_means'] = combined_means
    else:
        # Multiple models - create subplots
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 7 * n_rows))

        if n_models == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()

        colors = MORANDI_COLORS

        for idx, model_name in enumerate(model_names):
            all_results = results_dict[model_name]['results']
            ax = axes[idx]

            # Aggregate contributions
            max_pos = max(r['avg_positions'] for r in all_results.values())
            all_means = {pos: [] for pos in range(1, max_pos + 1)}
            total_samples = 0

            for category in categories:
                if category not in all_results:
                    continue

                result = all_results[category]
                total_samples += result['n_samples']

                for pos in range(1, max_pos + 1):
                    mean_val = result['means'].get(pos, 0)
                    all_means[pos].append(mean_val)

            combined_means = {}
            for pos, values in all_means.items():
                if values:
                    combined_means[pos] = np.mean(values)
                else:
                    combined_means[pos] = 0

            results_dict[model_name]['combined_means'] = combined_means

            # Prepare data for pie chart
            positions = sorted(combined_means.keys())
            values = [combined_means[pos] for pos in positions]

            threshold = 1.0
            filtered_positions = []
            filtered_values = []
            other_value = 0

            for pos, val in zip(positions, values):
                if val >= threshold:
                    filtered_positions.append(pos)
                    filtered_values.append(val)
                else:
                    other_value += val

            if other_value > 0:
                filtered_positions.append('Others')
                filtered_values.append(other_value)

            legend_labels = []
            for pos, val in zip(filtered_positions, filtered_values):
                if pos == 'Others':
                    legend_labels.append(f'Others ({val:.1f}%)')
                else:
                    legend_labels.append(f'Evi {pos} ({val:.1f}%)')

            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                filtered_values,
                labels=None,
                colors=colors[:len(filtered_values)],
                autopct=lambda pct: f'{pct:.1f}%' if pct >= 3 else '',
                startangle=90,
                pctdistance=0.65,
                explode=[0.03] * len(filtered_values),
                textprops={'fontsize': 10, 'fontweight': 'bold'}
            )

            ax.legend(wedges, legend_labels,
                      title="Evidence",
                      loc="center left",
                      bbox_to_anchor=(1, 0, 0.5, 1),
                      fontsize=9,
                      title_fontsize=10)

            title = f'{model_name}\n(avg across {len(all_results)} types)'
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

        # Hide extra subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')

        models_str = ', '.join(model_names)
        plt.suptitle(f'Combined Evidence Contribution Across Models\nModels: {models_str}\n(Averaged across all conflict types)',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'compare_models_combined.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
        plt.close()

    return results_dict


def get_model_display_name(model_path):
    """
    Short display name of a model: the last path component for an absolute
    results directory, otherwise the model name itself.
    """
    if os.path.isabs(model_path):
        return os.path.basename(model_path.rstrip('/'))
    return model_path


def create_combined_comparison_chart(results_dict, output_dir, save_suffix="", categories=None):
    """
    Create a combined comparison chart (model x conflict-type grid of pies) for
    multiple models from results already computed with analyze_model.

    Args:
        results_dict: {model: {'results': ..., 'display_name': ...}}
        output_dir: Directory the figure is written to
        save_suffix: Optional suffix for the saved filename (e.g., "_base_models")
        categories: Conflict types in column order (default: TARGET_CATEGORIES)
    """
    categories = categories or TARGET_CATEGORIES
    n_models = len(results_dict)
    n_categories = len(categories)

    if n_models == 0:
        print("No data to compare")
        return

    fig, axes = plt.subplots(n_models, n_categories,
                             figsize=(7 * n_categories, 6 * n_models))

    # Handle single model or single category case
    if n_models == 1 and n_categories == 1:
        axes = np.array([[axes]])
    elif n_models == 1:
        axes = axes.reshape(1, -1)
    elif n_categories == 1:
        axes = axes.reshape(-1, 1)

    colors = MORANDI_COLORS

    for model_idx, (model_path, data) in enumerate(results_dict.items()):
        all_results = data['results']
        display_name = data['display_name']

        for cat_idx, category in enumerate(categories):
            if category not in all_results:
                axes[model_idx, cat_idx].text(0.5, 0.5, 'No Data',
                                               ha='center', va='center', fontsize=12)
                axes[model_idx, cat_idx].axis('off')
                continue

            ax = axes[model_idx, cat_idx]
            result = all_results[category]
            means = result['means']

            # Get contribution values and labels
            positions = sorted(means.keys())
            values = [means[pos] for pos in positions]

            # Filter out very small contributions
            threshold = 1.0
            filtered_positions = []
            filtered_values = []
            other_value = 0

            for pos, val in zip(positions, values):
                if val >= threshold:
                    filtered_positions.append(pos)
                    filtered_values.append(val)
                else:
                    other_value += val

            if other_value > 0:
                filtered_positions.append('Others')
                filtered_values.append(other_value)

            # Create legend labels
            legend_labels = []
            for pos, val in zip(filtered_positions, filtered_values):
                if pos == 'Others':
                    legend_labels.append(f'Others ({val:.1f}%)')
                else:
                    legend_labels.append(f'Evi {pos} ({val:.1f}%)')

            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                filtered_values,
                labels=None,
                colors=colors[:len(filtered_values)],
                autopct=lambda pct: f'{pct:.1f}%' if pct >= 5 else '',
                startangle=90,
                pctdistance=0.6,
                explode=[0.02] * len(filtered_values),
                textprops={'fontsize': 9, 'fontweight': 'bold'}
            )

            # Add legend
            ax.legend(wedges, legend_labels,
                      title="Evidence",
                      loc="center left",
                      bbox_to_anchor=(1, 0, 0.5, 1),
                      fontsize=8)

            # Set title: model name and category
            title = f'{display_name}\n{category}'
            ax.set_title(title, fontsize=10, fontweight='bold', pad=5)

    plt.suptitle('Evidence Contribution Comparison Across Models/Methods',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save figure with optional suffix
    os.makedirs(output_dir, exist_ok=True)
    filename = f'compare_models{save_suffix}.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nCombined comparison figure saved to: {save_path}")
    plt.close()


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Mean evidence-contribution share per (sorted) evidence position, "
                    "drawn as pie charts per model and conflict type "
                    "(appendix 'Evidence Position Bias Across Tested Models').")
    parser.add_argument("--result-dir", required=True,
                        help="Root of the response files: "
                             "<result-dir>/<model>/<conflict_type>/[<subfolder>/]<id>.json")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Model sub-directories of --result-dir to analyze "
                             "(default: the seven models of the paper)")
    parser.add_argument("--categories", nargs="+", default=TARGET_CATEGORIES,
                        help="Conflict-type sub-directories to analyze "
                             "(default: the three summarization conflict types)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory the figures are written to")
    parser.add_argument("--weight-answer-only", type=float, default=WEIGHT_ANSWER_ONLY,
                        help="Weight of the answer-only evaluation in the per-sample combination")
    parser.add_argument("--weight-full", type=float, default=WEIGHT_FULL,
                        help="Weight of the full-response evaluation in the per-sample combination")
    parser.add_argument("--eval-key-full", default=EVAL_KEY_FULL,
                        help="JSON key of the full-response evaluation dict")
    parser.add_argument("--eval-key-answer-only", default=EVAL_KEY_ANSWER_ONLY,
                        help="JSON key of the answer-only evaluation dict")
    parser.add_argument("--no-weighted", action="store_true",
                        help="Do not combine the two evaluations; use the full-response one "
                             "(answer-only as fallback)")
    parser.add_argument("--combined", action="store_true",
                        help="Also draw one model x conflict-type grid comparing all analyzed models")
    parser.add_argument("--perspective-subcategories", action="store_true",
                        help="Also draw, per model, the allsides vs. perspectrum comparison "
                             "for perspective_conflict")
    parser.add_argument("--detailed-table", action="store_true",
                        help="Also print the mean/std table with separate columns")
    return parser


def main():
    args = build_arg_parser().parse_args()

    loader_kwargs = dict(
        use_weighted=not args.no_weighted,
        weight_answer_only=args.weight_answer_only,
        weight_full=args.weight_full,
        eval_key_full=args.eval_key_full,
        eval_key_answer_only=args.eval_key_answer_only,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("Evidence position bias: mean contribution share per evidence position")
    print(f"Result directory: {args.result_dir}")
    print(f"Models ({len(args.models)}): {', '.join(args.models)}")
    print(f"Conflict types: {', '.join(args.categories)}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)

    results_dict = {}

    # One figure per model (three pies, one per conflict type)
    for model in args.models:
        display_name = get_model_display_name(model)
        print(f"\n{'='*60}")
        print(f"Processing: {display_name}")
        print(f"{'='*60}")

        try:
            all_results, summary_df = analyze_model(
                model, args.result_dir, print_tables=True,
                categories=args.categories, **loader_kwargs
            )
            if not all_results:
                print(f"No data found for: {model}")
                continue

            if args.detailed_table:
                print_detailed_numeric_table(all_results, display_name, categories=args.categories)

            create_pie_charts_single_model(display_name, all_results, args.output_dir,
                                           categories=args.categories)

            if args.perspective_subcategories:
                create_perspective_subcategory_comparison(model, args.result_dir, args.output_dir,
                                                          **loader_kwargs)

            results_dict[model] = {
                'results': all_results,
                'df': summary_df,
                'display_name': display_name
            }
        except Exception as e:
            print(f"Error processing {model}: {e}")
            traceback.print_exc()

    # Optional model x conflict-type comparison grid
    if args.combined:
        if len(results_dict) > 1:
            create_combined_comparison_chart(results_dict, args.output_dir, categories=args.categories)
        else:
            print("\n--combined needs results for at least two models; skipping the comparison grid")

    print("\n" + "="*80)
    print(f"Done. {len(results_dict)} model(s) analyzed; figures written to {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
