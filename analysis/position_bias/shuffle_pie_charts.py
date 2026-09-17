#!/usr/bin/env python3
"""
Mean position shares and pie charts for the position shuffling experiment
(paper appendix "Position Shuffling Experiment", shuffle result tables).

Reads the JSON files written by analysis/position_bias/shuffle_experiment.py
(<shift_dir>/llama8b/*.json and <shift_dir>/gpt20b/*.json). For every model,
evaluation key and conflict type, the per-sample "prob_distribution" is
averaged over samples (zero-padded to the longest evidence list, then
renormalized to sum to 1). The printed "mean dist" lines are the numbers
reported in the paper's shuffle tables. One figure per model and evaluation
key (three pies, one per conflict type) is written to --output-dir as
shift_pie_<subfolder>_<eval_key>.png.

By default only the answer-only scores ("eval_answer_only") are used, as in
the paper; pass --eval-keys eval eval_answer_only to also process the
full-response scores.

Example (run from the repository root):
    python analysis/position_bias/shuffle_pie_charts.py \
        --shift-dir results/analysis/position_bias/shift \
        --output-dir results/analysis/position_bias/shuffle
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# Defaults (relative to the repository root)
DEFAULT_SHIFT_DIR = "results/analysis/position_bias/shift"
DEFAULT_OUTPUT_DIR = "results/analysis/position_bias/shuffle"

# Sub-folder of shuffle_experiment.py -> model display name
SUBFOLDERS = {
    "gpt20b": "gpt-oss-20b",
    "llama8b": "llama-3.1-8b-instruct",
}

CONFLICT_TYPES = [
    "ambiguity_conflict",
    "granularity_conflict",
    "perspective_conflict",
]

# Evaluation dicts written by shuffle_experiment.py; the paper tables use the answer-only scores
EVAL_DISPLAY = {
    "eval": "Full Response",
    "eval_answer_only": "Answer Only",
}
DEFAULT_EVAL_KEYS = ["eval_answer_only"]

# Morandi palette (same as the other pie charts of the project)
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


def load_shift_data(shift_dir, subfolder, eval_keys):
    """
    Load all shift JSONs of one model sub-folder.

    Returns:
        dict: {eval_key: {conflict_type: [prob_distribution arrays]}}
    """
    folder = os.path.join(shift_dir, subfolder)
    # {eval_key: {conflict_type: [np.array, ...]}}
    data = {ek: defaultdict(list) for ek in eval_keys}

    if not os.path.isdir(folder):
        print(f"[WARN] {folder} does not exist, skipping")
        return data

    json_files = sorted(
        [f for f in os.listdir(folder) if f.endswith(".json")],
        key=lambda x: int(x.replace(".json", ""))
    )

    for fname in json_files:
        fpath = os.path.join(folder, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            d = json.load(f)

        ct = d.get("conflict_type")
        if ct not in CONFLICT_TYPES:
            print(f"[WARN] {subfolder}/{fname}: unknown conflict_type '{ct}', skipping")
            continue

        for ek in eval_keys:
            prob = d.get(ek, {}).get("prob_distribution")
            if prob and isinstance(prob, list):
                data[ek][ct].append(np.array(prob, dtype=float))

    return data


def compute_mean_distribution(distributions):
    """
    Compute the mean prob_distribution across samples.

    Different samples may have different numbers of evidence sources.
    Shorter arrays are padded with 0 to the max length, then averaged.

    Returns:
        np.array: mean probability for each evidence position
    """
    if not distributions:
        return np.array([])

    max_len = max(len(d) for d in distributions)
    padded = np.zeros((len(distributions), max_len))
    for i, d in enumerate(distributions):
        padded[i, :len(d)] = d

    mean_vals = padded.mean(axis=0)
    # Renormalize so that the mean distribution sums to 1
    total = mean_vals.sum()
    if total > 0:
        mean_vals = mean_vals / total
    return mean_vals


def create_pie_chart(subfolder, model_display_name, eval_key, eval_display_name, data, output_dir):
    """
    Create a figure with 3 pie charts (one per conflict type) and save it to output_dir.
    """
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    colors = MORANDI_COLORS

    for idx, ct in enumerate(CONFLICT_TYPES):
        ax = axes[idx]
        distributions = data[eval_key].get(ct, [])

        if not distributions:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
            ax.set_title(ct, fontsize=12, fontweight='bold')
            ax.axis('off')
            continue

        mean_dist = compute_mean_distribution(distributions)
        n_evidence = len(mean_dist)
        percentages = mean_dist * 100  # convert to percent

        # Filter out very small contributions
        threshold = 1.0  # show only positions with >= 1%
        filtered_labels = []
        filtered_values = []
        other_value = 0.0

        for pos in range(n_evidence):
            val = percentages[pos]
            if val >= threshold:
                filtered_labels.append(f'Evidence {pos + 1}')
                filtered_values.append(val)
            else:
                other_value += val

        if other_value > 0:
            filtered_labels.append('Others')
            filtered_values.append(other_value)

        # Legend labels
        legend_labels = []
        for label, val in zip(filtered_labels, filtered_values):
            legend_labels.append(f'{label} ({val:.1f}%)')

        wedges, texts, autotexts = ax.pie(
            filtered_values,
            labels=None,
            colors=colors[:len(filtered_values)],
            autopct=lambda pct: f'{pct:.1f}%' if pct >= 5 else '',
            startangle=90,
            pctdistance=0.6,
            explode=[0.02] * len(filtered_values),
            textprops={'fontsize': 10, 'fontweight': 'bold'},
        )

        ax.legend(
            wedges, legend_labels,
            title="Evidence",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize=9,
        )

        ax.set_title(ct, fontsize=12, fontweight='bold', pad=10)

    plt.suptitle(
        f'Evidence Contribution Distribution by Conflict Type\n'
        f'Model: {model_display_name} \n'
        f'Mean prob_distribution over shift samples',
        fontsize=14, fontweight='bold', y=1.05,
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_name = f'shift_pie_{subfolder}_{eval_key}.png'
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[SAVED] {save_path}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Mean evidence share per position and pie charts for the position "
                    "shuffling experiment (output of shuffle_experiment.py)")
    parser.add_argument("--shift-dir", default=DEFAULT_SHIFT_DIR,
                        help="Output directory of shuffle_experiment.py (sub-folders llama8b/ and gpt20b/)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory the figures are written to")
    parser.add_argument("--eval-keys", nargs="+", default=DEFAULT_EVAL_KEYS,
                        choices=sorted(EVAL_DISPLAY.keys()),
                        help="Evaluation dicts to process (default: eval_answer_only, as in the paper)")
    return parser


def main():
    args = build_arg_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    n_figures = 0
    for subfolder, model_name in SUBFOLDERS.items():
        print(f"\n{'=' * 60}")
        print(f"Loading data for {model_name} ({subfolder})")
        print(f"{'=' * 60}")

        data = load_shift_data(args.shift_dir, subfolder, args.eval_keys)

        # Print the mean distribution per conflict type (the numbers reported in the paper tables)
        for ek in args.eval_keys:
            print(f"\n  [{ek}]")
            for ct in CONFLICT_TYPES:
                dists = data[ek].get(ct, [])
                if dists:
                    mean_d = compute_mean_distribution(dists)
                    print(f"    {ct}: {len(dists)} samples, "
                          f"{len(mean_d)} evidence positions, "
                          f"mean dist = [{', '.join(f'{v:.3f}' for v in mean_d)}]")
                else:
                    print(f"    {ct}: 0 samples")

        # One figure per model and evaluation key
        for ek in args.eval_keys:
            create_pie_chart(subfolder, model_name, ek, EVAL_DISPLAY[ek], data, args.output_dir)
            n_figures += 1

    print(f"\nAll done! {n_figures} figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
