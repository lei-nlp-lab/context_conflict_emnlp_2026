"""
Accuracy metric for the reasoning conflict types of ContextConflict.

Section 3.1 of "Large Language Models in Resolving Contextual Knowledge Conflicts"
(EMNLP 2026) scores the three reasoning conflict types (inferential, misinformation,
temporal) with accuracy: the text inside the model's <answer></answer> tags is
compared with the gold answer(s) stored in each sample, and a sample is correct when
any gold answer matches (case/punctuation-insensitive containment, with a word-boundary
match for answers of up to three words; see `exact_match`). Two further branches are
kept from the research code for completeness but are not part of the paper's tables:
`ambiguity_conflict` uses BLEU-4 > 0.5 against all gold answers and
`granularity_conflict/medical_qa` requires every gold answer to match. The
summarization conflict types (ambiguity, granularity, perspective) are scored with the
Balance score in evaluation/balance_score.py instead.

Expected layout (written by generate_responses_api.py / generate_responses_local.py):
    <result-dir>/<model>/<conflict_type>/[<subfolder>/]<id>.json
Each JSON must contain the gold answer under "answer" (inferential_conflict/entailment_bank
files may store it under "ground_truth", which takes precedence) and the model output under
--response-key (default "response"; "{model}" is expanded to the model directory name).

Examples (from the repository root):
    python evaluation/accuracy.py --result-dir results/responses
    python evaluation/accuracy.py --result-dir results/responses \
        --models gpt-5 llama-3.1-8b-instruct --categories inferential_conflict --csv --plot

Outputs (all under --output-dir, default results/evaluation/accuracy):
    accuracy.json        correct / total / accuracy per model and conflict type
    wrong_samples.csv    paths of incorrectly answered files (only with --csv)
    accuracy_*.png       bar charts (only with --plot)
"""

import os
import re
import csv
import json
import math
import argparse
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODELS = [
    "gpt-5",
    "claude-4.5-sonnet",
    "gemini-2.5-pro",
    "gpt-oss-120b",
    "gpt-oss-20b",
    "llama-3.1-70b-instruct",
    "llama-3.1-8b-instruct",
]

# Reasoning conflict types: the ones the paper scores with accuracy.
DEFAULT_CATEGORIES = [
    "inferential_conflict",
    "misinformation_conflict",
    "temporal_conflict",
]

ALL_CATEGORIES = DEFAULT_CATEGORIES + [
    "ambiguity_conflict",
    "granularity_conflict",
    "perspective_conflict",
]

DEFAULT_OUTPUT_DIR = "results/evaluation/accuracy"


def resolve_path(path):
    """Return `path` unchanged if absolute, otherwise relative to the repository root."""
    return path if os.path.isabs(path) else os.path.join(str(REPO_ROOT), path)


def load_all_jsons(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".json"):
                yield os.path.join(root, f)


def get_subtask_from_path(file_path, category):
    """Return the sub-folder name between <category>/ and the file (e.g. 'entailment_bank'), or None."""
    try:
        path_parts = file_path.split(os.sep)
        cat_idx = path_parts.index(category)
        if cat_idx + 1 < len(path_parts):
            next_part = path_parts[cat_idx + 1]
            if not next_part.endswith('.json'):
                return next_part
    except Exception:
        pass
    return None


def extract_answer_field(data, category, file_path):
    """
    Return (gold_answers, eval_mode) for one sample.

    eval_mode "any": correct if any gold answer matches; "all": every gold answer must match;
    None: the sample is not scored.
    """
    subtask = get_subtask_from_path(file_path, category)

    if category == "ambiguity_conflict":
        answers = data.get("answer", [])
        if not isinstance(answers, list):
            answers = [answers]
        return answers, "all"

    elif category == "granularity_conflict":
        if subtask == "medical_qa":
            answers = data.get("answer", [])
            if not isinstance(answers, list):
                answers = [answers]
            return answers, "all"
        else:
            return [], None

    elif category == "inferential_conflict":
        if subtask == "entailment_bank":
            # The response files used for the paper store the short gold answer under
            # "ground_truth"; the released dataset stores the same string under "answer".
            ground_truth = data.get("ground_truth", "")
            if not ground_truth:
                ground_truth = data.get("answer", "")
            if isinstance(ground_truth, list):
                return ground_truth, "any"
            return [ground_truth] if ground_truth else [], "any"
        else:
            answers = data.get("answer", [])
            if not isinstance(answers, list):
                answers = [answers]
            return answers, "any"

    else:
        answers = data.get("answer", [])
        if not isinstance(answers, list):
            answers = [answers]
        return answers, "any"


def extract_model_response(data, response_key="response"):
    """Return the concatenated text inside <answer></answer> tags of the model output, or ''."""
    raw = data.get(response_key, "")
    if not raw:
        return ""

    matches = re.findall(r"<answer>(.*?)</answer>", raw, re.DOTALL | re.IGNORECASE)
    if matches:
        return " ".join(m.strip() for m in matches)

    return ""


def calculate_bleu(references, candidate):
    """
    Calculate BLEU-4 score.
    references: list of string references
    candidate: string candidate
    """
    def get_ngrams(segment, max_order):
        ngram_counts = Counter()
        for order in range(1, max_order + 1):
            for i in range(len(segment) - order + 1):
                ngram = tuple(segment[i:i+order])
                ngram_counts[ngram] += 1
        return ngram_counts

    def tokenize(text):
        # Simple tokenization: lowercase and split by non-alphanumeric
        return re.findall(r'\w+', text.lower())

    candidate_tokens = tokenize(candidate)
    if not candidate_tokens:
        return 0.0

    max_order = 4

    # Clipped n-gram counts
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order

    reference_tokens_list = [tokenize(ref) for ref in references]

    for order in range(1, max_order + 1):
        candidate_ngrams = Counter()
        for i in range(len(candidate_tokens) - order + 1):
            ngram = tuple(candidate_tokens[i:i+order])
            candidate_ngrams[ngram] += 1

        possible_matches_by_order[order-1] = sum(candidate_ngrams.values())

        # Max ref counts
        reference_ngrams_max = Counter()
        for ref_tokens in reference_tokens_list:
            ref_ngrams = Counter()
            for i in range(len(ref_tokens) - order + 1):
                ngram = tuple(ref_tokens[i:i+order])
                ref_ngrams[ngram] += 1
            for ngram, count in ref_ngrams.items():
                reference_ngrams_max[ngram] = max(reference_ngrams_max[ngram], count)

        # Clipped matches
        for ngram, count in candidate_ngrams.items():
            matches_by_order[order-1] += min(count, reference_ngrams_max[ngram])

    precisions = [0] * max_order
    for i in range(max_order):
        if possible_matches_by_order[i] > 0:
            precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
        else:
            precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0.0

    # Brevity penalty
    c = len(candidate_tokens)
    r = min((len(ref) for ref in reference_tokens_list), key=lambda ref_len: abs(ref_len - c))

    bp = 1.0
    if c <= r:
         bp = math.exp(1 - r / c) if c > 0 else 0.0

    return geo_mean * bp


def normalize_text(text):
    if not text:
        return ""
    text = str(text).lower().strip()

    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def ngram_match(text1, text2, n=3, threshold=0.6):
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)

    if not text1 or not text2:
        return False

    if len(text1.split()) <= 3:
        return text1 in text2

    ratio = SequenceMatcher(None, text1, text2).ratio()
    if ratio >= threshold:
        return True

    def get_ngrams(text, n):
        words = text.split()
        return set([' '.join(words[i:i+n]) for i in range(len(words)-n+1)])

    ngrams1 = get_ngrams(text1, n)
    ngrams2 = get_ngrams(text2, n)

    if not ngrams1:
        return text1 in text2

    intersection = len(ngrams1 & ngrams2)
    ngram_ratio = intersection / len(ngrams1)

    return ngram_ratio >= threshold


def exact_match(answer, response):
    answer_norm = normalize_text(answer)
    response_norm = normalize_text(response)

    if not answer_norm:
        return False

    if len(answer_norm.split()) <= 3:
        pattern = r'\b' + re.escape(answer_norm) + r'\b'
        return bool(re.search(pattern, response_norm))
    else:
        return answer_norm in response_norm


def is_match(answer, response, category):
    return exact_match(answer, response)


def evaluate_model(result_dir, model, category, response_key="response"):
    """
    Score every JSON under <result_dir>/<model>/<category>.

    Returns (correct, total, correct_files, wrong_files). Samples without a gold answer or
    without an <answer></answer> span in the response are not counted.
    """
    path = os.path.join(result_dir, model, category)
    if not os.path.exists(path):
        print(f"Warning: Path does not exist: {path}")
        return 0, 0, [], []

    key = response_key.replace("{model}", os.path.basename(model.rstrip("/")))

    total = 0
    correct = 0
    correct_files = []
    wrong_files = []

    for file_path in load_all_jsons(path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            answers, eval_mode = extract_answer_field(data, category, file_path)

            if not answers or eval_mode is None:
                continue

            response = extract_model_response(data, key)
            if not response:
                continue

            total += 1
            hit = False
            if category == "ambiguity_conflict":
                hit = calculate_bleu(answers, response) > 0.5
            elif eval_mode == "all":
                hit = all(is_match(ans, response, category) for ans in answers)
            else:
                hit = any(is_match(ans, response, category) for ans in answers)

            if hit:
                correct += 1
                correct_files.append(file_path)
            else:
                wrong_files.append(file_path)

        except Exception as e:
            print(f"Warning: Error processing {file_path}: {e}")
            continue

    return correct, total, correct_files, wrong_files


# ============================================================
# Optional plots (--plot)
# ============================================================

# Morandi color palette (lighter, more vibrant)
MORANDI_COLORS = [
    '#D4A5A5',  # Dusty pink
    '#A8C5A6',  # Soft sage
    '#B8C9D9',  # Powder blue
    '#E8C5A1',  # Warm sand
    '#C6B8D4',  # Lavender gray
    '#A8C5C5',  # Mint blue
    '#D9C4B0',  # Warm taupe
]


def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_accuracy_bars(results, models, categories, output_dir):
    """Grouped bars: one group per conflict type, one bar per model. results[model][cat] in [0, 1] or None."""
    plt = _get_plt()
    x = np.arange(len(categories))
    width = 0.09

    plt.figure(figsize=(20, 8))

    for i, model in enumerate(models):
        accs = [results[model][cat] if results[model][cat] is not None else 0 for cat in categories]
        color = MORANDI_COLORS[i % len(MORANDI_COLORS)]
        bars = plt.bar(x + i * width, accs, width=width, label=model,
                      color=color, edgecolor='black', linewidth=0.7, alpha=0.9)

        for bar, acc in zip(bars, accs):
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{acc:.3f}',
                        ha='center', va='bottom', fontsize=8)

    plt.xticks(x + width * (len(models) - 1) / 2,
               [cat.replace('_', ' ').title() for cat in categories],
               fontsize=12, fontweight='bold')

    plt.ylabel("Accuracy", fontsize=14, fontweight='bold')
    plt.xlabel("Conflict Type", fontsize=14, fontweight='bold')
    plt.title("Model Accuracy across Conflict Types", fontsize=16, fontweight='bold', pad=20)

    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left',
              frameon=True, fontsize=11, title='Models', title_fontsize=12)

    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8)
    plt.tight_layout()
    out = os.path.join(output_dir, "accuracy_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {out}")


def plot_accuracy_separate_subplots(results, models, categories, output_dir):
    """One horizontal-bar subplot per conflict type."""
    plt = _get_plt()
    fig, axes = plt.subplots(1, len(categories), figsize=(22, 7))
    axes = np.atleast_1d(axes)

    for idx, cat in enumerate(categories):
        ax = axes[idx]
        model_names = []
        accuracies = []
        colors_used = []

        for i, model in enumerate(models):
            if results[model][cat] is not None:
                model_names.append(model)
                accuracies.append(results[model][cat])
                colors_used.append(MORANDI_COLORS[i % len(MORANDI_COLORS)])

        y_pos = np.arange(len(model_names))
        bars = ax.barh(y_pos, accuracies, color=colors_used,
                      edgecolor='black', linewidth=0.8, alpha=0.9)

        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            width = bar.get_width()
            ax.text(width + 0.01, i, f'{acc:.3f}',
                   va='center', fontsize=10, fontweight='bold')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names, fontsize=11)
        ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title(cat.replace('_', ' ').title(), fontsize=13, fontweight='bold', pad=10)
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', linestyle='--', alpha=0.4)

    plt.suptitle('Model Performance Comparison by Conflict Type',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(output_dir, "accuracy_comparison_subplots.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {out}")


def plot_grouped_comparison(results, models, categories, output_dir):
    """Grouped bars: one group per model, one bar per conflict type."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(len(models))
    width = 0.25

    # Morandi color palette for categories (lighter, more vibrant)
    morandi_cat_colors = ['#D4A5A5', '#A8C5A6', '#B8C9D9']  # Dusty pink, Soft sage, Powder blue

    for i, cat in enumerate(categories):
        accs = [results[model][cat] if results[model][cat] is not None else 0
                for model in models]
        bars = ax.bar(x + i * width, accs, width=width,
                     label=cat.replace('_', ' ').title(),
                     color=morandi_cat_colors[i % len(morandi_cat_colors)],
                     edgecolor='black', linewidth=0.8, alpha=0.9)

        for bar, acc in zip(bars, accs):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{acc:.3f}',
                       ha='center', va='bottom', fontsize=8, rotation=0)

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_title('Accuracy Comparison: Models Grouped by Conflict Type',
                fontsize=15, fontweight='bold', pad=15)
    ax.legend(title='Conflict Type', fontsize=11, title_fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    out = os.path.join(output_dir, "accuracy_grouped_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {out}")


# ============================================================
# Entry point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Accuracy of <answer></answer> spans against the gold answers "
                    "(paper metric for the reasoning conflict types).")
    parser.add_argument("--result-dir", type=str, required=True,
                        help="Directory containing <model>/<conflict_type>/... response JSON files "
                             "(e.g. results/responses). Relative paths are resolved from the repository root.")
    parser.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS,
                        help="Model sub-directories of --result-dir to score. Default: the seven models of the paper.")
    parser.add_argument("--categories", type=str, nargs="+", default=DEFAULT_CATEGORIES, choices=ALL_CATEGORIES,
                        help="Conflict types to score. Default: the three reasoning types "
                             "(inferential_conflict misinformation_conflict temporal_conflict).")
    parser.add_argument("--response-key", type=str, default="response",
                        help="JSON key holding the model output; '{model}' is expanded to the model directory name "
                             "(generate_responses_*.py write '<model>_response'). Default: response")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Where accuracy.json (and optional CSV/plots) are written. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--csv", action="store_true",
                        help="Also write the paths of wrongly answered samples to <output-dir>/wrong_samples.csv")
    parser.add_argument("--plot", action="store_true",
                        help="Also write accuracy bar charts (PNG, requires matplotlib) to <output-dir>")
    return parser.parse_args()


def main():
    args = parse_args()

    result_dir = resolve_path(args.result_dir)
    output_dir = resolve_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Result dir : {result_dir}")
    print(f"Models     : {', '.join(args.models)}")
    print(f"Categories : {', '.join(args.categories)}")
    print(f"Output dir : {output_dir}")

    results = {}
    acc_table = {}
    all_wrong_rows = []

    for model in args.models:
        print(f"\n{model}")
        results[model] = {}
        acc_table[model] = {}
        for cat in args.categories:
            correct, total, correct_files, wrong_files = evaluate_model(
                result_dir, model, cat, response_key=args.response_key)
            acc = correct / total * 100 if total > 0 else 0
            print(f"  {cat}: {correct}/{total} ({acc:.1f}%)")
            results[model][cat] = {
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total > 0 else None,
            }
            acc_table[model][cat] = results[model][cat]["accuracy"]
            if args.csv:
                for fp in wrong_files:
                    all_wrong_rows.append([model, cat, fp])

    summary_path = os.path.join(output_dir, "accuracy.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAccuracy summary saved to {summary_path}")

    if args.csv and all_wrong_rows:
        csv_path = os.path.join(output_dir, "wrong_samples.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "conflict_type", "wrong_file"])
            writer.writerows(all_wrong_rows)
        print(f"Wrong samples saved to {csv_path}")

    if args.plot:
        plot_accuracy_bars(acc_table, args.models, args.categories, output_dir)
        plot_accuracy_separate_subplots(acc_table, args.models, args.categories, output_dir)
        plot_grouped_comparison(acc_table, args.models, args.categories, output_dir)


if __name__ == "__main__":
    main()
