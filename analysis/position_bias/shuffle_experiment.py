#!/usr/bin/env python3
"""
Position shuffling experiment (paper appendix "Position Shuffling Experiment").

Tests whether the evidence-position bias seen in the pie charts of
analysis/position_bias/evidence_order_bias_pie.py follows the position of an
evidence document rather than its content:

1. Sample SAMPLES_PER_TYPE (33) items from each summarization conflict type
   (ambiguity, granularity, perspective) of ContextConflict, 99 in total.
2. For each model in MODELS (Llama-3.1-8B-Instruct, GPT-OSS-20B) randomly
   permute the evidence list of every sampled item and generate a response.
3. Score every response with the Llama-3.2-1B perplexity/Shapley scorer
   (Evaluator.perplexity_evaluation) in two modes: on the full response
   ("eval") and on the <answer>...</answer> span only ("eval_answer_only").
4. Write one JSON per item to <output_dir>/llama8b/ and <output_dir>/gpt20b/
   holding the original sample fields plus "response", "conflict_type",
   "original_path", "eval" and "eval_answer_only".

analysis/position_bias/shuffle_pie_charts.py turns these files into the
mean share per evidence position (paper shuffle tables) and pie charts.

Example (run from the repository root; needs a GPU, and HF_TOKEN for gated models):
    python analysis/position_bias/shuffle_experiment.py \
        --data-root ContextConflict_Dataset/data \
        --output-dir results/analysis/position_bias/shift \
        --seed 42 --max-new-tokens 512
"""

import argparse
import glob
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.evaluator import HFGenerativeModel, Evaluator


# Model config: (label, subfolder name, HF model path)
MODELS = [
    ("llama-3.1-8b-instruct", "llama8b", "meta-llama/Llama-3.1-8B-Instruct"),
    ("gpt-oss-20b",           "gpt20b",  "openai/gpt-oss-20b"),
]

CONFLICT_TYPES = ["ambiguity_conflict", "granularity_conflict", "perspective_conflict"]
SAMPLES_PER_TYPE = 33
EVAL_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

DEFAULT_DATA_ROOT = "ContextConflict_Dataset/data"
DEFAULT_OUTPUT_DIR = "results/analysis/position_bias/shift"


def extract_evidence(data):
    """Extract evidence list from either 'content' or 'perspectives' format."""
    if "perspectives" in data:
        evidence = []
        for perspective in ["Left", "Center", "Right"]:
            if perspective in data["perspectives"]:
                for article in data["perspectives"][perspective]:
                    if isinstance(article, dict):
                        content = article.get("content", "").strip()
                        if content:
                            evidence.append(content)
        return evidence
    elif "content" in data:
        content = data["content"]
        source_keys = sorted(content.keys())
        return [
            content[sk].strip()
            for sk in source_keys
            if isinstance(content[sk], str) and content[sk].strip()
        ]
    return []


def extract_answer_only(response):
    """Extract content within <answer></answer> tags."""
    if not response:
        return response
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    print(f"    [WARN] No <answer> tags found, using full response for answer_only")
    return response


def run_perplexity_eval(evaluator, question, evidence, response):
    """Run perplexity evaluation, return dict with shapley_values and prob_distribution."""
    try:
        results = evaluator.perplexity_evaluation(
            question=question,
            evidence=evidence,
            response=response,
            return_raw=True
        )
        num_clusters = len(results['shapley_values'])
        shapley_list = [results['shapley_values'].get(i, 0.0) for i in range(num_clusters)]
        prob_list = [results['prob_distribution'].get(i, 0.0) for i in range(num_clusters)]
        return {"shapley_values": shapley_list, "prob_distribution": prob_list}
    except Exception as e:
        print(f"    [EVAL ERROR] {e}")
        return None


def sample_data(data_root, seed):
    """Sample SAMPLES_PER_TYPE JSONs from each of the 3 conflict types (99 total)."""
    random.seed(seed)
    sampled = []

    for ct in CONFLICT_TYPES:
        ct_dir = os.path.join(data_root, ct)
        all_files = glob.glob(os.path.join(ct_dir, "**", "*.json"), recursive=True)
        print(f"  {ct}: {len(all_files)} total files")

        if len(all_files) < SAMPLES_PER_TYPE:
            print(f"  [WARN] Only {len(all_files)} files, using all")
            selected = all_files
        else:
            selected = random.sample(all_files, SAMPLES_PER_TYPE)

        for fpath in selected:
            sampled.append({"path": fpath, "conflict_type": ct})

    random.shuffle(sampled)
    print(f"  Total sampled: {len(sampled)}")
    return sampled


def main():
    parser = argparse.ArgumentParser(
        description="Generate responses with shuffled evidence order and score them "
                    "with the Llama-3.2-1B perplexity scorer (position shuffling experiment)")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT,
                        help="Root of the ContextConflict data (one sub-directory per conflict type)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory; one sub-folder per model (llama8b/, gpt20b/)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (sampling and shuffling)")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens")
    parser.add_argument("--eval-model", type=str, default=EVAL_MODEL, help="Scorer model (HF id or path)")
    parser.add_argument("--skip-gen", action="store_true", help="Skip generation, only evaluate")
    args = parser.parse_args()

    # ---- Step 1: Sample data ----
    print("=" * 60)
    print("Step 1: Sampling data")
    print("=" * 60)
    sampled = sample_data(args.data_root, args.seed)

    # ---- Step 2: Generate responses for each model ----
    for model_label, subfolder, hf_path in MODELS:
        out_dir = os.path.join(args.output_dir, subfolder)
        os.makedirs(out_dir, exist_ok=True)

        if not args.skip_gen:
            print(f"\n{'=' * 60}")
            print(f"Step 2: Generating responses with {model_label}")
            print(f"  Model: {hf_path}")
            print(f"  Output: {out_dir}/")
            print(f"{'=' * 60}")

            model = HFGenerativeModel(hf_path)
            print("Model loaded.\n")

            random.seed(args.seed)  # reset seed so shuffles are reproducible

            for idx, item in enumerate(sampled, start=1):
                out_path = os.path.join(out_dir, f"{idx}.json")

                # Skip if already generated
                if os.path.exists(out_path):
                    with open(out_path, "r") as f:
                        existing = json.load(f)
                    if existing.get("response"):
                        print(f"[SKIP] {idx}.json: already has response")
                        continue

                with open(item["path"], "r") as f:
                    data = json.load(f)

                evidence_list = extract_evidence(data)
                question = data.get("question", "")

                if not evidence_list or not question:
                    print(f"[SKIP] {idx}.json: missing question or evidence")
                    continue

                # Shuffle evidence
                shuffled = evidence_list.copy()
                random.shuffle(shuffled)

                print(f"[GEN] {idx}.json (type={item['conflict_type']}, {len(evidence_list)} sources)")

                try:
                    response = model.generate(question, shuffled, max_new_tokens=args.max_new_tokens)
                except Exception as e:
                    print(f"  [ERROR] generation failed: {e}")
                    continue

                if not response or not response.strip():
                    print(f"  [FAIL] empty response")
                    continue

                # Build output JSON: keep original data + add response
                out_data = dict(data)
                out_data["response"] = response
                out_data["conflict_type"] = item["conflict_type"]
                out_data["original_path"] = item["path"]

                with open(out_path, "w") as f:
                    json.dump(out_data, f, ensure_ascii=False, indent=2)

                print(f"  [OK] saved ({len(response)} chars)")

            # Free GPU memory
            del model
            import torch
            torch.cuda.empty_cache()
            print(f"\nModel {model_label} unloaded.")

    # ---- Step 3: Evaluate all generated responses ----
    print(f"\n{'=' * 60}")
    print(f"Step 3: Evaluating with {args.eval_model}")
    print(f"{'=' * 60}")

    evaluator = Evaluator(
        model=args.eval_model,
        nli_model=None,
        helper_model=None,
        enable_clustering=False,
        use_api=False
    )
    print("Evaluator loaded.\n")

    for _, subfolder, _ in MODELS:
        out_dir = os.path.join(args.output_dir, subfolder)
        if not os.path.exists(out_dir):
            print(f"[SKIP] {out_dir} does not exist")
            continue

        json_files = sorted(
            [f for f in os.listdir(out_dir) if f.endswith(".json")],
            key=lambda x: int(x.replace(".json", ""))
        )
        print(f"\nEvaluating {subfolder}/ ({len(json_files)} files)")

        for fname in json_files:
            fpath = os.path.join(out_dir, fname)
            with open(fpath, "r") as f:
                data = json.load(f)

            response = data.get("response", "")
            if not response:
                print(f"  [SKIP] {fname}: no response")
                continue

            # Skip if already fully evaluated
            if "eval" in data and "eval_answer_only" in data:
                print(f"  [SKIP] {fname}: already evaluated")
                continue

            question = data.get("question", "")
            evidence = extract_evidence(data)
            if not evidence:
                print(f"  [SKIP] {fname}: no evidence")
                continue

            print(f"  [EVAL] {fname}")

            # Normal mode
            if "eval" not in data:
                print(f"    Running perplexity (normal)...")
                eval_result = run_perplexity_eval(evaluator, question, evidence, response)
                if eval_result:
                    data["eval"] = eval_result

            # Answer-only mode
            if "eval_answer_only" not in data:
                answer_only_response = extract_answer_only(response)
                print(f"    Running perplexity (answer_only)...")
                eval_ao_result = run_perplexity_eval(evaluator, question, evidence, answer_only_response)
                if eval_ao_result:
                    data["eval_answer_only"] = eval_ao_result

            with open(fpath, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"    [DONE] {fname}")

    print(f"\n{'=' * 60}")
    print("All done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
