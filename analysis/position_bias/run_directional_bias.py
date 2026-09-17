#!/usr/bin/env python3
"""
Directional evidence position-bias measurement (paper Section 4.3).

Reproduces the representation-level position-bias figures of
"Large Language Models in Resolving Contextual Knowledge Conflicts"
(EMNLP 2026): bias_simple_prompt_all_conflicts.png (Llama-3.1-8B-Instruct,
simple system prompt) and bias_stacked_area_all_conflicts_gpt20b.png
(GPT-OSS-20B). See also the appendix "Bias Measurement Implementation Notes".

Flow:
    1. Load a causal LM and collect, for every selected sample of the
       summarization conflict types (ambiguity, granularity, perspective),
       the final-token residual-stream activation of every layer for one
       combined-evidence prompt (c) and K single-evidence prompts (a_i)
       (analysis/position_bias/evidence_collector.py).
    2. Per sample and layer compute b_i = cos(c - mu, a_i - mu) with
       mu = mean_i a_i, and per layer the fraction of samples whose largest
       b_i is evidence i (analysis/position_bias/directional_bias.py).
    3. Save the activations, per-layer metrics and the stacked-area figures.

Sample selection is identical to the activation-steering training split:
sorted rglob("*.json") of the conflict-type directory, random.seed(seed),
shuffle, first max(int(n * train_ratio), min(10, n)) files, capped at
--max_samples.

Outputs (in --output_dir):
    evidence_activations_<prompt_type>.pkl         List[EvidenceActivations]
    collection_summary_<prompt_type>.json          run configuration and sample counts
    bias_metrics_per_layer_<prompt_type>.json      compute_bias_metrics_per_layer per conflict type
    directional_bias_ratios_<prompt_type>.json     conflict type -> layer -> evidence -> bias ratio
    bias_stacked_area_all_conflicts_<prompt_type>.png   three-panel figure (paper figure)
    bias_stacked_area_averaged_<prompt_type>.png        average over conflict types

Example (paper settings; run from the repository root):
    python analysis/position_bias/run_directional_bias.py \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --prompt_type simple \
        --output_dir results/analysis/position_bias/llama_8b

    python analysis/position_bias/run_directional_bias.py \
        --model_name openai/gpt-oss-20b \
        --prompt_type detailed \
        --output_dir results/analysis/position_bias/gpt_20b

Gated models read HF_TOKEN from the environment.
"""

import argparse
import json
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from analysis.position_bias.evidence_collector import EvidenceSynthesisCollector
from analysis.position_bias.directional_bias import (
    group_samples_by_conflict_type,
    compute_bias_metrics_per_layer,
    compute_layer_evidence_bias_matrix,
    visualize_bias_stacked_area_all_conflicts,
    visualize_bias_stacked_area_combined,
)


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def conflict_type_to_task_type(conflict_type: str) -> str:
    """Map a dataset directory name to the task_type label used by the collector."""
    if "ambiguity" in conflict_type:
        return "ambiguity"
    elif "granularity" in conflict_type:
        return "granularity"
    elif "perspective" in conflict_type:
        return "perspective"
    else:
        return "other"


def select_train_files(type_dir: Path, train_ratio: float, max_samples: int, seed: int):
    """
    Select the sample files of one conflict type.

    Same selection as the activation-steering training split: sorted
    rglob("*.json"), shuffle with random.seed(seed), take the first
    max(int(n * train_ratio), min(10, n)) files, then cap at max_samples.
    """
    all_files = list(type_dir.rglob("*.json"))
    all_files = sorted(all_files, key=lambda p: str(p))
    random.seed(seed)
    random.shuffle(all_files)
    n_train = int(len(all_files) * train_ratio)
    n_train = max(n_train, min(10, len(all_files)))
    train_files = all_files[:n_train]
    return train_files[:max_samples], len(all_files)


def to_jsonable(obj):
    """Convert nested dicts with int keys and numpy scalars/arrays to JSON-serializable objects."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def parse_args():
    parser = argparse.ArgumentParser(
        description="Directional evidence position-bias measurement in internal representations "
                    "(paper Section 4.3)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Hugging Face model id or local path (e.g. meta-llama/Llama-3.1-8B-Instruct, "
             "openai/gpt-oss-20b); gated models read HF_TOKEN from the environment"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="ContextConflict_Dataset/data",
        help="Root directory of the ContextConflict dataset (default: ContextConflict_Dataset/data)"
    )
    parser.add_argument(
        "--conflict_types",
        type=str,
        nargs="+",
        default=["ambiguity_conflict", "granularity_conflict", "perspective_conflict"],
        help="Conflict-type directories under --data_dir to process "
             "(paper: the three summarization types)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/analysis/position_bias",
        help="Directory for activations, metrics and figures (default: results/analysis/position_bias)"
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        choices=["simple", "detailed"],
        default="simple",
        help="Combined-evidence prompt variant: 'simple' (minimal instructions; paper figure for "
             "Llama-3.1-8B) or 'detailed' (neutrality instructions; paper figure for GPT-OSS-20B)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.2,
        help="Fraction of files per conflict type used for the measurement (default: 0.2)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10000,
        help="Maximum number of samples per conflict type (default: 10000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the file selection (default: 42)"
    )
    parser.add_argument(
        "--activation_mode",
        type=str,
        choices=["last_token", "mean_token"],
        default="last_token",
        help="How the activation is read from the hidden states (paper: last_token)"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Layers to collect and plot (default: all layers of the model)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device the tokenized prompts are moved to (default: cuda)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_all_seeds(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("DIRECTIONAL EVIDENCE POSITION-BIAS MEASUREMENT")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Data dir: {args.data_dir}")
    print(f"Conflict types: {args.conflict_types}")
    print(f"Prompt type: {args.prompt_type}")
    print(f"Activation mode: {args.activation_mode}")
    print(f"Train ratio: {args.train_ratio}, max samples: {args.max_samples}, seed: {args.seed}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    sys.stdout.flush()

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    print(f"\n[Status] Loading model: {args.model_name}")
    sys.stdout.flush()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    layers = args.layers if args.layers is not None else list(range(n_layers))
    print(f"[Status] Model loaded: {n_layers} layers; collecting {len(layers)} layers: {layers}")
    sys.stdout.flush()

    collector = EvidenceSynthesisCollector(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        target_layers=layers,
        activation_mode=args.activation_mode
    )

    # -------------------------------------------------------------------------
    # Collect single- and combined-evidence activations
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"[Collection] Collecting activations with the {args.prompt_type.upper()} prompt")
    print("=" * 70)
    sys.stdout.flush()

    data_dir = Path(args.data_dir)
    samples = []
    files_per_type = {}

    for conflict_type in args.conflict_types:
        type_dir = data_dir / conflict_type
        if not type_dir.exists():
            print(f"  Skipping {conflict_type} (not found: {type_dir})")
            continue

        task_type = conflict_type_to_task_type(conflict_type)
        train_files, total_files = select_train_files(
            type_dir, args.train_ratio, args.max_samples, args.seed
        )
        files_per_type[conflict_type] = {"selected": len(train_files), "total": total_files}

        print(f"  Collecting {conflict_type} ({len(train_files)}/{total_files} samples)...")
        sys.stdout.flush()
        for train_file in tqdm(train_files, desc=f"    {task_type}", leave=False):
            try:
                with open(train_file) as f:
                    sample_data = json.load(f)
                sample_id = f"{task_type}_{train_file.stem}"
                sample = collector.collect_sample_with_prompt_type(
                    sample_data, task_type, sample_id, prompt_type=args.prompt_type
                )
                if sample is not None:
                    samples.append(sample)
            except Exception as e:
                print(f"    Error processing {train_file}: {e}")
                continue

    print(f"\n[Collection] Collected {len(samples)} samples ({args.prompt_type} prompt)")
    sys.stdout.flush()

    # Save activations
    pkl_path = output_dir / f"evidence_activations_{args.prompt_type}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(samples, f)
    print(f"  Saved activations to {pkl_path}")

    # Save collection summary
    summary = {
        "model_name": args.model_name,
        "prompt_type": args.prompt_type,
        "timestamp": datetime.now().isoformat(),
        "config": vars(args).copy(),
        "n_samples": len(samples),
        "target_layers": layers,
        "activation_mode": args.activation_mode,
        "hidden_size": collector.hidden_size,
        "files_per_conflict_type": files_per_type,
        "samples_per_task_type": {},
        "samples_per_n_sources": {},
    }
    for sample in samples:
        tt = sample.task_type
        summary["samples_per_task_type"][tt] = summary["samples_per_task_type"].get(tt, 0) + 1
        n = sample.n_sources
        summary["samples_per_n_sources"][n] = summary["samples_per_n_sources"].get(n, 0) + 1
    summary_path = output_dir / f"collection_summary_{args.prompt_type}.json"
    with open(summary_path, 'w') as f:
        json.dump(to_jsonable(summary), f, indent=2)
    print(f"  Saved collection summary to {summary_path}")

    if not samples:
        print("No samples collected; nothing to analyze.")
        return

    # -------------------------------------------------------------------------
    # Per-layer bias metrics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[Analysis] Computing directional bias metrics per layer")
    print("=" * 70)

    samples_by_type = group_samples_by_conflict_type(samples)
    metrics_by_type = {}
    for conflict_type, ct_samples in samples_by_type.items():
        metrics_by_type[conflict_type] = compute_bias_metrics_per_layer(ct_samples, layers)
        print(f"  {conflict_type}: {len(ct_samples)} samples, "
              f"{len(metrics_by_type[conflict_type])} layers with metrics")

    metrics_path = output_dir / f"bias_metrics_per_layer_{args.prompt_type}.json"
    with open(metrics_path, 'w') as f:
        json.dump(to_jsonable(metrics_by_type), f, indent=2)
    print(f"  Saved per-layer metrics to {metrics_path}")

    # Layer x evidence bias ratios (the quantity plotted in the stacked-area figures)
    bias_ratios = compute_layer_evidence_bias_matrix(samples, layers)
    ratios_path = output_dir / f"directional_bias_ratios_{args.prompt_type}.json"
    with open(ratios_path, 'w') as f:
        json.dump(to_jsonable(bias_ratios), f, indent=2)
    print(f"  Saved layer x evidence bias ratios to {ratios_path}")

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[Visualization] Stacked-area bias figures")
    print("=" * 70)

    title_suffix = " (Simple Prompt)" if args.prompt_type == "simple" else ""

    visualize_bias_stacked_area_all_conflicts(
        samples=samples,
        target_layers=layers,
        output_path=str(output_dir / f"bias_stacked_area_all_conflicts_{args.prompt_type}.png"),
        steering_layers=None,
        title_suffix=title_suffix
    )
    visualize_bias_stacked_area_combined(
        samples=samples,
        target_layers=layers,
        output_path=str(output_dir / f"bias_stacked_area_averaged_{args.prompt_type}.png"),
        title_suffix=title_suffix
    )

    # Short console summary: average bias ratio per evidence over all layers
    print("\n" + "=" * 70)
    print("DIRECTIONAL BIAS SUMMARY (average bias ratio over layers)")
    print("=" * 70)
    for conflict_type, ct_data in bias_ratios.items():
        per_evidence = {}
        for layer_data in ct_data.values():
            for evi_id, ratio in layer_data.items():
                per_evidence.setdefault(evi_id, []).append(ratio)
        print(f"\n{conflict_type}:")
        for evi_id in sorted(per_evidence):
            print(f"  Evidence {evi_id}: {np.mean(per_evidence[evi_id]):.1%}")

    print(f"\nDone. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
