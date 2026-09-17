"""
Generate model responses for ContextConflict with a local Hugging Face model.

This is the response-generation step that precedes all metrics of Section 3.1 of
"Large Language Models in Resolving Contextual Knowledge Conflicts" (EMNLP 2026)
for open-weight models (e.g. Llama-3.1-8B/70B-Instruct, Mistral-7B-Instruct,
gpt-oss-20b/120b). It uses the same conflict-type specific prompt as
evaluation/generate_responses_api.py, greedy decoding (do_sample=False) with
bfloat16 weights sharded over the available GPUs (device_map="auto"), and writes
a copy of each sample with the answer added under the key "<model-name>_response" to

    <result-root>/<model-name>/<conflict_type>/[<subfolder>/]<id>.json

Existing output files that already contain a non-error response are skipped, so
the script can be re-run to fill gaps.

Environment variables: HF_TOKEN (gated models), HF_HOME (cache location); both optional.

Example (from the repository root):
    python evaluation/generate_responses_local.py \
        --model-path meta-llama/Llama-3.1-8B-Instruct --model-name llama-3.1-8b-instruct
    python evaluation/generate_responses_local.py \
        --model-path mistralai/Mistral-7B-Instruct-v0.3 --model-name mistral-7b-instruct \
        --conflict-types ambiguity_conflict granularity_conflict perspective_conflict

Downstream: evaluation/batch_evaluate.py --model <model-name> --response-key "{model}_response"
"""

import os
import json
import sys
import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]

# The six contextual conflict types of ContextConflict (sub-directories of the data root).
CONFLICT_TYPES = [
    "ambiguity_conflict",
    "granularity_conflict",
    "inferential_conflict",
    "misinformation_conflict",
    "perspective_conflict",
    "temporal_conflict",
]


def find_all_json_files(root_dir: str) -> list:
    """Find all JSON files in directory and subdirectories."""
    json_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return sorted(json_files)


def load_model(model_path: str, device_map: str = "auto"):
    """Load HuggingFace model and tokenizer."""
    print(f"Loading model: {model_path}")
    print(f"Device map: {device_map}")

    # Clear GPU cache before loading
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Get HF token from environment if available
    hf_token = os.environ.get("HF_TOKEN", None)
    if hf_token:
        print(f"Using HF_TOKEN for authentication")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        token=hf_token
    )

    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        token=hf_token,
        low_cpu_mem_usage=True
    )

    model.eval()

    print(f"Model loaded successfully")
    if hasattr(model, 'hf_device_map'):
        print(f"Device map: {model.hf_device_map}")
    print(f"Model dtype: {model.dtype}")

    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate response using the model (greedy decoding)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)

    # Move inputs to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response.strip()


def extract_content_from_data(data: dict) -> dict:
    """
    Extract the evidence dict from a sample.

    The released dataset stores evidence under data["content"] as
    {"source_1": text, "source_2": text, ...}. The "perspectives" branch handles a
    legacy article layout (Left/Center/Right lists) that is not used by the released
    data but is kept for compatibility.
    """
    content = {}

    if 'perspectives' in data:
        # Legacy perspective layout
        perspectives = data['perspectives']
        idx = 1
        for side in ['Left', 'Center', 'Right']:
            if side in perspectives and isinstance(perspectives[side], list):
                for article in perspectives[side]:
                    if isinstance(article, dict):
                        article_content = article.get('content', '').strip()
                        if article_content:
                            content[f"source_{idx}"] = article_content
                            idx += 1
    else:
        # Standard layout
        content = data.get("content", {})

    return content


def get_conflict_type(file_path: str, data_root: str) -> str:
    """Extract conflict type from file path."""
    rel_path = os.path.relpath(file_path, data_root)
    conflict_type = rel_path.split(os.sep)[0]
    return conflict_type


def create_prompt(question: str, content: dict, conflict_type: str) -> str:
    """Create appropriate prompt based on conflict type (identical to generate_responses_api.py)."""

    # Format evidence
    evidence_parts = []
    # Handle content dict (key: evidence_text)
    # Sort keys to ensure deterministic order if keys are like source_1, source_2
    for idx, (key, text) in enumerate(sorted(content.items()), 1):
        evidence_parts.append(f"[Evidence {idx}]: {text}")

    formatted_evidence = "\n\n".join(evidence_parts)

    # Define instructions based on conflict type (simplified from templates)
    if conflict_type == "ambiguity_conflict":
        instruction = (
            "[Instructions]: The evidence may contain ambiguous or multiple interpretations. "
            "First, briefly summarize what each evidence suggests. "
            "Then, provide ALL possible answers within <answer></answer> tags.\n"
            "Keep your entire response within 300 words."
        )
    elif conflict_type == "granularity_conflict":
        instruction = (
            "[Instructions]: This is a summarization task. First, briefly identify what each evidence provides. "
            "Then, synthesize information from ALL evidence pieces into a comprehensive and balanced summary "
            "within <answer></answer> tags. Do not favor any particular evidence.\n"
            "Keep your entire response within 300 words."
        )
    elif conflict_type == "perspective_conflict":
        instruction = (
            "[Instructions]: First, briefly identify the perspective of each evidence piece. "
            "Then, synthesize ALL perspectives into a comprehensive and balanced summary "
            "within <answer></answer> tags. Do not favor any particular viewpoint.\n"
            "Keep your entire response within 300 words."
        )
    elif conflict_type == "temporal_conflict":
        instruction = (
            "[Instructions]: Pay attention to the time point specified in the question. "
            "Select the most temporally appropriate evidence for that specific time. "
            "Explain your reasoning, then provide your final answer within <answer></answer> tags.\n"
            "Keep your entire response within 300 words."
        )
    elif conflict_type == "misinformation_conflict":
        instruction = (
            "[Instructions]: Evaluate the credibility of each piece of evidence. "
            "Identify and use only accurate evidence to reason through the question. "
            "Explain which evidence you trust and why, then provide your final answer within <answer></answer> tags.\n"
            "Keep your entire response within 300 words."
        )
    elif conflict_type == "inferential_conflict":
        instruction = (
            "[Instructions]: Analyze all evidence carefully. Select the most relevant evidence for reasoning, "
            "follow the question's output requirements, explain your reasoning process, "
            "then provide your final answer within <answer></answer> tags.\n"
            "Keep your entire response within 300 words."
        )
    else:
        # Default fallback
        instruction = (
            "[Instructions]: Analyze all evidence carefully and provide a well-reasoned answer. "
            "Explain your reasoning process first, then provide your final answer within <answer></answer> tags.\n"
            "Keep your entire response within 300 words."
        )

    # System instruction (common prefix)
    system_msg = (
        "You are a helpful assistant. You will be given a question and multiple pieces of evidence. "
        "Your task is to carefully consider ALL evidence pieces and provide a well-reasoned answer."
    )

    # Final prompt layout:
    #   <system instruction>
    #   [Instructions]: ...
    #   [Evidence 1]: ...
    #   [Evidence 2]: ...
    #
    #   [Question]: ...
    #
    #   [Response]:

    return f"""{system_msg}
{instruction}
{formatted_evidence}

[Question]: {question}

[Response]:"""


def process_json_file(file_path: str, data_root: str, result_root: str,
                      model_name: str, model, tokenizer, max_new_tokens: int = 512):
    """Process a single JSON file."""
    try:
        # Calculate relative path from data root
        rel_path = os.path.relpath(file_path, data_root)

        # Create output path in result directory with model name
        output_path = os.path.join(result_root, model_name, rel_path)

        # Response key for this run
        response_key = f"{model_name}_response"

        # Load existing result file if it exists, otherwise load from source
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Check if this response already exists
                    # We check for the key directly, and also handle hyphen/underscore variations just in case
                    keys_to_check = [response_key, response_key.replace("-", "_")]
                    found = False
                    for k in keys_to_check:
                        if k in data and data[k] and not str(data[k]).startswith("ERROR:"):
                            found = True
                            break

                    if found:
                        return "skipped"
            except (json.JSONDecodeError, Exception):
                # If error reading result file, load from source
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
        else:
            # No result file exists, load from source
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        question = data.get("question", "")
        content = extract_content_from_data(data)

        # Get conflict type and create appropriate prompt
        conflict_type = get_conflict_type(file_path, data_root)
        prompt = create_prompt(question, content, conflict_type)

        # Generate response
        answer = generate_response(model, tokenizer, prompt, max_new_tokens=max_new_tokens)

        # Add response to data (preserving all existing fields)
        data[response_key] = answer

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write to output file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        return "success"

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return "error"


def process_directory(root_dir: str, data_root: str, result_root: str,
                      model_name: str, model, tokenizer, max_new_tokens: int = 512):
    """Process all JSON files in a directory."""
    json_files = find_all_json_files(root_dir)

    if not json_files:
        print(f"No JSON files found in {root_dir}")
        return

    print(f"Found {len(json_files)} JSON files in {root_dir}")

    success = 0
    skipped = 0
    errors = 0

    for file_path in tqdm(json_files, desc=f"Processing {os.path.basename(root_dir)}"):
        result = process_json_file(file_path, data_root, result_root,
                                   model_name, model, tokenizer, max_new_tokens=max_new_tokens)
        if result == "success":
            success += 1
        elif result == "skipped":
            skipped += 1
        else:
            errors += 1

    print(f"  Success: {success}, Skipped: {skipped}, Errors: {errors}")


def count_missing_files(root_dir: str, data_root: str, result_root: str, model_name: str) -> tuple:
    """Count total and missing files for a directory."""
    json_files = find_all_json_files(root_dir)
    total = len(json_files)
    missing = 0

    response_key = f"{model_name}_response"

    for file_path in json_files:
        rel_path = os.path.relpath(file_path, data_root)
        output_path = os.path.join(result_root, model_name, rel_path)

        if not os.path.exists(output_path):
            missing += 1
            continue

        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Check for key
                keys_to_check = [response_key, response_key.replace("-", "_")]
                found = False
                for k in keys_to_check:
                    if k in data and data[k] and not str(data[k]).startswith("ERROR:"):
                        found = True
                        break
                if not found:
                    missing += 1
        except Exception:
            missing += 1

    return total, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ContextConflict responses with a local Hugging Face causal LM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (run from the repository root):
  python evaluation/generate_responses_local.py --model-path meta-llama/Llama-3.1-8B-Instruct --model-name llama-3.1-8b-instruct
  python evaluation/generate_responses_local.py --model-path meta-llama/Llama-3.1-70B-Instruct --model-name llama-3.1-70b-instruct \\
      --conflict-types inferential_conflict misinformation_conflict temporal_conflict
        """,
    )
    parser.add_argument("--model-path", type=str, required=True,
                        help="Hugging Face model id or local checkpoint directory (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Short name used as the output sub-directory and in the response key '<model-name>_response' (e.g. llama-3.1-8b-instruct)")
    parser.add_argument("--data-root", type=str, default="ContextConflict_Dataset/data",
                        help="Dataset root containing one sub-directory per conflict type (default: ContextConflict_Dataset/data, relative to the repository root)")
    parser.add_argument("--result-root", type=str, default="results/responses",
                        help="Directory where <result-root>/<model-name>/... response files are written (default: results/responses, relative to the repository root)")
    parser.add_argument("--conflict-types", type=str, nargs="+", default=CONFLICT_TYPES, choices=CONFLICT_TYPES,
                        metavar="TYPE", help="Conflict types to process (default: all six). Choices: " + ", ".join(CONFLICT_TYPES))
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Maximum number of generated tokens per response (default: 512, as used in the paper)")
    parser.add_argument("--device-map", type=str, default="auto",
                        help="Value passed to from_pretrained(device_map=...) (default: auto)")
    return parser.parse_args()


def main():
    args = parse_args()

    hf_model_path = args.model_path
    model_name = args.model_name

    print("="*80)
    print(f"Local Model Response Generation")
    print(f"HuggingFace Model: {hf_model_path}")
    print(f"Output Model Name: {model_name}")
    print("="*80)

    # Setup paths (relative paths are resolved from the repository root)
    data_root = args.data_root if os.path.isabs(args.data_root) else os.path.join(str(REPO_ROOT), args.data_root)
    result_root = args.result_root if os.path.isabs(args.result_root) else os.path.join(str(REPO_ROOT), args.result_root)

    if not os.path.exists(data_root):
        print(f"Error: Directory {data_root} does not exist")
        sys.exit(1)

    # Create result directory if it doesn't exist
    os.makedirs(result_root, exist_ok=True)

    print(f"Data root: {data_root}")
    print(f"Result root: {result_root}")
    print(f"Conflict types: {', '.join(args.conflict_types)}")
    print(f"Max new tokens: {args.max_new_tokens}")

    # Pre-scan missing files
    print("\nScanning for missing files...")
    total_files_all = 0
    total_missing_all = 0

    dir_stats = {}

    for sub in args.conflict_types:
        target_dir = os.path.join(data_root, sub)
        if not os.path.exists(target_dir):
            continue

        t, m = count_missing_files(target_dir, data_root, result_root, model_name)
        dir_stats[sub] = (t, m)
        total_files_all += t
        total_missing_all += m
        print(f"  {sub}: {m}/{t} missing ({(m/t*100) if t>0 else 0:.1f}%)")

    print(f"\nTotal Progress: {total_files_all - total_missing_all}/{total_files_all} files complete")
    print(f"Missing to process: {total_missing_all}")

    if total_missing_all == 0:
        print("\nAll files are already processed. Exiting.")
        return

    # Load model
    print("\n" + "="*80)
    print("Loading Model")
    print("="*80)
    model, tokenizer = load_model(hf_model_path, device_map=args.device_map)

    # Process each subdirectory
    for sub in args.conflict_types:
        target_dir = os.path.join(data_root, sub)
        if not os.path.exists(target_dir):
            print(f"Warning: Subdirectory {target_dir} does not exist, skipping")
            continue

        total_in_dir, missing_in_dir = dir_stats.get(sub, (0, 0))
        if missing_in_dir == 0:
            print(f"\nSkipping {sub} (0 missing)")
            continue

        print("\n" + "="*80)
        print(f"Processing: {sub} (Missing: {missing_in_dir}/{total_in_dir})")
        print("="*80)

        process_directory(target_dir, data_root, result_root, model_name, model, tokenizer,
                          max_new_tokens=args.max_new_tokens)

    print("\n" + "="*80)
    print("All files processed!")
    print("="*80)


if __name__ == "__main__":
    main()
