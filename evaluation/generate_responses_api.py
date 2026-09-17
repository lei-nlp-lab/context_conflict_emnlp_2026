"""
Generate model responses for ContextConflict with an OpenAI-compatible API.

This is the response-generation step that precedes all metrics of Section 3.1 of
"Large Language Models in Resolving Contextual Knowledge Conflicts" (EMNLP 2026)
for API-served models (e.g. GPT, Claude and Gemini through an OpenAI-compatible
gateway, or DeepSeek). For every sample JSON under the data root it builds a
conflict-type specific prompt (question + all evidence pieces + instructions),
queries the model once, and writes a copy of the sample with the answer added
under the key "<model>_response" to

    <result-root>/<model>/<conflict_type>/[<subfolder>/]<id>.json

Existing output files that already contain a non-error response are skipped, so
the script can be re-run to fill gaps. Local Hugging Face models use the same
prompt through evaluation/generate_responses_local.py.

Credentials come from environment variables only:
    OPENAI_API_KEY   (or the variable named by --api-key-env, e.g. DEEPSEEK_API_KEY)
    OPENAI_BASE_URL  optional; leave unset for the provider default endpoint

Example (from the repository root):
    export OPENAI_API_KEY=...
    python evaluation/generate_responses_api.py --model gpt-5
    python evaluation/generate_responses_api.py --model deepseek-chat --api-key-env DEEPSEEK_API_KEY \
        --conflict-types ambiguity_conflict perspective_conflict

Downstream: evaluation/batch_evaluate.py --model <model> --response-key "{model}_response"
"""

import os
import sys
import json
import argparse
import asyncio
from pathlib import Path
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


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


async def call_llm_api(prompt: str, model: str = "gpt-5", api_key: str = None, base_url: str = None) -> str:
    """Send one chat completion request; returns the text or an "ERROR: ..." string."""
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")

    if base_url is None:
        base_url = os.environ.get("OPENAI_BASE_URL")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    print(f"--- Calling OpenAI-compatible API (Model: {model}) ---")
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred while calling the API: {e}")
        return f"ERROR: {e}"


def find_all_json_files(root_dir: str) -> list:
    json_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files


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
                        # Prefer the short preview over the full article text
                        article_content = article.get('preview', '').strip() or article.get('content', '').strip()
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
    """Create appropriate prompt based on conflict type."""

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

    return f"""{system_msg}
{instruction}
{formatted_evidence}

[Question]: {question}

[Response]:"""


async def process_json_file(file_path: str, data_root: str, result_root: str, model: str = "gpt-5", api_key: str = None, base_url: str = None):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Calculate relative path from data root
        rel_path = os.path.relpath(file_path, data_root)

        # Create output path in result directory with model name
        output_path = os.path.join(result_root, model, rel_path)

        # Check if output file already exists and has the response
        response_key = f"{model}_response"
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)

                    # Check for key (and variations)
                    keys_to_check = [response_key, response_key.replace("-", "_")]
                    found = False
                    for k in keys_to_check:
                        if k in existing_data and existing_data[k] and not str(existing_data[k]).startswith("ERROR:"):
                            found = True
                            break

                    if found:
                        # Already has a valid response
                        return

            except (json.JSONDecodeError, Exception) as e:
                print(f"Re-processing {os.path.basename(file_path)} - existing file corrupted: {e}")

        question = data.get("question", "")
        content = extract_content_from_data(data)

        # Get conflict type and create appropriate prompt
        conflict_type = get_conflict_type(file_path, data_root)
        prompt = create_prompt(question, content, conflict_type)

        print(f"Processing: {file_path}")
        answer = await call_llm_api(prompt, model=model, api_key=api_key, base_url=base_url)

        # Add response to data (keep original structure)
        data[response_key] = answer

        # The evidence dict is only used to build the prompt; the original sample
        # fields are written unchanged.

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write to output file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"Saved response to {output_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


async def process_directory(root_dir: str, data_root: str, result_root: str, model: str = "gpt-5", api_key: str = None, base_url: str = None, desc: str = ""):
    json_files = find_all_json_files(root_dir)

    # Files are processed sequentially (one request at a time) with a progress bar
    for file_path in tqdm_asyncio(json_files, desc=desc):
        await process_json_file(file_path, data_root, result_root, model, api_key, base_url)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ContextConflict responses with an OpenAI-compatible chat API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (run from the repository root):
  export OPENAI_API_KEY=...
  python evaluation/generate_responses_api.py --model gpt-5
  python evaluation/generate_responses_api.py --model gpt-5 --conflict-types temporal_conflict misinformation_conflict
  OPENAI_BASE_URL=https://api.deepseek.com python evaluation/generate_responses_api.py \\
      --model deepseek-chat --api-key-env DEEPSEEK_API_KEY
        """,
    )
    parser.add_argument("--model", type=str, default="gpt-5",
                        help="Model name sent to the API; also used as the output sub-directory and in the response key '<model>_response' (default: gpt-5)")
    parser.add_argument("--data-root", type=str, default="ContextConflict_Dataset/data",
                        help="Dataset root containing one sub-directory per conflict type (default: ContextConflict_Dataset/data, relative to the repository root)")
    parser.add_argument("--result-root", type=str, default="results/responses",
                        help="Directory where <result-root>/<model>/... response files are written (default: results/responses, relative to the repository root)")
    parser.add_argument("--conflict-types", type=str, nargs="+", default=CONFLICT_TYPES, choices=CONFLICT_TYPES,
                        metavar="TYPE", help="Conflict types to process (default: all six). Choices: " + ", ".join(CONFLICT_TYPES))
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY",
                        help="Name of the environment variable holding the API key (default: OPENAI_API_KEY; e.g. DEEPSEEK_API_KEY). The base URL is read from OPENAI_BASE_URL if set.")
    return parser.parse_args()


async def main():
    args = parse_args()

    model = args.model
    api_key = os.environ.get(args.api_key_env)
    base_url = os.environ.get("OPENAI_BASE_URL")

    if not api_key:
        print(f"Error: environment variable {args.api_key_env} is not set")
        sys.exit(1)

    data_root = args.data_root if os.path.isabs(args.data_root) else os.path.join(str(REPO_ROOT), args.data_root)
    result_root = args.result_root if os.path.isabs(args.result_root) else os.path.join(str(REPO_ROOT), args.result_root)

    if not os.path.exists(data_root):
        print(f"Error: Directory {data_root} does not exist")
        sys.exit(1)

    # Create result directory if it doesn't exist
    os.makedirs(result_root, exist_ok=True)

    print(f"Running with model: {model}")
    print(f"API key from: {args.api_key_env}")
    print(f"Base URL: {base_url if base_url else 'provider default'}")
    print(f"Data root: {data_root}")
    print(f"Result root: {result_root}")
    print(f"Conflict types: {', '.join(args.conflict_types)}")

    # Pre-scan missing files
    print("\nScanning for missing files...")
    total_files_all = 0
    total_missing_all = 0

    dir_stats = {}

    for sub in args.conflict_types:
        target_dir = os.path.join(data_root, sub)
        if not os.path.exists(target_dir):
            continue

        t, m = count_missing_files(target_dir, data_root, result_root, model)
        dir_stats[sub] = (t, m)
        total_files_all += t
        total_missing_all += m
        print(f"  {sub}: {m}/{t} missing ({(m/t*100) if t>0 else 0:.1f}%)")

    print(f"\nTotal Progress: {total_files_all - total_missing_all}/{total_files_all} files complete")
    print(f"Missing to process: {total_missing_all}")

    if total_missing_all == 0:
        print("\nAll files are already processed. Exiting.")
        return

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
        print(f"Processing: {sub}")
        print("="*80)

        await process_directory(
            target_dir,
            data_root,
            result_root,
            model,
            api_key,
            base_url,
            desc=f"Processing {sub} (Missing: {missing_in_dir}/{total_in_dir})"
        )

    print("All files processed!")


if __name__ == "__main__":
    asyncio.run(main())
