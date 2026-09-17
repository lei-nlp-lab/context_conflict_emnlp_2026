"""
Batch Shapley-based evidence attribution over generated responses.

This is the driver behind the Evidence Balance metric of Section 3.1 of
"Large Language Models in Resolving Contextual Knowledge Conflicts" (EMNLP 2026).
For every response JSON below a model directory it

  1. reads the question, the evidence pieces and the model response,
  2. calls Evaluator.perplexity_evaluation (evaluation/evaluator.py), which computes
     one Shapley value per evidence piece using the length-normalized log-likelihood
     of the response under a frozen scorer (default meta-llama/Llama-3.2-1B-Instruct),
     clips negative values at zero and normalizes them to contribution shares, and
  3. writes the result back into the same JSON file under --output-key (default
     "eval") as {"shapley_values": [...], "prob_distribution": [...]}.

Files are updated in place under an exclusive file lock, and files that already
contain --output-key are skipped, so an interrupted run can simply be restarted.
The Balance score (Gini coefficient of prob_distribution) is computed afterwards by
evaluation/balance_score.py.

Expected layout (produced by generate_responses_api.py / generate_responses_local.py):
    <result-dir>/<model>/<conflict_type>/[<subfolder>/]<id>.json

Examples (from the repository root):
    # responses written by generate_responses_api.py --model gpt-5 (key "gpt-5_response")
    python evaluation/batch_evaluate.py --model gpt-5 --response-key "{model}_response" \
        --eval-methods perplexity
    # answer-only variant: score only the text inside <answer></answer>
    python evaluation/batch_evaluate.py --model gpt-5 --response-key "{model}_response" \
        --eval-methods perplexity --answer-only --output-key eval_answer_only
    # an arbitrary result directory whose JSON files store the output under "response"
    python evaluation/batch_evaluate.py --model results/responses/my_model
"""

import os
import json
import sys
import argparse
import fcntl
import time
from pathlib import Path
from typing import List, Dict, Optional

# Make the repository root importable so that this file also works when executed
# directly as "python evaluation/batch_evaluate.py" from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluator import Evaluator  # noqa: E402


# Frozen external scorer used for the value function in the paper.
DEFAULT_EVAL_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# Result sub-directories of the models evaluated in the paper -> Hugging Face id / provider id.
# The value is only used as the scorer when --eval-model is explicitly set to an empty string;
# by default every model directory is scored with DEFAULT_EVAL_MODEL.
MODELS_TO_PROCESS = {
    "claude-4.5-sonnet": "anthropic/claude-4.5-sonnet",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
    "gpt-5": "openai/gpt-5",
    "llama-3.1-70b-instruct": "meta-llama/Llama-3.1-70B-Instruct",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.3",
}


def extract_evidence_from_data(data: dict) -> List[str]:
    """Return the evidence pieces of a sample in a deterministic order.

    The released dataset stores evidence under data["content"] as a dict
    {"source_1": text, "source_2": text, ...}; keys are sorted. The "perspectives"
    branch handles a legacy article layout (Left/Center/Right lists) that is not
    used by the released data but is kept for compatibility with older result files.
    """
    evidence = []

    if 'perspectives' in data:
        perspectives = data['perspectives']

        for side in ['Left', 'Center', 'Right']:
            if side in perspectives and isinstance(perspectives[side], list):
                for article in perspectives[side]:
                    if isinstance(article, dict):
                        content = article.get('content', '').strip()
                        if content:
                            evidence.append(content)

    elif 'content' in data:
        content = data['content']

        for key in sorted(content.keys()):
            if isinstance(content[key], str) and content[key].strip():
                evidence.append(content[key].strip())

    return evidence


def extract_response_content(response: str, answer_only: bool = False) -> str:
    """Extract content within <answer></answer> tags if answer_only is True."""
    if not response:
        return response

    if answer_only:
        import re
        match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        print(f"  [WARN] No <answer> tags found, using full response")

    return response


def process_json_file(file_path: str, model: str, evaluator: Evaluator, output_key: str = 'eval', eval_methods: list = None, answer_only: bool = False, response_key_template: str = None) -> bool:
    """Evaluate one JSON file in place under an exclusive lock (with retries)."""
    max_retries = 5
    retry_delay = 1  # seconds

    for attempt in range(max_retries):
        try:
            # Open file with exclusive lock
            with open(file_path, 'r+', encoding='utf-8') as f:
                # Try to acquire exclusive lock (will block if another process has it)
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)

                try:
                    # Read current data
                    f.seek(0)
                    data = json.load(f)

                    # Check if already evaluated with this output key
                    if output_key in data:
                        print(f"  [SKIP] {os.path.basename(file_path)} - already evaluated (key: {output_key})")
                        return True

                    # Continue with evaluation...
                    return _do_evaluation(f, data, file_path, model, evaluator, output_key, eval_methods, answer_only, response_key_template)

                finally:
                    # Release lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        except BlockingIOError:
            if attempt < max_retries - 1:
                print(f"  [WAIT] {os.path.basename(file_path)} - file locked, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"  [ERROR] {os.path.basename(file_path)} - could not acquire lock after {max_retries} attempts")
                return False
        except Exception as e:
            print(f"  [ERROR] {os.path.basename(file_path)} - {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    return False


def _do_evaluation(f, data: dict, file_path: str, model: str, evaluator: Evaluator, output_key: str, eval_methods: list, answer_only: bool, response_key_template: str) -> bool:
    """Helper function to perform evaluation and write results."""
    try:

        question = data.get('question', '')

        # Resolve the key holding the model output. "{model}" in the template is replaced
        # by the model (directory) name, e.g. "{model}_response" -> "gpt-5_response".
        if response_key_template:
            response_key = response_key_template.replace('{model}', model)
        else:
            response_key = "response"

        raw_response = data.get(response_key, '')

        if not raw_response:
            print(f"  [WARN] {os.path.basename(file_path)} - no response found for key '{response_key}'")
            return False

        evidence = extract_evidence_from_data(data)

        if not evidence:
            print(f"  [WARN] {os.path.basename(file_path)} - no evidence found")
            return False

        response = extract_response_content(raw_response, answer_only=answer_only)

        if not response:
            print(f"  [WARN] {os.path.basename(file_path)} - response is empty after extraction")
            return False

        print(f"  [EVAL] {os.path.basename(file_path)} - {len(evidence)} evidence items")

        # Default to all methods if not specified
        if eval_methods is None:
            eval_methods = ['perplexity', 'atomic_claim', 'faithfulness']

        # Initialize results
        perplexity_results = None
        atomic_claim_results = None
        factscore = None

        # Run selected evaluation methods
        if 'perplexity' in eval_methods:
            print(f"  [EVAL] Running perplexity evaluation...")
            perplexity_results = evaluator.perplexity_evaluation(
                question=question,
                evidence=evidence,
                response=response,
                return_raw=True
            )

        if 'atomic_claim' in eval_methods:
            print(f"  [EVAL] Running atomic claim evaluation...")
            atomic_claim_results = evaluator.atomic_claim_evaluation(
                evidence=evidence,
                response=response
            )

        if 'faithfulness' in eval_methods:
            print(f"  [EVAL] Running faithfulness evaluation...")
            factscore = evaluator.faithfulness_evaluation(
                evidence=evidence,
                response=response
            )

        # Build evaluation result dictionary
        evaluation = {}

        # Add perplexity results if computed
        if perplexity_results is not None:
            num_clusters = len(perplexity_results['shapley_values'])

            shapley_values_list = []
            prob_distribution_list = []

            for i in range(num_clusters):
                shapley_values_list.append(perplexity_results['shapley_values'].get(i, 0.0))
                prob_distribution_list.append(perplexity_results['prob_distribution'].get(i, 0.0))

            evaluation['shapley_values'] = shapley_values_list
            evaluation['prob_distribution'] = prob_distribution_list

        # Add atomic claim results if computed
        if atomic_claim_results is not None:
            # Use num_clusters from perplexity if available, otherwise from atomic_claim
            if perplexity_results is not None:
                num_clusters = len(perplexity_results['shapley_values'])
            else:
                num_clusters = max(atomic_claim_results.keys()) + 1 if atomic_claim_results else len(evidence)

            atomic_claim = {}
            atomic_claim_percentage = {
                'support': [],
                'undermine': [],
                'net_contribution': [],
                'net_contribution_abs': []
            }

            for i in range(num_clusters):
                cluster_key = f"cluster_{i+1}"
                if i in atomic_claim_results:
                    atomic_claim[cluster_key] = {
                        'support': atomic_claim_results[i]['support'],
                        'undermine': atomic_claim_results[i]['undermine'],
                        'net_contribution': atomic_claim_results[i]['net_contribution'],
                        'net_contribution_abs': atomic_claim_results[i]['net_contribution_abs']
                    }
                    atomic_claim_percentage['support'].append(atomic_claim_results[i]['support'])
                    atomic_claim_percentage['undermine'].append(atomic_claim_results[i]['undermine'])
                    atomic_claim_percentage['net_contribution'].append(atomic_claim_results[i]['net_contribution'])
                    atomic_claim_percentage['net_contribution_abs'].append(atomic_claim_results[i]['net_contribution_abs'])
                else:
                    atomic_claim[cluster_key] = {
                        'support': 0.0,
                        'undermine': 0.0,
                        'net_contribution': 0.0,
                        'net_contribution_abs': 0.0
                    }
                    atomic_claim_percentage['support'].append(0.0)
                    atomic_claim_percentage['undermine'].append(0.0)
                    atomic_claim_percentage['net_contribution'].append(0.0)
                    atomic_claim_percentage['net_contribution_abs'].append(0.0)

            evaluation['atomic_claim'] = atomic_claim
            evaluation['atomic_claim_percentage'] = atomic_claim_percentage

        if factscore is not None:
            evaluation['factscore'] = factscore
        data[output_key] = evaluation

        f.seek(0)
        f.truncate()
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.flush()
        os.fsync(f.fileno())

        print(f"  [DONE] {os.path.basename(file_path)} - evaluation saved to key '{output_key}'")
        if 'shapley_values' in evaluation:
            print(f"    - Shapley values clusters: {len(evaluation['shapley_values'])}")
        if 'atomic_claim' in evaluation:
            print(f"    - Atomic claim clusters: {len(evaluation['atomic_claim'])}")
        if 'factscore' in evaluation:
            print(f"    - FactScore: {evaluation['factscore']:.4f}")
        return True

    except Exception as e:
        print(f"  [ERROR] {os.path.basename(file_path)} - {e}")
        import traceback
        traceback.print_exc()
        return False


def find_all_json_files(root_dir: str) -> List[str]:
    """Find all JSON files in directory and subdirectories."""
    json_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files


def process_model_directory(model_result_name: str, model_path: str, result_root: str, eval_model: Optional[str] = None, output_key: str = 'eval', eval_methods: list = None, answer_only: bool = False, response_key_template: str = None):
    """Load the scorer once and evaluate every JSON file below one model directory."""

    if os.path.isabs(model_path) and os.path.exists(model_path):
        model_dir = model_path
    else:
        model_dir = os.path.join(result_root, model_result_name)

    if not os.path.exists(model_dir):
        print(f"\n[SKIP] Model directory not found: {model_dir}")
        return

    print(f"\n{'='*80}")
    print(f"Processing model: {model_result_name}")
    print(f"Model path (results): {model_path}")
    print(f"{'='*80}")

    json_files = find_all_json_files(model_dir)
    print(f"Found {len(json_files)} JSON files")

    if not json_files:
        print(f"No JSON files found in {model_dir}")
        return

    evaluator_model = eval_model if eval_model else model_path
    print(f"\n[Evaluator] Initializing local evaluator")
    print(f"  Results from: {model_result_name}")
    print(f"  Evaluating with: {evaluator_model}")

    # The NLI model is only needed by the atomic-claim and FActScore-style checks;
    # skip loading it when only the Shapley attribution (perplexity) is requested.
    needs_nli = eval_methods is None or any(m in eval_methods for m in ('atomic_claim', 'faithfulness'))

    evaluator = Evaluator(
        model=evaluator_model,
        nli_model="roberta-large-mnli" if needs_nli else None,
        helper_model=None,
        enable_clustering=False,
        use_api=False
    )

    print(f"  [INFO] Local evaluator ready: clustering OFF")

    success_count = 0
    skip_count = 0
    error_count = 0

    for i, file_path in enumerate(json_files, 1):
        print(f"\n[{i}/{len(json_files)}] Processing: {os.path.relpath(file_path, model_dir)}")

        result = process_json_file(file_path, model_result_name, evaluator, output_key=output_key, eval_methods=eval_methods, answer_only=answer_only, response_key_template=response_key_template)

        if result:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if output_key in data:
                    success_count += 1
                else:
                    skip_count += 1
        else:
            error_count += 1

    print(f"\n{'='*80}")
    print(f"Model {model_result_name} completed:")
    print(f"  Success: {success_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Errors:  {error_count}")
    print(f"{'='*80}")


def main():
    """Main function with command line argument support."""

    parser = argparse.ArgumentParser(
        description='Batch Shapley-based evidence attribution (Evidence Balance) over generated responses',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (run from the repository root):

  # all model directories listed in MODELS_TO_PROCESS under results/responses
  python evaluation/batch_evaluate.py --response-key "{model}_response" --eval-methods perplexity

  # one model directory; responses were written by generate_responses_*.py under "<model>_response"
  python evaluation/batch_evaluate.py --model llama-3.1-8b-instruct --response-key "{model}_response"

  # an arbitrary directory of response JSON files (relative to the repository root or absolute)
  python evaluation/batch_evaluate.py --model results/responses/my_model --response-key response
        """
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Model directory to process: a name under --result-dir (e.g. llama-3.1-8b-instruct) or a path to a directory of response JSON files. If not specified, all models in MODELS_TO_PROCESS are processed.'
    )
    parser.add_argument(
        '--eval-model',
        type=str,
        default=DEFAULT_EVAL_MODEL,
        help=f'Hugging Face causal LM used as the frozen scorer for the value function (default: {DEFAULT_EVAL_MODEL}). The paper also reports meta-llama/Llama-3.1-8B-Instruct and google/gemma-2b for robustness.'
    )
    parser.add_argument(
        '--output-key',
        type=str,
        default='eval',
        help='Key under which the evaluation results are stored in each JSON file (default: eval). Files that already contain this key are skipped.'
    )
    parser.add_argument(
        '--eval-methods',
        type=str,
        nargs='+',
        choices=['perplexity', 'atomic_claim', 'faithfulness'],
        default=None,
        help='Evaluation methods to run. "perplexity" is the Shapley attribution used for Evidence Balance in the paper. If not specified, all methods will be run (this also loads roberta-large-mnli).'
    )
    parser.add_argument(
        '--answer-only',
        action='store_true',
        help='Extract only content within <answer></answer> tags from model response for evaluation'
    )
    parser.add_argument(
        '--response-key',
        type=str,
        default='response',
        help='Key (or template) holding the model output in each JSON file. Use {model} as a placeholder for the model directory name, e.g. "{model}_response" for files written by generate_responses_api.py / generate_responses_local.py. Default: response'
    )
    parser.add_argument(
        '--result-dir',
        type=str,
        default='results/responses',
        help='Directory containing per-model sub-directories of response JSON files (files are modified in place). Relative paths are resolved from the repository root. Default: results/responses'
    )
    args = parser.parse_args()

    # Hugging Face credentials and cache location are taken from the environment
    # (HF_TOKEN, HF_HOME) if set; nothing is written to os.environ here.
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        print(f"[Setup] HuggingFace cache directory (HF_HOME): {hf_home}")
    if not (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")):
        print("[Setup] HF_TOKEN not set; gated scorer models (e.g. meta-llama/*) may fail to download")

    result_root = args.result_dir
    if not os.path.isabs(result_root):
        result_root = os.path.join(str(REPO_ROOT), result_root)
    print(f"[Setup] Result files will be read from and modified in-place: {result_root}")

    if not os.path.exists(result_root):
        print(f"Error: Result directory not found: {result_root}")
        sys.exit(1)

    if args.model:
        # Accept, in this order: an absolute path, a path relative to the repository
        # root, or the name of a sub-directory of the result directory.
        candidates = []
        if os.path.isabs(args.model):
            candidates.append(args.model)
        else:
            candidates.append(os.path.join(str(REPO_ROOT), args.model))
            candidates.append(os.path.join(result_root, args.model))
        model_path_check = next((c for c in candidates if os.path.isdir(c)), None)

        if model_path_check is not None:
            model_name = os.path.basename(model_path_check.rstrip('/'))
            models_to_run = {model_name: model_path_check}
            print(f"[Mode] Direct path evaluation: {model_path_check}")
        elif args.model in MODELS_TO_PROCESS:
            models_to_run = {args.model: MODELS_TO_PROCESS[args.model]}
            print(f"[Mode] Single model evaluation: {args.model}")
        else:
            print(f"Error: Model '{args.model}' not found in MODELS_TO_PROCESS and is not a valid directory")
            print(f"  Checked paths: {', '.join(candidates)}")
            print(f"Available models: {', '.join(MODELS_TO_PROCESS.keys())}")
            sys.exit(1)
    else:
        models_to_run = MODELS_TO_PROCESS
        print(f"[Mode] Batch evaluation: all models")

    print("="*80)
    print("Batch Shapley Evidence Attribution (local scorer)")
    print("="*80)
    print(f"Result directory: {result_root}")
    print(f"HuggingFace cache: {hf_home if hf_home else 'default (~/.cache/huggingface)'}")
    print(f"Models to process: {len(models_to_run)}")
    if args.eval_model:
        print(f"Evaluation model (fixed): {args.eval_model}")
    else:
        print(f"Evaluation model: same as result model")
    print(f"Output key: {args.output_key}")
    if args.eval_methods:
        print(f"Evaluation methods: {', '.join(args.eval_methods)}")
    else:
        print(f"Evaluation methods: all (perplexity, atomic_claim, faithfulness)")
    print(f"Answer only mode: {'ON (extract <answer> tags only)' if args.answer_only else 'OFF (use full response)'}")
    print(f"Response key template: {args.response_key}")
    for model_name, model_path in models_to_run.items():
        status = "SKIP (API-only)" if model_path is None else model_path
        print(f"  - {model_name} -> {status}")

    for model_result_name, model_path in models_to_run.items():

        if model_path is None:
            print(f"\n[SKIP] {model_result_name} - API-only model, not available locally")
            continue

        try:
            process_model_directory(
                model_result_name,
                model_path,
                result_root,
                eval_model=args.eval_model,
                output_key=args.output_key,
                eval_methods=args.eval_methods,
                answer_only=args.answer_only,
                response_key_template=args.response_key
            )
        except Exception as e:
            print(f"\n[FATAL ERROR] Failed to process model {model_result_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*80)
    print("All models processed!")
    print("="*80)


if __name__ == "__main__":
    main()
