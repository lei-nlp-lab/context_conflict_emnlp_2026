"""
Faithfulness metric (RAGAS) for all six ContextConflict conflict types.

Section 3.1 of "Large Language Models in Resolving Contextual Knowledge Conflicts"
(EMNLP 2026) reports Faithfulness for every conflict type: the fraction of claims in
a model response that are supported by the evidence it was given, computed with the
RAGAS `faithfulness` metric (claim decomposition followed by per-claim verification
by a judge LLM). The judge is an OpenAI-compatible chat model, by default DeepSeek
`deepseek-chat` via langchain_openai.ChatOpenAI with temperature 0 and max_tokens 2048.

For each response JSON the script builds one RAGAS record (question, answer = the
model response, contexts = the evidence pieces taken from `content` or, for
perspective data, `perspectives`), runs the metric, writes the per-sample score back
into the JSON under "faithfulness" (in place; files that already hold a valid score
are skipped, so interrupted runs resume), and records mean/std per
(model, conflict type) in a TSV file.

Expected layout: <result-dir>/<model>/<conflict_type>/[<subfolder>/]<id>.json
--result-dir may also point directly at one model directory that contains the
conflict-type sub-directories.

Modes:
    sequential  (default) evaluate every (model, conflict type) pair in one process
                and write --output from scratch
    single      evaluate one pair (--model + --conflict, or --task-id from the task
                table) and append one line to --output; meant for job arrays
    list        print the task table (task_id, model, model_path, conflict_type)

Examples (from the repository root):
    export DEEPSEEK_API_KEY=...
    python evaluation/faithfulness_ragas.py --result-dir results/responses
    python evaluation/faithfulness_ragas.py --result-dir results/responses --mode list
    python evaluation/faithfulness_ragas.py --result-dir results/responses --mode single --task-id 3
    python evaluation/faithfulness_ragas.py --result-dir results/responses/gpt-5 \
        --mode single --conflict temporal_conflict
    # a different OpenAI-compatible judge
    python evaluation/faithfulness_ragas.py --result-dir results/responses \
        --judge-model gpt-4o --base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY

Environment variables: DEEPSEEK_API_KEY (or the variable named by --api-key-env);
DEEPSEEK_BASE_URL optionally overrides the default endpoint https://api.deepseek.com.
"""

import os
import json
import glob
import argparse
from pathlib import Path

import numpy as np

from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness
from ragas.llms import LangchainLLMWrapper

from langchain_openai import ChatOpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]

# ============================================================
# Judge LLM configuration (OpenAI-compatible endpoint, DeepSeek by default)
# ============================================================

DEFAULT_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEFAULT_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEFAULT_JUDGE_MODEL = "deepseek-chat"

_llm_instance = None


def get_llm(judge_model=DEFAULT_JUDGE_MODEL, base_url=DEFAULT_BASE_URL, api_key_env=DEFAULT_API_KEY_ENV):
    """Create (once) the RAGAS-wrapped judge LLM. The API key is read from the environment only."""
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    api_key = os.environ.get(api_key_env, "")
    if not api_key or api_key.strip() == "":
        raise ValueError(
            f"[ERROR] {api_key_env} is not set. Export it before running, e.g. "
            f"export {api_key_env}=<your key>"
        )

    print(f"[INFO] Initializing judge LLM")
    print(f"[INFO] Model: {judge_model}")
    print(f"[INFO] Base URL: {base_url}")
    print(f"[INFO] API key: read from ${api_key_env}")

    # Note: no global response_format; RAGAS handles JSON parsing internally and its
    # prompts already ask for JSON output.
    chat_llm = ChatOpenAI(
        model=judge_model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,  # Deterministic output for consistent JSON
        max_tokens=2048,  # Increased for complex JSON responses
    )

    _llm_instance = LangchainLLMWrapper(chat_llm)

    print(f"[INFO] Judge LLM initialized successfully")
    return _llm_instance


# ============================================================
# Configuration
# ============================================================

DEFAULT_MODELS = [
    "gpt-5",
    "claude-4.5-sonnet",
    "gemini-2.5-pro",
    "gpt-oss-120b",
    "gpt-oss-20b",
    "llama-3.1-70b-instruct",
    "llama-3.1-8b-instruct",
]

ALL_CONFLICT_TYPES = [
    "inferential_conflict",
    "misinformation_conflict",
    "temporal_conflict",
    "ambiguity_conflict",
    "granularity_conflict",
    "perspective_conflict",
]

DEFAULT_OUTPUT_PATH = "results/evaluation/faithfulness.tsv"


def resolve_path(path):
    """Return `path` unchanged if absolute, otherwise relative to the repository root."""
    return path if os.path.isabs(path) else os.path.join(str(REPO_ROOT), path)


def resolve_model_dirs(result_dir, models, conflict_types):
    """
    Return the list of (model_name, model_path) pairs to evaluate.

    - `models` given: each is a sub-directory of `result_dir`.
    - otherwise, if `result_dir` itself contains a conflict-type directory it is treated
      as a single model directory (named after its basename);
    - otherwise the seven default models under `result_dir` are used.
    """
    if models:
        return [(m, os.path.join(result_dir, m)) for m in models]
    if any(os.path.isdir(os.path.join(result_dir, ct)) for ct in conflict_types):
        return [(os.path.basename(os.path.normpath(result_dir)), result_dir)]
    return [(m, os.path.join(result_dir, m)) for m in DEFAULT_MODELS]


# ============================================================
# Utility Functions
# ============================================================

def find_all_json_recursive(root_dir: str):
    """
    Recursively find all .json files under the given directory.
    If the directory does not exist, return an empty list.
    """
    if not os.path.exists(root_dir):
        return []

    pattern = os.path.join(root_dir, "**", "*.json")
    files = glob.glob(pattern, recursive=True)
    return sorted(files)


def extract_contexts_from_data(data):
    """
    Extract contexts (evidence) from JSON data.
    Supports multiple formats:
    1. content: dict/list - standard format
    2. perspectives: dict with Left/Center/Right - perspective_conflict format
    """
    if "content" in data:
        content = data["content"]
        if isinstance(content, dict):
            return [str(v) for v in content.values()]
        elif isinstance(content, list):
            return [str(v) for v in content]

    if "perspectives" in data:
        perspectives = data["perspectives"]
        contexts = []
        if isinstance(perspectives, dict):
            for side, items in perspectives.items():
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict) and "content" in item:
                            contexts.append(str(item["content"]))
                        elif isinstance(item, str):
                            contexts.append(item)
        if contexts:
            return contexts

    return None


def is_null_value(value):
    """
    Check if a value is null/None/NaN, including string representations.
    Handles: None, "null", "NULL", "none", "nan", "NaN", "", float NaN, etc.
    """
    if value is None:
        return True
    # float NaN check
    if isinstance(value, (float, int)):
        try:
            if np.isnan(float(value)):
                return True
        except (ValueError, TypeError):
            pass
    # numpy floating NaN
    if isinstance(value, np.floating) and np.isnan(value):
        return True
    # string null-ish
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("null", "none", "", "nan"):
            return True
    return False


def check_faithfulness_exists(file_path):
    """
    Check if a JSON file already has a valid (non-null, non-NaN) faithfulness attribute.
    Returns True if already evaluated with valid score (skip).
    Returns False if needs evaluation (no attribute, null, NaN, or invalid).
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return False

        if "faithfulness" not in data:
            return False

        val = data["faithfulness"]

        # Check for null/None/NaN string values
        if is_null_value(val):
            return False

        # Verify it can be converted to a valid float
        try:
            s = float(val)
            if np.isnan(s):
                return False
            return True
        except (ValueError, TypeError):
            return False

    except Exception:
        return False


def filter_unevaluated_files(json_files):
    """
    Filter out files that have not been evaluated for faithfulness.
    Returns list of files to evaluate and count of skipped files.

    Files are re-evaluated if:
    - No faithfulness attribute
    - faithfulness is null/None (previous evaluation failed)
    """
    unevaluated = []
    skipped = 0
    null_count = 0

    for path in json_files:
        if check_faithfulness_exists(path):
            skipped += 1
        else:
            # Check if it has null faithfulness (needs re-evaluation)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "faithfulness" in data and is_null_value(data["faithfulness"]):
                    null_count += 1
            except Exception:
                pass
            unevaluated.append(path)

    if null_count > 0:
        print(f"[INFO] Found {null_count} files with null faithfulness (will re-evaluate)")

    return unevaluated, skipped


def build_ragas_dataset_from_json_files(json_files, response_key="response"):
    """
    Convert a list of JSON files into a RAGAS-compatible HuggingFace Dataset.
    Each JSON must contain:
        - question: str
        - <response_key>: str (the model's output, default key "response")
        - content: dict/list OR perspectives: dict (for perspective_conflict)

    Output fields for RAGAS:
        - question
        - answer
        - contexts (list of evidences)
    """
    questions = []
    answers = []
    contexts_list = []
    valid_paths = []

    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load JSON, skipping: {path} | error: {e}")
            continue

        if not isinstance(data, dict):
            print(f"[WARN] JSON root is not a dict, skipping: {path}")
            continue

        if "question" not in data or response_key not in data:
            print(f"[WARN] Missing question/{response_key} fields, skipping: {path}")
            continue

        question = data["question"]
        answer = data[response_key]

        contexts = extract_contexts_from_data(data)
        if contexts is None or len(contexts) == 0:
            print(f"[WARN] No valid contexts found, skipping: {path}")
            continue

        if not question or not answer:
            print(f"[WARN] Empty question/answer, skipping: {path}")
            continue

        questions.append(str(question))
        answers.append(str(answer))
        contexts_list.append(contexts)
        valid_paths.append(path)

    if len(questions) == 0:
        return None, []

    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
        }
    )
    return ds, valid_paths


def save_faithfulness_to_json(file_path, score):
    """
    Save faithfulness score to a JSON file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["faithfulness"] = score if not np.isnan(score) else None

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return True
    except Exception as e:
        print(f"[WARN] Failed to save faithfulness to {file_path}: {e}")
        return False


def get_existing_faithfulness_scores(json_files):
    """
    Get existing faithfulness scores from JSON files.
    Returns (scores list, paths list) containing only files with valid (non-NaN) scores.
    """
    scores = []
    paths = []

    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            val = data.get("faithfulness", None) if isinstance(data, dict) else None

            # Skip null/None/NaN values
            if is_null_value(val):
                continue

            # Try to convert to float and check for NaN
            s = float(val)
            if np.isnan(s):
                continue

            scores.append(s)
            paths.append(path)
        except Exception:
            continue

    return scores, paths


def compute_faithfulness_stats_for_files(json_files, llm=None, save_to_json=True, response_key="response"):
    """
    Given a list of JSON files:
        1. Filter out already evaluated files (auto-resume)
        2. Build a RAGAS dataset for unevaluated files
        3. Evaluate with the RAGAS faithfulness metric
        4. Save faithfulness scores back to JSON files
        5. Compute mean and std (including previously evaluated files)

    Returns:
        (mean, std, n_samples)
    If no valid samples exist, returns (None, None, 0).
    """
    unevaluated_files, skipped_count = filter_unevaluated_files(json_files)

    if skipped_count > 0:
        print(f"[INFO] Auto-resume: Skipping {skipped_count} already evaluated files")

    existing_scores, _ = get_existing_faithfulness_scores(json_files)

    new_scores = []

    if len(unevaluated_files) > 0:
        result_data = build_ragas_dataset_from_json_files(unevaluated_files, response_key=response_key)
        if result_data is not None and result_data[0] is not None:
            ds, valid_paths = result_data

            if len(ds) > 0:
                print(f"[INFO] Evaluating {len(ds)} new samples")

                if llm is not None:
                    faithfulness.llm = llm

                # RAGAS run configuration (kept as used for the paper):
                # timeout: max time per sample (seconds)
                # max_workers: sequential processing for stability
                # max_retries: retry on failure
                run_config = RunConfig(
                    timeout=600,      # 10 minutes per sample
                    max_workers=1,    # Sequential processing for stability
                    max_retries=2,    # Retry on failure
                    max_wait=300,     # Max wait between retries
                )
                print(f"[INFO] RunConfig: timeout=600s, max_workers=1, max_retries=2")

                result = evaluate(
                    ds,
                    metrics=[faithfulness],
                    llm=llm,
                    raise_exceptions=False,
                    run_config=run_config
                )

                scores = None
                try:
                    df = result.to_pandas()
                    col_candidates = [c for c in df.columns if "faithfulness" in c.lower()]
                    if not col_candidates:
                        raise RuntimeError(
                            f"No faithfulness column found in output. Columns: {df.columns}"
                        )
                    col = col_candidates[0]
                    scores = df[col].astype(float).to_numpy()
                except Exception:
                    if isinstance(result, dict) and "faithfulness" in result:
                        scores = np.array(result["faithfulness"], dtype=float).reshape(-1)
                    else:
                        raise

                if save_to_json and len(valid_paths) == len(scores):
                    saved_count = 0
                    for path, score in zip(valid_paths, scores):
                        if save_faithfulness_to_json(path, float(score)):
                            saved_count += 1
                    print(f"[INFO] Saved faithfulness scores to {saved_count}/{len(valid_paths)} JSON files")

                for s in scores:
                    if not np.isnan(s):
                        new_scores.append(float(s))
    else:
        print(f"[INFO] All files already evaluated, using cached scores")

    all_scores = existing_scores + new_scores

    if len(all_scores) == 0:
        print(f"[WARN] No valid scores found")
        return None, None, 0

    all_scores_arr = np.array(all_scores)
    print(f"[INFO] Total valid scores: {len(all_scores)} (existing: {len(existing_scores)}, new: {len(new_scores)})")

    return float(np.mean(all_scores_arr)), float(np.std(all_scores_arr)), len(all_scores)


TSV_HEADER = "model\tmodel_path\tconflict_type\tn_samples\tfaithfulness_mean\tfaithfulness_std\n"


def format_row(model, model_path, conflict_type, n_samples, mean, std):
    mean_str = f"{mean:.4f}" if mean is not None else "N/A"
    std_str = f"{std:.4f}" if std is not None else "N/A"
    return f"{model}\t{model_path}\t{conflict_type}\t{n_samples}\t{mean_str}\t{std_str}\n"


def save_single_result(output_path, model, model_path, conflict_type, n_samples, mean, std):
    """
    Append a single evaluation result to a TSV file (append mode with file locking).
    """
    import fcntl

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    line = format_row(model, model_path, conflict_type, n_samples, mean, std)

    # Use file locking to prevent race conditions when multiple jobs write simultaneously
    with open(output_path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            # Check if file is empty and write header
            f.seek(0, 2)  # Go to end
            if f.tell() == 0:
                f.write(TSV_HEADER)
            f.write(line)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    print(f"[INFO] Result saved to {output_path}")


# ============================================================
# Single Task Evaluation (for parallel execution)
# ============================================================

def evaluate_single_task(model, model_path, conflict_type, output_path, llm_kwargs=None, response_key="response"):
    """
    Evaluate a single model + conflict_type combination.
    This function is designed to be called by parallel (e.g. SLURM array) jobs.

    If all files are already evaluated, it will skip LLM initialization
    and directly output the cached results.
    """
    llm_kwargs = llm_kwargs or {}

    print(f"===== Single Task Evaluation =====")
    print(f"[INFO] Model: {model}")
    print(f"[INFO] Model Path: {model_path}")
    print(f"[INFO] Conflict Type: {conflict_type}")
    print(f"[INFO] Output Path: {output_path}")

    root_dir = os.path.join(model_path, conflict_type)

    json_files = find_all_json_recursive(root_dir)

    if not json_files:
        print(f"[WARN] No JSON files found in: {root_dir}")
        return

    print(f"[INFO] Found {len(json_files)} JSON files in {root_dir}")

    # Check if all files are already evaluated
    unevaluated_files, skipped_count = filter_unevaluated_files(json_files)

    if len(unevaluated_files) == 0:
        # All files already evaluated - no need to initialize LLM
        print(f"[INFO] All {skipped_count} files already evaluated, using cached scores (skipping LLM init)")
        existing_scores, _ = get_existing_faithfulness_scores(json_files)

        if len(existing_scores) == 0:
            print(f"[WARN] No valid scores found in evaluated files")
            return

        all_scores_arr = np.array(existing_scores)
        mean = float(np.mean(all_scores_arr))
        std = float(np.std(all_scores_arr))
        n = len(existing_scores)

        print(f"[INFO] Total valid scores: {n}")
    else:
        # Need to evaluate new files - initialize LLM
        print(f"[INFO] {len(unevaluated_files)} files need evaluation, {skipped_count} already done")
        llm = get_llm(**llm_kwargs)
        print(f"[INFO] LLM initialized successfully")

        # Compute faithfulness (will handle both existing and new files)
        mean, std, n = compute_faithfulness_stats_for_files(json_files, llm=llm, response_key=response_key.replace("{model}", os.path.basename(model.rstrip("/"))))

        if n == 0 or mean is None:
            print(f"[WARN] No valid samples for evaluation")
            return

    # Save result
    save_single_result(output_path, model, model_path, conflict_type, n, mean, std)

    print(f"[INFO] Evaluation complete: mean={mean:.4f}, std={std:.4f}, n={n}")


# ============================================================
# Task table (for job arrays)
# ============================================================

def list_all_tasks(model_dirs, conflict_types):
    """
    List all model + conflict_type combinations as tasks.
    Output format: task_id, model, model_path, conflict_type
    """
    tasks = []
    task_id = 0

    for model, model_path in model_dirs:
        for conflict in conflict_types:
            tasks.append({
                "task_id": task_id,
                "model": model,
                "model_path": model_path,
                "conflict_type": conflict,
            })
            task_id += 1

    return tasks


def print_tasks(model_dirs, conflict_types):
    """
    Print all tasks in TSV format for debugging or job submission.
    """
    tasks = list_all_tasks(model_dirs, conflict_types)
    print(f"# Total tasks: {len(tasks)}")
    print("task_id\tmodel\tmodel_path\tconflict_type")
    for t in tasks:
        print(f"{t['task_id']}\t{t['model']}\t{t['model_path']}\t{t['conflict_type']}")


# ============================================================
# Main Evaluation Loop (sequential mode)
# ============================================================

def main_sequential(model_dirs, conflict_types, output_path, llm_kwargs=None, response_key="response"):
    """Evaluate every (model, conflict type) pair in one process and write the TSV from scratch."""
    llm_kwargs = llm_kwargs or {}
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"===== Initializing judge LLM =====")
    llm = get_llm(**llm_kwargs)
    print(f"[INFO] LLM initialized successfully")

    rows = []

    print(f"===== Evaluating {len(model_dirs)} model directories =====")
    for model, model_path in model_dirs:
        for conflict in conflict_types:
            root_dir = os.path.join(model_path, conflict)
            json_files = find_all_json_recursive(root_dir)

            if not json_files:
                print(f"[WARN] No JSON files found: model={model}, conflict={conflict}")
                continue

            print(f"[INFO] model={model}, conflict={conflict}, #files={len(json_files)}")

            mean, std, n = compute_faithfulness_stats_for_files(json_files, llm=llm, response_key=response_key.replace("{model}", os.path.basename(model.rstrip("/"))))
            if n == 0 or mean is None:
                print(f"[WARN] No valid samples for: model={model}, conflict={conflict}")
                continue

            rows.append({
                "model": model,
                "model_path": model_path,
                "conflict_type": conflict,
                "n_samples": n,
                "faithfulness_mean": mean,
                "faithfulness_std": std,
            })

    print(f"===== Writing results to {output_path} =====")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(TSV_HEADER)
        for row in rows:
            f.write(format_row(row['model'], row['model_path'], row['conflict_type'],
                               row['n_samples'], row['faithfulness_mean'], row['faithfulness_std']))

    print("===== Finished! =====")


# ============================================================
# Main Entry Point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="RAGAS faithfulness of model responses against the provided evidence "
                    "(scores are written back into each JSON under 'faithfulness').")
    parser.add_argument("--result-dir", type=str, required=True,
                        help="Directory containing <model>/<conflict_type>/... response JSON files "
                             "(e.g. results/responses), or a single model directory that directly contains "
                             "the conflict-type sub-directories. Relative paths are resolved from the repository root.")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Model sub-directories of --result-dir to evaluate. Default: the seven models of "
                             "the paper, or --result-dir itself when it is a model directory.")
    parser.add_argument("--conflict-types", type=str, nargs="+", default=ALL_CONFLICT_TYPES,
                        choices=ALL_CONFLICT_TYPES,
                        help="Conflict types to evaluate. Default: all six.")
    parser.add_argument("--response-key", type=str, default="response",
                        help="JSON key holding the model output; '{model}' is expanded to the model directory name "
                             "(generate_responses_*.py write '<model>_response'). Default: response")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH,
                        help=f"TSV summary file. Default: {DEFAULT_OUTPUT_PATH}")
    parser.add_argument("--mode", type=str, default="sequential",
                        choices=["sequential", "single", "list"],
                        help="sequential: all pairs in one process; single: one (model, conflict) pair; "
                             "list: print the task table")

    # Arguments for single task mode
    parser.add_argument("--task-id", type=int,
                        help="single mode: 0-based index into the task table printed by --mode list")
    parser.add_argument("--model", type=str,
                        help="single mode: model sub-directory of --result-dir (omit when --result-dir is a model directory)")
    parser.add_argument("--conflict", type=str, choices=ALL_CONFLICT_TYPES,
                        help="single mode: conflict type to evaluate")

    # Judge LLM
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL,
                        help=f"Chat model used as RAGAS judge. Default: {DEFAULT_JUDGE_MODEL}")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL,
                        help="OpenAI-compatible endpoint. Default: $DEEPSEEK_BASE_URL or https://api.deepseek.com")
    parser.add_argument("--api-key-env", type=str, default=DEFAULT_API_KEY_ENV,
                        help=f"Environment variable holding the judge API key. Default: {DEFAULT_API_KEY_ENV}")
    return parser.parse_args()


def main():
    args = parse_args()

    result_dir = resolve_path(args.result_dir)
    output_path = resolve_path(args.output)
    conflict_types = list(args.conflict_types)
    model_dirs = resolve_model_dirs(result_dir, args.models, conflict_types)
    llm_kwargs = {
        "judge_model": args.judge_model,
        "base_url": args.base_url,
        "api_key_env": args.api_key_env,
    }

    if args.mode == "list":
        print_tasks(model_dirs, conflict_types)

    elif args.mode == "single":
        # Single task mode: select by task_id OR by explicit parameters
        if args.task_id is not None:
            tasks = list_all_tasks(model_dirs, conflict_types)
            if args.task_id < 0 or args.task_id >= len(tasks):
                raise ValueError(f"Invalid task_id: {args.task_id}. Valid range: 0-{len(tasks)-1}")
            task = tasks[args.task_id]
            evaluate_single_task(
                model=task["model"],
                model_path=task["model_path"],
                conflict_type=task["conflict_type"],
                output_path=output_path,
                llm_kwargs=llm_kwargs,
                response_key=args.response_key,
            )
        elif args.conflict:
            if args.model:
                model, model_path = args.model, os.path.join(result_dir, args.model)
            elif len(model_dirs) == 1:
                model, model_path = model_dirs[0]
            else:
                raise ValueError(
                    "For single mode with several candidate models, provide --model (or use --task-id)"
                )
            evaluate_single_task(
                model=model,
                model_path=model_path,
                conflict_type=args.conflict,
                output_path=output_path,
                llm_kwargs=llm_kwargs,
                response_key=args.response_key,
            )
        else:
            raise ValueError(
                "For single mode, either provide --task-id OR --conflict (plus --model unless "
                "--result-dir is a single model directory)"
            )

    else:
        main_sequential(model_dirs, conflict_types, output_path,
                        llm_kwargs=llm_kwargs, response_key=args.response_key)


if __name__ == "__main__":
    main()
