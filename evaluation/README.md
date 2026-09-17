# Evaluation

Response generation and the three metrics reported in Section 3 of the paper: Accuracy on the reasoning conflict types, Evidence Balance on the summarization types, and Faithfulness on all types.

## Files

| File | Purpose |
|---|---|
| `generate_responses_api.py` | Generates answers with an OpenAI-compatible API model |
| `generate_responses_local.py` | Generates answers with a local Hugging Face model, greedy decoding |
| `evaluator.py` | Core scoring library: Shapley-based evidence attribution, perplexity, NLI-based atomic claim checks, faithfulness |
| `batch_evaluate.py` | Applies `evaluator.py` to every response JSON in a directory, writing results back in place |
| `accuracy.py` | Aggregates Accuracy over the reasoning conflict types |
| `balance_score.py` | Aggregates Evidence Balance (normalized Gini) plus JS and KL divergence to uniform |
| `faithfulness_ragas.py` | Faithfulness via RAGAS with a DeepSeek judge |
| `scripts/` | SLURM and bash templates for the steps above |

## Result layout

Generation writes one JSON per sample, mirroring the dataset tree:

```
results/responses/<model>/<conflict_type>/[<sub-source>/]<id>.json
```

Each file keeps the original sample fields (`question`, `content.source_k`, `answer`) and adds the model answer under `<model>_response`. `batch_evaluate.py` then adds an evaluation dict under the key given by `--output-key` (default `eval`), containing `shapley_values` and `prob_distribution`.

## Metrics

**Evidence Balance.** Each evidence piece receives a Shapley value whose value function is the length-normalized log-likelihood of the response under a frozen external scorer (default `meta-llama/Llama-3.2-1B-Instruct`). Negative contributions are clipped, the rest are normalized into a share distribution, and Balance is its normalized Gini coefficient. Lower is more balanced. Implemented in `Evaluator.perplexity_evaluation`.

**Accuracy.** Proportion of answers matching the gold label, using exact match with n-gram and BLEU fallbacks for free-form answers.

**Faithfulness.** Proportion of atomic claims in the response that the provided evidence supports, computed with RAGAS. It measures grounding, not factual correctness.

## Running

Generation:

```bash
OPENAI_API_KEY=... python evaluation/generate_responses_api.py --model gpt-5

python evaluation/generate_responses_local.py \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --model-name llama-3.1-8b-instruct \
    --max-new-tokens 512
```

Both accept `--data-root` (default `ContextConflict_Dataset/data`), `--result-root` (default `results/responses`) and `--conflict-types`. Both skip samples that already have an answer, so an interrupted run can be resumed by re-running the same command.

Attribution:

```bash
python evaluation/batch_evaluate.py \
    --model llama-3.1-8b-instruct \
    --result-dir results/responses \
    --response-key "{model}_response" \
    --eval-model meta-llama/Llama-3.2-1B-Instruct \
    --output-key eval \
    --eval-methods perplexity \
    --answer-only
```

`--response-key` must match the key the generation step wrote; `{model}` is substituted with the value of `--model`. `--answer-only` scores just the text inside `<answer></answer>` tags, which is the setting used for the numbers in the paper. Omitting `--model` processes every model in `MODELS_TO_PROCESS`. Files are updated in place under a file lock, so the command is safe to re-run and to shard across jobs.

Metrics:

```bash
python evaluation/accuracy.py --result-dir results/responses \
    --response-key "{model}_response" --csv

python evaluation/balance_score.py --result-dir results/responses --eval-key eval

DEEPSEEK_API_KEY=... python evaluation/faithfulness_ragas.py --result-dir results/responses
```

`accuracy.py` defaults to the three reasoning conflict types and `balance_score.py` to the three summarization types; override with `--categories` and `--conflict-types`. Outputs go to `results/evaluation/`.

## Environment variables

| Variable | Used by |
|---|---|
| `OPENAI_API_KEY` | `generate_responses_api.py`; the flag `--api-key-env` selects a different variable name |
| `OPENAI_BASE_URL` | Optional, for OpenAI-compatible endpoints |
| `DEEPSEEK_API_KEY` | `faithfulness_ragas.py`; override the endpoint with `--base-url` and the judge with `--judge-model` |
| `HF_TOKEN` | Gated models in `generate_responses_local.py` and `batch_evaluate.py` |
| `HF_HOME` | Optional cache location; respected if already set |
