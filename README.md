# Large Language Models in Resolving Contextual Knowledge Conflicts
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ContextConflict-yellow)](https://huggingface.co/datasets/AsherYang/ContextConflict)
[![arXiv](https://img.shields.io/badge/arXiv-2609.03148-b31b1b.svg)](https://arxiv.org/abs/2609.03148)

Data and code for the EMNLP 2026 paper *Large Language Models in Resolving Contextual Knowledge Conflicts*.

Most prior work on knowledge conflict studies the tension between an LLM's parametric memory and external context. This work instead studies conflicts that arise **within the contextual evidence itself**. We introduce a taxonomy of six contextual conflict types (misinformation, inferential, temporal, granularity, perspective, ambiguity) and release **ContextConflict**, a dataset of 5,781 samples covering both reasoning and summarization tasks with explicit and implicit conflicts. We evaluate seven LLMs, analyze how conflict is detected and geometrically organized inside the model, and show a systematic bias toward earlier evidence at both the representation and the output level.

## Dataset

The dataset is in [`ContextConflict_Dataset/`](ContextConflict_Dataset/) (see its [README](ContextConflict_Dataset/README.md) for the schema) and on the Hugging Face Hub:

```python
from datasets import load_dataset

ds = load_dataset("AsherYang/ContextConflict", "misinformation_conflict")
```

Available configs: `ambiguity_conflict`, `temporal_conflict`, `granularity_conflict`, `inferential_conflict`, `misinformation_conflict`, `perspective_conflict`.

| Conflict type | Samples | Task | Sub-sources |
|---|---|---|---|
| Ambiguity | 1000 | Summarization | AmbigDocs |
| Temporal | 960 | Reasoning | root, `cb_claim_evidence_tmp` |
| Granularity | 1020 | Summarization | `medical_qa`, `ROAST-ABSA` |
| Inferential | 787 | Reasoning | `entailment_bank`, `folio`, `medical_qa` |
| Misinformation | 1004 | Reasoning | `cb_claim_evidence`, `sci_data` |
| Perspective | 1010 | Summarization | `allsides`, `perspectrum` |

## Repository structure

```
ContextConflict_Dataset/      Dataset: 5,781 JSON samples across the six conflict types
evaluation/                   Response generation and the three paper metrics
  evaluator.py                Shapley-based evidence attribution, NLI and claim-level scoring
  batch_evaluate.py           Runs the attribution over a directory of generated responses
  generate_responses_api.py   Response generation with API models
  generate_responses_local.py Response generation with local Hugging Face models
  accuracy.py                 Accuracy on the reasoning conflict types
  balance_score.py            Evidence Balance (normalized Gini) on the summarization types
  faithfulness_ragas.py       Faithfulness via RAGAS
  scripts/                    SLURM/bash templates
analysis/                     Mechanistic analyses
  conflict_aware_data_loader.py     Builds conflict/consistent prompt pairs per conflict type
  run_cross_conflict_analysis.py    Concept vectors + spectral energy across the six types
  run_subfolder_analysis.py         The same, per sub-source
  concept_vector/                   Layer-wise probes and AUC
  spectral_energy/                  Energy ratio and delta-ER
  position_bias/                    Directional projection, pie charts, shuffling experiment
  scripts/                          SLURM/bash templates
```

## Setup

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

Environment variables, set only what the step you run needs:

| Variable | Needed by |
|---|---|
| `HF_TOKEN` | Gated Hugging Face models (Llama) |
| `HF_HOME` | Optional; Hugging Face cache location |
| `OPENAI_API_KEY` | `generate_responses_api.py` |
| `OPENAI_BASE_URL` | Optional; OpenAI-compatible endpoints other than the default |
| `DEEPSEEK_API_KEY` | `faithfulness_ragas.py` |

All scripts take paths relative to the repository root and default to `ContextConflict_Dataset/data` for input and `results/` for output.

## Experiment

### Section 3: evaluation

Generate responses, then score them. The generation scripts write each sample to `results/responses/<model>/<conflict_type>/[<sub-source>/]<id>.json` with the answer under the key `<model>_response`, so the scoring step is told which key to read.

```bash
# API models
OPENAI_API_KEY=... python evaluation/generate_responses_api.py --model gpt-5

# Local Hugging Face models
python evaluation/generate_responses_local.py \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --model-name llama-3.1-8b-instruct

# Shapley-based evidence attribution, written back into each JSON under the key "eval"
python evaluation/batch_evaluate.py \
    --model llama-3.1-8b-instruct \
    --response-key "{model}_response" \
    --eval-model meta-llama/Llama-3.2-1B-Instruct \
    --eval-methods perplexity \
    --answer-only
```

Then compute the three metrics:

```bash
# Accuracy on reasoning tasks
python evaluation/accuracy.py --result-dir results/responses --response-key "{model}_response"

# Evidence Balance (normalized Gini) on summarization tasks
python evaluation/balance_score.py --result-dir results/responses --eval-key eval

# Faithfulness (RAGAS)
DEEPSEEK_API_KEY=... python evaluation/faithfulness_ragas.py --result-dir results/responses
```

### Section 4.1: conflict awareness via concept vectors

A logistic-regression probe is trained on the residual stream at every layer to separate conflict from consistent inputs; the reported number is 5-fold stratified cross-validated AUC.

```bash
python analysis/run_cross_conflict_analysis.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sample_limit 200 --skip_sea

# Per sub-source (inferential and perspective conflicts)
python analysis/run_subfolder_analysis.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --conflict_type inferential_conflict --skip_sea
```

### Section 4.2: spectral energy analysis

Energy ratio of the top-10 singular values of the centered activation matrix, and its conflict-minus-consistent difference.

```bash
python analysis/spectral_energy/run_sea_cross_conflict.py \
    --model_path openai/gpt-oss-20b \
    --sample_limit 400 --top_k 10 --n_bootstrap 500

python analysis/spectral_energy/run_sea_subfolder.py \
    --model_path openai/gpt-oss-20b \
    --conflict_type perspective_conflict --sample_limit 200
```

`run_cross_conflict_analysis.py` runs both analyses in one pass; use `--skip_concept_vector` or `--skip_sea` to run only one.

### Section 4.3: evidence position bias

Representation level. For each sample the script builds one combined-evidence prompt and one prompt per evidence piece, then measures which evidence direction the combined activation leans toward at every layer.

```bash
python analysis/position_bias/run_directional_bias.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --prompt_type simple
```

Output level. Mean evidence contribution share per position, from the Shapley attribution already stored in the response JSONs.

```bash
python analysis/position_bias/evidence_order_bias_pie.py --result-dir results/responses
```

Position shuffling. Permutes evidence order at inference time and re-measures the contribution distribution.

```bash
python analysis/position_bias/shuffle_experiment.py
python analysis/position_bias/shuffle_pie_charts.py
```


## Citation

```bibtex
@misc{yang2026largelanguagemodelsresolving,
      title={Large Language Models in Resolving Contextual Knowledge Conflicts},
      author={Xinye Yang and Zhenyang Liu and Ruisi Li and Yuanyuan Lei},
      year={2026},
      eprint={2609.03148},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.03148},
}
```

