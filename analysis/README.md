# Analysis

The three mechanistic analyses in Section 4 of the paper: conflict awareness via concept vectors, representational geometry via spectral energy, and evidence position bias.

## Files

| File | Purpose |
|---|---|
| `conflict_aware_data_loader.py` | Builds a conflict prompt and a matched consistent prompt for every sample, with per-type rules |
| `run_cross_conflict_analysis.py` | Concept vector and spectral energy analysis across the six conflict types |
| `run_subfolder_analysis.py` | The same analyses per sub-source within one conflict type |
| `concept_vector/cross_conflict_concept_vector.py` | Layer-wise logistic probes, cross-validated AUC, concept vector similarity, TF-IDF lexical baseline |
| `spectral_energy/cross_conflict_sea.py` | Energy ratio and delta energy ratio with bootstrap confidence intervals |
| `spectral_energy/run_sea_cross_conflict.py` | Spectral energy analysis across conflict types |
| `spectral_energy/run_sea_subfolder.py` | Spectral energy analysis per sub-source |
| `position_bias/evidence_collector.py` | Collects single-evidence and combined-evidence activations |
| `position_bias/directional_bias.py` | Directional projection metric and stacked-area plots |
| `position_bias/run_directional_bias.py` | Runner for the representation-level measurement |
| `position_bias/evidence_order_bias_pie.py` | Output-level contribution share per evidence position |
| `position_bias/shuffle_experiment.py` | Generates responses with permuted evidence order and scores them |
| `position_bias/shuffle_pie_charts.py` | Aggregates the shuffling results into the reported distributions |
| `scripts/` | SLURM and bash templates for the analyses above |

## Conflict and consistent pairs

Every analysis compares a conflicting input against a matched consistent one built from the same sample, so the difference isolates conflict rather than topic. The rules per type, implemented in `conflict_aware_data_loader.py`:

| Conflict type | Consistent counterpart |
|---|---|
| Misinformation | Only the evidence labeled factually correct |
| Inferential (EntailmentBank, FOLIO) | Original entailment chain without the generated conflicting branch |
| Inferential (medical QA) | Only the pieces marked valid by `accuracy_labels` |
| Temporal | Evidence consistent with a single time anchor |
| Granularity | Evidence at one level of specificity |
| Perspective | Evidence sharing one stance label |
| Ambiguity | Evidence referring to a single entity |

## Concept vectors (Section 4.1)

At each layer a logistic-regression probe (`C=1.0`, `max_iter=1000`) is fit on the residual-stream activation of the last token and scored with 5-fold stratified cross-validated AUC.

```bash
python analysis/run_cross_conflict_analysis.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sample_limit 200 \
    --skip_sea

python analysis/run_subfolder_analysis.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --conflict_type inferential_conflict \
    --sample_limit 100 \
    --skip_sea
```

`--conflict_type` accepts `inferential_conflict` or `perspective_conflict`, the two types with sub-sources. Output goes to `results/analysis/cross_conflict` and `results/analysis/subfolder`: AUC curves per layer, best-layer comparison, concept vector similarity heatmap, and the raw scores as JSON.

## Spectral energy (Section 4.2)

Hidden states at the last non-padding token form a matrix per layer, which is centered; the energy ratio is the share of Frobenius norm held by the top-k singular values (k=10, via `torch.svd_lowrank`). Delta-ER is conflict minus consistent: positive means the conflict representation is more concentrated, negative means more dispersed.

```bash
python analysis/spectral_energy/run_sea_cross_conflict.py \
    --model_path openai/gpt-oss-20b \
    --sample_limit 400 --top_k 10 --n_bootstrap 500 --seed 42

python analysis/spectral_energy/run_sea_subfolder.py \
    --model_path openai/gpt-oss-20b \
    --conflict_type perspective_conflict \
    --sample_limit 200 --n_bootstrap 500
```

Bootstrap confidence intervals are on by default; pass `--no_ci` to skip them. Output goes to `results/analysis/sea`.

Running both analyses in one model load is cheaper than running them separately:

```bash
python analysis/run_cross_conflict_analysis.py --model_path meta-llama/Llama-3.1-8B-Instruct
```

## Position bias (Section 4.3)

**Representation level.** For a sample with K evidence pieces the collector builds one combined-evidence prompt and K single-evidence prompts. At each layer, with the neutral center defined as the mean of the single-evidence activations, the combined activation is projected onto each evidence direction; the evidence with the largest projection is the one the model leans toward. The reported figure is the per-layer share of samples leaning toward each position.

```bash
python analysis/position_bias/run_directional_bias.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --prompt_type simple \
    --train_ratio 0.2 --seed 42
```

`--prompt_type simple` uses the short system prompt, which is the setting behind the figure in the paper. Output goes to `results/analysis/position_bias`: the collected activations, per-layer metrics as JSON, and the stacked-area plots.

**Output level.** Mean contribution share per evidence position, read from the Shapley attribution already stored in the response JSONs by `evaluation/batch_evaluate.py`.

```bash
python analysis/position_bias/evidence_order_bias_pie.py \
    --result-dir results/responses \
    --output-dir results/analysis/position_bias/pie_charts
```

By default this combines the full-response and answer-only evaluations with weights 0.4 and 0.6; `--no-weighted` uses the full-response evaluation alone, and `--eval-key-full` / `--eval-key-answer-only` select which keys to read.

**Position shuffling.** Samples 33 items per summarization conflict type, permutes the evidence order, generates with Llama-3.1-8B-Instruct and GPT-OSS-20B, and scores with the Llama-3.2-1B scorer.

```bash
python analysis/position_bias/shuffle_experiment.py --seed 42
python analysis/position_bias/shuffle_pie_charts.py
```

The second script prints the mean contribution distribution per position, which is what the paper tables report. Pass `--skip-gen` to the first script to re-score existing generations.

## Requirements

These analyses load full model weights and read hidden states, so they need a GPU. The 70B model was run on two accelerators; the 8B and 20B models fit on one. `HF_TOKEN` is required for gated Llama checkpoints.
