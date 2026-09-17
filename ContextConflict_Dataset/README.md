# ContextConflict-Dataset

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ContextConflict-yellow)](https://huggingface.co/datasets/AsherYang/ContextConflict)
[![arXiv](https://img.shields.io/badge/arXiv-2609.03148-b31b1b.svg)](https://arxiv.org/abs/2609.03148)

**ContextConflict-Dataset** is introduced in the paper *Large Language Models in Resolving Contextual Knowledge Conflicts*.

The dataset is also available on the Hugging Face Hub, with one config per conflict type:

```python
from datasets import load_dataset

ds = load_dataset("AsherYang/ContextConflict", "misinformation_conflict")
```

The dataset is designed to evaluate how large language models (LLMs) handle conflicts arising **within contextual knowledge itself**.

It covers **six types of contextual knowledge conflicts** across multiple domains and supports both **reasoning-based** and **summarization-based** tasks.

---

## Directory Structure

```
data/
├── ambiguity_conflict/        # Ambiguity conflicts
├── temporal_conflict/         # Temporal conflicts
│   ├── *.json                 # Implicit temporal conflicts
│   └── cb_claim_evidence_tmp/
├── granularity_conflict/      # Granularity conflicts
│   ├── medical_qa/            # Implicit granularity conflicts
│   └── ROAST-ABSA/
├── inferential_conflict/      # Inferential conflicts
│   ├── entailment_bank/
│   ├── folio/                 # Implicit inferential conflicts
│   └── medical_qa/
├── misinformation_conflict/   # Misinformation conflicts
│   ├── cb_claim_evidence/
│   └── sci_data/
└── perspective_conflict/      # Perspective conflicts
    ├── allsides/              # Implicit perspective conflicts
    └── perspectrum/
```

---

## Conflict Types

| Conflict Type | Description | Task Type |
|---------------|-------------|-----------|
| **Ambiguity Conflict** | Same name refers to different entities | Summarization |
| **Temporal Conflict** | Information from different time periods | Reasoning |
| **Granularity Conflict** | Different levels of abstraction or detail | Summarization |
| **Inferential Conflict** | Different reasoning paths lead to contradictory conclusions | Reasoning |
| **Misinformation Conflict** | Correct vs. incorrect/fabricated information | Reasoning |
| **Perspective Conflict** | Different viewpoints or stances on the same topic | Summarization |

---

## Unified Data Schema

All subsets in ContextConflict-Dataset follow a unified data format, with optional fields depending on the conflict type and task setting.

```json
{
  "question": "...",
  "content": {
    "source_1": "...",
    "source_2": "...",
    "source_3": "..."
  },
  "answer": "...",
  "accuracy_labels": [true, false, true]
}
```

---

## Field Description

| Field | Required | Description |
|-------|----------|-------------|
| `question` | ✓ | Task prompt or question |
| `content` | ✓ | A set of multiple evidence sources providing conflicting information |
| `answer` | ✓ | Reference answer |
| `accuracy_labels` | ○ | Optional boolean labels aligned with evidence sources |

### `answer` Field

- **For reasoning-based tasks**: Ground-truth answer
- **For summarization-based tasks**: A balanced, normative response

### `accuracy_labels` Field

Boolean labels indicating the accuracy of each evidence source:

- `true`: Factually correct / valid reasoning
- `false`: Incorrect information / invalid reasoning

**Used in:**
- Misinformation Conflict
- Inferential Conflict (`medical_qa` subset)

---

## Task Types

### Reasoning-based Tasks

Models must identify the correct answer under conflicting evidence.

| Conflict Type | Subsets |
|---------------|---------|
| Inferential Conflict | `entailment_bank/`, `folio/`, `medical_qa/` |
| Misinformation Conflict | `cb_claim_evidence/`, `sci_data/` |
| Temporal Conflict | `*.json`, `cb_claim_evidence_tmp/` |

### Summarization-based Tasks

Models are expected to integrate multiple conflicting sources without privileging a single viewpoint.

| Conflict Type | Subsets |
|---------------|---------|
| Ambiguity Conflict | - |
| Granularity Conflict | `medical_qa/`, `ROAST-ABSA/` |
| Perspective Conflict | `allsides/`, `perspectrum/` |



## Citation

If you use this dataset in your research, please cite:

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

