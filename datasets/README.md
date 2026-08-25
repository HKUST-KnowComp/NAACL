# Datasets

The repository bundles the controlled 3/5-passage inputs used in the NOVA experiments.

```text
datasets/
└── prepared/
    ├── threePassages/  # in-distribution controlled inputs
    └── fivePassage/    # passage-count OOD inputs
```

## Released files

The values below are JSON record counts in the current release.

| Dataset | 3 passages | 5 passages |
|---|---:|---:|
| Bamboogle | 150 | 150 |
| HotpotQA | 745 | 745 |
| Natural Questions | 799 | 799 |
| StrategyQA | 717 | 717 |

HotpotQA also includes 745 three-passage and five-passage training records for exercising the response-generation pipeline.

## Schemas

Records use string IDs and contain exactly three or five passages:

```json
{
  "id": "h0c",
  "question": "...",
  "gt_answer": ["..."],
  "passages": [
    {"content": "...", "type": "counterfactual"}
  ],
  "consistent_answer": ["..."]
}
```

Passage types are `counterfactual`, `relevant`, `irrelevant`, `consistent`, and `gt_passage`. Use `datasets/prepared/` directly for controlled NOVA inference or as input to `noise_generation/inference.py`.
