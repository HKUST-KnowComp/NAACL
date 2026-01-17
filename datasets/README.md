# Datasets

This directory contains all dataset files used in the project, organized into three main subdirectories.

## Directory Structure

```
datasets/
├── original/           # Original dataset files
│   ├── bamboogle/
│   ├── hotpotqa/
│   ├── nq/
│   └── strategyqa/
├── prepared/           # Preprocessed datasets
│   ├── threePassages/  # Datasets with 3 passages per question
│   └── fivePassage/    # Datasets with 5 passages per question
└── noise_generated/    # Generated noise passages
    ├── bamboogle/
    ├── hotpotqa/
    ├── nq/
    └── strategyqa/
```

## Description

### `original/`

Contains the original dataset files in JSON format. Each dataset (bamboogle, hotpotqa, nq, strategyqa) has its train and/or test splits.

### `prepared/`

Contains preprocessed datasets with structured passages:
- **threePassages/**: Each question is associated with exactly 3 passages
- **fivePassage/**: Each question is associated with 5 passages

These prepared datasets are used for inference tasks.

### `noise_generated/`

Contains generated noise passages created by the `noise_generation/` module. Each dataset subdirectory contains noise passages for different noise types:
- Counterfactual passages
- Relevant noise passages  
- Irrelevant noise passages
- Consistent passages

## Usage

These datasets are used by:
1. **Noise Generation** (`../noise_generation/`) - Reads from `original/` and writes to `noise_generated/`
2. **Inference** (`../inference/`) - Reads from `prepared/` and `noise_generated/` for model inference

## Data Format

Each dataset file is a JSON list where each item contains:
- `id`: Unique identifier
- `question`: The question text
- `answer` or `gold_answers`: Ground truth answer(s)
- `passages`: List of passage objects with `content` and `type` fields
- `facts` or other dataset-specific fields

## Note

- All paths in this directory are relative to the `NAACL/` root directory
- Generated noise files can be regenerated using the `noise_generation/` module
- Prepared datasets are typically created through preprocessing pipelines

