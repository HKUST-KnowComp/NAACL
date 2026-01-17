# Noise Generation

This module generates different types of noise passages for question-answering datasets using LLM APIs.

## Overview

The noise generation module creates synthetic passages that can be used to test model robustness. It supports four types of noise generation:

1. **Counterfactual** (`gen_counterfactual`) - Passages that contradict the ground truth answer while remaining semantically relevant
2. **Relevant** (`gen_relevant`) - Passages that share topics/keywords with the question but lack sufficient information
3. **Irrelevant** (`gen_irrelevant`) - Passages with no semantic connection to the question
4. **Consistent** (`gen_consistent`) - Passages that support the ground truth answer with consistent information

## Directory Structure

```
noise_generation/
├── inference.py        # Main inference script for noise generation
├── prompt_template.py  # Prompt templates for different noise types
└── generate_noise.sh   # Batch generation script
```

## Requirements

- Python 3.x
- OpenAI-compatible API (configured in `inference.py`)
- Required packages: `openai`, `tqdm`, `asyncio`

## Configuration

### API Configuration

Set your API credentials in `inference.py` or as environment variables:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="your-api-base-url"
```

Or modify directly in `inference.py`:
```python
OPENAI_API_KEY = "your-api-key"
OPENAI_BASE_URL = "your-api-base-url"
```

### Model Configuration

Model settings are configured in `prompt_template.py`:
- `MODEL_NAME`: Model to use (default: "gemini-2.5-pro")
- `Generation_Config`: Generation parameters (temperature, max_output_tokens)

## Usage

### Single Task Generation

Generate noise passages for a specific dataset:

```bash
# From NAACL/ directory
python noise_generation/inference.py \
    --input_path datasets/original/hotpotqa/test.json \
    --output_path datasets/noise_generated/hotpotqa/test.json \
    --task gen_counterfactual \
    --max_concurrent_tasks 10
```

**Arguments:**
- `--input_path`: Path to input JSON file
- `--output_path`: Path to output JSON file
- `--task`: Noise generation task type
  - `gen_counterfactual` - Generate counterfactual passages
  - `gen_relevant` - Generate relevant noise passages
  - `gen_irrelevant` - Generate irrelevant noise passages
  - `gen_consistent` - Generate consistent passages
- `--max_concurrent_tasks`: Maximum number of concurrent API calls (default: 10)
- `--start_idx`: Start index for processing (default: 0)
- `--end_idx`: End index for processing (if 0, processes all from start_idx)

### Batch Generation

Generate all noise types for all datasets:

```bash
# From NAACL/ directory
bash noise_generation/generate_noise.sh [max_concurrent_tasks]
```

This script will:
1. Process all datasets: bamboogle, hotpotqa, nq, strategyqa
2. Generate all noise types for each dataset
3. Save outputs to `datasets/noise_generated/`

**Example:**
```bash
# Use default max_concurrent_tasks (256)
bash noise_generation/generate_noise.sh

# Custom max_concurrent_tasks
bash noise_generation/generate_noise.sh 64
```

## Examples

### Generate Counterfactual Passages

```bash
python noise_generation/inference.py \
    --input_path datasets/original/strategyqa/test.json \
    --output_path datasets/noise_generated/strategyqa/test.json \
    --task gen_counterfactual \
    --max_concurrent_tasks 10
```

### Generate Relevant Noise (Subset)

```bash
python noise_generation/inference.py \
    --input_path datasets/original/hotpotqa/test.json \
    --output_path datasets/noise_generated/hotpotqa/test.json \
    --task gen_relevant \
    --start_idx 0 \
    --end_idx 100 \
    --max_concurrent_tasks 20
```

### Generate Irrelevant Noise

```bash
python noise_generation/inference.py \
    --input_path datasets/original/nq/test.json \
    --output_path datasets/noise_generated/nq/test.json \
    --task gen_irrelevant \
    --max_concurrent_tasks 10
```

## Output Format

Generated noise passages are added to the output JSON file. Each passage includes:
- `content`: The passage text
- `type`: Passage type (counterfactual, relevant, irrelevant, consistent)
- Additional metadata based on the noise type (e.g., "Counterfactual Answer" for counterfactual passages)

## Notes

- If the output file already exists, it will be loaded and new passages will be appended
- The script uses asynchronous API calls for efficient batch processing
- Progress is tracked using tqdm
- All paths should be relative to the `NAACL/` root directory

