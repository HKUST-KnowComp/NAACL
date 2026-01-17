# Inference

This module handles model inference and evaluation for question-answering tasks with confidence estimation.

## Overview

The inference module consists of two main components:
1. **Generator** (`generator/`) - Generates model responses for QA tasks
2. **Evaluator** (`eval_utils/`) - Extracts answers from responses and evaluates model performance

## Supported Tasks

The module supports five main tasks:

1. **ckpt_test** - Checkpoint testing with passage labeling
2. **base_without_rules** - Baseline inference without specific rules
3. **base_pure** - Pure baseline inference
4. **base_sample** - Baseline inference with step-by-step reasoning and passage classification (used for training data generation)
5. **rag_test** - RAG (Retrieval-Augmented Generation) testing with different fact sources and prompt types

## Directory Structure

```
inference/
├── generator/           # Model response generation
│   ├── base.py         # Base generator class
│   ├── budget_forcing.py  # Main inference script
│   ├── inference_utils.py  # Prompt loading utilities
│   ├── prompts.py      # Prompt templates
│   └── .sh/            # Shell scripts for running inference
├── eval_utils/         # Evaluation utilities
│   ├── extractor.py    # Answer extraction from responses
│   ├── evaluator.py    # Model performance evaluation
│   ├── judge.py        # Answer correctness judging
│   └── .sh/            # Shell scripts for evaluation
└── process_utils/      # Data processing utilities
    └── filter_rule.py  # Filter training data (for base_sample)
```

## Requirements

- Python 3.x
- vLLM or compatible model serving infrastructure
- Required packages: `json`, `argparse`, `tqdm`

## Configuration

### Model Serving

Start model servers using vLLM. You can either use the provided shell scripts or start servers manually:

#### Using Shell Scripts (Recommended)

The module provides three serve scripts for different model types:

**1. Base Models (`base_serve.sh`)**

Serves base models for `base_without_rules`, `base_pure`, `base_sample`, and `rag_test` tasks:

```bash
# From NAACL/ directory
bash inference/generator/.sh/base_serve.sh
```

This script serves 4 models on ports 40001-40004:
- `meta-llama/Llama-3.1-8B-Instruct` on port 40001
- `Qwen/Qwen2.5-7B-Instruct` on port 40002
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` on port 40003
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` on port 40004

**2. Checkpoint Models (`ckpt_serve.sh`)**

Serves LoRA checkpoint models for `ckpt_test` task:

```bash
# From NAACL/ directory
# First, edit the script to set CHECKPOINT_PATH
bash inference/generator/.sh/ckpt_serve.sh
```

**Note**: Before running, update `CHECKPOINT_PATH` in the script with your actual checkpoint directory path. The script serves checkpoints on ports 37000-37005.

**3. Checkpoint Models - No Explanation (`ckpt-noexp_serve.sh`)**

Similar to `ckpt_serve.sh` but for checkpoints without explanation:

```bash
# From NAACL/ directory
# First, edit the script to set CHECKPOINT_PATH
bash inference/generator/.sh/ckpt-noexp_serve.sh
```

**Note**: Update `CHECKPOINT_PATH` before running.

#### Manual Server Start

Alternatively, start model servers manually:

```bash
# Example: Serve DeepSeek-R1 model on port 10000
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --port 10000

# Serve other models on different ports
vllm serve Qwen/Qwen2.5-7B-Instruct --port 10002
vllm serve meta-llama/Llama-3.1-8B --port 10003
```

### Environment

Set the model server port via `--port` argument or use default ports configured in the code.

## Usage

### Quick Start with Shell Scripts

The easiest way to run inference is using the provided shell scripts. All scripts automatically:
- Load datasets from `datasets/` directory
- Save outputs to `inference/output_data/TASK_NAME/TIME_STAMP/`
- Handle path resolution relative to `NAACL/` root

**1. Start Model Servers**

```bash
# From NAACL/ directory
# For base models (used by base_without_rules, base_pure, base_sample, rag_test)
bash inference/generator/.sh/base_serve.sh

# For checkpoint models (used by ckpt_test)
# Edit CHECKPOINT_PATH in the script first!
bash inference/generator/.sh/ckpt_serve.sh
```

**2. Run Inference**

```bash
# From NAACL/ directory

# ckpt_test task
bash inference/generator/.sh/ckpt_test.sh

# base_without_rules task
bash inference/generator/.sh/base_without_rules.sh

# base_pure task
bash inference/generator/.sh/base_pure.sh

# base_sample task
bash inference/generator/.sh/base_sample.sh

# rag_test task
bash inference/generator/.sh/rag_test.sh
```

**Output Structure:**

All outputs are saved in the following structure:
```
inference/output_data/
├── ckpt_test/
│   └── MM-DD-HH-MM/
│       └── dataset_model.json
├── base_without_rules/
│   └── MM-DD-HH-MM/
│       └── dataset_model.json
├── base_pure/
│   └── MM-DD-HH-MM/
│       └── dataset_model.json
├── base_sample/
│   └── MM-DD-HH-MM/
│       └── dataset_model.json
└── rag_test/
    └── MM-DD-HH-MM/
        └── dataset_model.json
```

**Script Configuration:**

Each script can be customized by editing the following variables:
- `DATASET_NAME`: Dataset to use (strategyqa, hotpotqa, nq, bamboogle)
- `DATASET_SPLIT`: Split to use (test, train)
- `PASSAGE_COUNT`: Number of passages (threePassages, fivePassage)
- `MODEL_TO_RUN`: Model index for checkpoint tasks (0-3)
- `PROMPT_TYPE`: Prompt style (vanilla, cot, multi-step) for base tasks
- `CHECKPOINTS`: List of checkpoint numbers for ckpt_test
- `PORTS`: Port numbers for model servers

### Manual Usage

### 1. Generate Responses

Generate model responses for a specific task manually:

```bash
# From NAACL/ directory

# Example: ckpt_test task
python inference/generator/budget_forcing.py \
    --input_file datasets/prepared/threePassages/strategyqa/test.json \
    --dataset strategyqa \
    --output_file inference/output_data/ckpt_test/$(date +"%m-%d-%H-%M")/strategyqa-test_Qwen2.5-7B-Instruct_ckpt20.json \
    --task ckpt_test \
    --question_type bi \
    --sample_num 1 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --temperature 0.0 \
    --start_index 0 \
    --end_index 10

# Example: base_without_rules task (prompt_type is optional, defaults to vanilla)
python inference/generator/budget_forcing.py \
    --input_file datasets/prepared/threePassages/hotpotqa/test.json \
    --dataset hotpotqa \
    --output_file inference/output_data/base_without_rules/$(date +"%m-%d-%H-%M")/hotpotqa-test_Qwen2.5-7B-Instruct.json \
    --task base_without_rules \
    --prompt_type vanilla \
    --question_type oe \
    --sample_num 1 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --temperature 0.0

# Example: base_pure task (prompt_type is optional, defaults to vanilla)
python inference/generator/budget_forcing.py \
    --input_file datasets/prepared/threePassages/nq/test.json \
    --dataset nq \
    --output_file inference/output_data/base_pure/$(date +"%m-%d-%H-%M")/nq-test_Qwen2.5-7B-Instruct.json \
    --task base_pure \
    --prompt_type vanilla \
    --question_type oe \
    --sample_num 1 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --temperature 0.0

# Example: base_sample task (for training data generation)
python inference/generator/budget_forcing.py \
    --input_file datasets/prepared/threePassages/hotpotqa/train.json \
    --dataset hotpotqa \
    --output_file inference/output_data/base_sample/$(date +"%m-%d-%H-%M")/hotpotqa-train_Qwen2.5-7B-Instruct.json \
    --task base_sample \
    --question_type oe \
    --sample_num 5 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --temperature 0.7

# Example: rag_test task (requires --fact_used and --prompt_type)
python inference/generator/budget_forcing.py \
    --input_file datasets/rag/strategyqa/test.json \
    --dataset strategyqa \
    --output_file inference/output_data/rag_test/$(date +"%m-%d-%H-%M")/strategyqa_Qwen2.5-7B-Instruct.json \
    --task rag_test \
    --prompt_type vanilla \
    --question_type bi \
    --sample_num 1 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --fact_used bm25-facts \
    --temperature 0.0

# rag_test with different prompt types (vanilla, cot, multi-step)
python inference/generator/budget_forcing.py \
    --input_file datasets/rag/strategyqa/test.json \
    --dataset strategyqa \
    --output_file inference/output_data/rag_test/$(date +"%m-%d-%H-%M")/strategyqa_Qwen2.5-7B-Instruct.json \
    --task rag_test \
    --prompt_type cot \
    --question_type bi \
    --sample_num 1 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --fact_used Contriever-facts \
    --temperature 0.0
```

**Common Arguments:**
- `--input_file`: Path to input JSON file
- `--dataset`: Dataset name (strategyqa, hotpotqa, nq, bamboogle)
- `--output_file`: Path to save output JSON file
- `--task`: Task type (`ckpt_test`, `base_without_rules`, `base_pure`, `base_sample`, `rag_test`)
- `--prompt_type`: Prompt style (default: `vanilla`) - Only used for `rag_test` which supports `vanilla`, `cot`, `multi-step`. For `base_without_rules` and `base_pure`, this parameter exists but doesn't change the prompt template (only affects output structure). Not used for `ckpt_test` or `base_sample`.
- `--question_type`: Question format (`bi`, `mc`, `oe`)
- `--sample_num`: Number of samples to generate per prompt (default: 1)
- `--model_name`: Model identifier
- `--temperature`: Sampling temperature (default: 0.0)
- `--start_index`: Start index for processing (default: 0)
- `--end_index`: End index for processing (if 0, processes all)
- `--port`: Model server port (optional)
- `--fact_used`: Fact source for `rag_test` (`bm25-facts` or `Contriever-facts`)

### 2. Extract Answers

Extract answers and confidence scores from model responses:

```bash
# From NAACL/ directory
python inference/eval_utils/extractor.py \
    --input_path inference/output_data/ckpt_test/TIME_STAMP \
    --output_path inference/output_data/ckpt_test/TIME_STAMP/extracted \
    --extractor ckpt_test \
    --mode overwrite
```

**Extractor Options:**
- `ckpt_test` - Extract answers, confidence, and passage labels
- `base_without_rules` - Extract answers and confidence
- `base_pure` - Extract answers and confidence
- `base_sample` - Extract answers, confidence, and passage classifications (for training data)
- `rag_test` - Extract answers and confidence from RAG outputs

**Arguments:**
- `--input_path`: Directory containing JSON files with model responses
- `--output_path`: Directory to save extracted results
- `--extractor`: Extractor name (auto-detected from path if not specified)
- `--mode`: Processing mode (`add` or `overwrite`)
- `--unmatched_strategy`: How to handle unmatched samples (`skip`, `empty_100`, `empty_0`)

### 3. Evaluate Results

Evaluate extracted results to compute metrics:

```bash
# From NAACL/ directory
python inference/eval_utils/evaluator.py \
    --input-dir inference/output_data/ckpt_test/TIME_STAMP/extracted \
    --output-dir inference/output_data/ckpt_test/TIME_STAMP/evaluated \
    --extractor ckpt_test \
    --original-data-dir datasets/prepared/threePassages \
    --mode overwrite
```

**Arguments:**
- `--input-dir`: Directory containing extracted JSON files
- `--output-dir`: Directory to save evaluation results
- `--extractor`: Extractor name (optional, for metadata)
- `--original-data-dir`: Directory with original data files (required for `ckpt_test` with labels)
- `--mode`: Processing mode (`add` or `overwrite`)
- `--adaptive-ece`: Use adaptive binning for ECE computation
- `--separate-ece`: Compute separate ECE per passage group
- `--decimal-places`: Number of decimal places for results (default: 3)

### 4. Combined Workflow (Recommended)

Use the automated evaluation script for convenience:

```bash
# From NAACL/ directory
bash inference/eval_utils/.sh/eval.sh <input_path> [options]
```

**Examples:**
```bash
# Auto-detect extractor from path
bash inference/eval_utils/.sh/eval.sh inference/output_data/ckpt_test/01-15-10-30

# Specify options
bash inference/eval_utils/.sh/eval.sh inference/output_data/base_without_rules/01-15-10-30 \
    --mode overwrite \
    --separate-ece

# With original data directory (for ckpt_test)
bash inference/eval_utils/.sh/eval.sh inference/output_data/ckpt_test/01-15-10-30 \
    --original-data-dir datasets/prepared/threePassages
```

The script automatically:
- Detects the appropriate extractor from the input path
- Runs extraction
- Runs evaluation
- Creates output directories

### 5. Filter Training Data (for base_sample)

Filter and normalize `base_sample` responses for training data generation:

```bash
# From NAACL/ directory
python inference/process_utils/filter_rule.py \
    --input inference/output_data/base_sample/TIME_STAMP \
    --output inference/output_data/base_sample/TIME_STAMP/filtered \
    --enable-drop 0.05 \
    --tolarate-mismatch
```

**Arguments:**
- `--input`: Directory containing `base_sample` JSON files
- `--output`: Directory to save filtered results
- `--enable-drop`: Fraction of longest responses to drop (e.g., 0.05 for top 5%)
- `--tolarate-mismatch`: Allow at most one mismatch between Relevant and Irrelevant

**Filter Process:**

The filter enforces:
1. Format requirements: All responses must have `Answer`, `Confidence`, and `Passage Classifications`
2. Step-by-step thinking: Non-distill models must include "Step 4:" marker
3. Passage classification accuracy: Labels must match ground truth (with optional tolerance)
4. Length filtering: Drops the longest responses to avoid truncation artifacts

**Note**: The script supports both `sample_prompt` (from baseline_test) and `base_sample` (from NAACL) response formats.

## Metrics

The evaluator computes the following metrics:

- **accuracy** - Answer correctness
- **label_accuracy** - Passage label accuracy (for tasks with labels)
- **ave_conf** - Average confidence score
- **ece** - Expected Calibration Error
- **auroc** - Area Under ROC Curve
- **auprc** - Area Under Precision-Recall Curve
- **valid_sample_portion** - Proportion of valid samples
- **reliability_diagram** - Calibration visualization data

## Data Formats

### Input Format

Input JSON files should contain a list of items with:
- `id`: Unique identifier
- `question`: Question text
- `answer` or `gold_answers`: Ground truth answer(s)
- `passages`: List of passage objects with `content` and optionally `type`

### Output Format (Generation)

Output JSON files contain the same structure with added `response` fields:

For `base_without_rules`, `base_pure`, and `rag_test`:
```json
{
  "id": "...",
  "question": "...",
  "passages": [...],
  "response": {
    "task_name": {
      "prompt_type": ["response1", "response2", ...]
    }
  }
}
```

For `ckpt_test` and `base_sample`:
```json
{
  "id": "...",
  "question": "...",
  "passages": [...],
  "response": {
    "task_name": ["response1", "response2", ...]
  }
}
```

### Output Format (Extraction)

Extracted JSON files contain:
```json
{
  "response/task_name/prompt_type": [
    {
      "id": "...",
      "answer": "...",
      "confidence": "85",
      "true_answer": [...],
      "passage_1_label": "...",  // for ckpt_test
      ...
    }
  ]
}
```

## Notes

- **All paths are relative to the `NAACL/` root directory**
- **Model servers must be running before generating responses** - Use `*.serve.sh` scripts or start manually
- **Output directories** are automatically created in `inference/output_data/TASK_NAME/TIME_STAMP/` format
- **Datasets** are automatically loaded from `datasets/prepared/` or `datasets/rag/` directories
- For `ckpt_test`, provide `--original-data-dir` if you want label accuracy metrics
- The extractor automatically detects patterns from model responses
- Multiple prompt types can be used (vanilla, cot, multi-step) for `rag_test`
- Shell scripts automatically handle path resolution and timestamp generation
- For checkpoint serving scripts (`ckpt_serve.sh`, `ckpt-noexp_serve.sh`), remember to set `CHECKPOINT_PATH` before running
- `base_sample` is designed for training data generation with step-by-step reasoning and passage classification

