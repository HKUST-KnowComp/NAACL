# Noise Confidence for Question Answering

This repository contains the implementation for studying noise robustness and confidence calibration in question-answering systems, particularly in retrieval-augmented generation (RAG) settings.

## Overview

This project explores how question-answering models handle different types of noise in retrieved passages and how well they can estimate their own confidence. The codebase is organized into three main modules:

1. **Dataset Management** - Stores and organizes original, prepared, and noise-generated datasets
2. **Noise Generation** - Generates synthetic noise passages using LLM APIs
3. **Inference & Evaluation** - Runs model inference and evaluates performance with confidence metrics

## Directory Structure

```
NAACL/
├── datasets/              # Dataset storage
│   ├── original/         # Original QA datasets
│   ├── prepared/         # Preprocessed datasets (3/5 passages per question)
│   └── noise_generated/  # Generated noise passages
├── noise_generation/     # Noise passage generation module
│   ├── inference.py      # Main noise generation script
│   ├── prompt_template.py # Prompt templates for noise types
│   └── generate_noise.sh  # Batch generation script
├── inference/            # Model inference and evaluation
│   ├── generator/        # Model response generation
│   │   ├── budget_forcing.py  # Main inference script
│   │   ├── prompts.py    # Prompt templates
│   │   └── .sh/          # Inference scripts
│   └── eval_utils/       # Evaluation utilities
│       ├── extractor.py  # Answer extraction
│       ├── evaluator.py  # Performance evaluation
│       └── .sh/          # Evaluation scripts
└── rag/                  # RAG-related utilities
```

## Workflow

The typical workflow consists of three main steps:

```
1. Prepare Data
   └── datasets/original/ → datasets/prepared/

2. Generate Noise (Optional)
   └── datasets/original/ → datasets/noise_generated/

3. Run Inference & Evaluation
   └── datasets/prepared/ → inference → output/ → evaluation results
```

### Detailed Workflow

1. **Data Preparation**: Start with original datasets in `datasets/original/`
2. **Preprocessing**: Prepare datasets with 3 or 5 passages per question → `datasets/prepared/`
3. **Noise Generation** (Optional): Generate synthetic noise passages → `datasets/noise_generated/`
4. **Model Inference**: Generate model responses for QA tasks → `output/`
5. **Answer Extraction**: Extract answers and confidence scores from responses → `output/extracted/`
6. **Evaluation**: Compute metrics (accuracy, ECE, calibration, etc.) → `output/evaluated/`

## Supported Tasks

The inference module supports five main task types:

- **ckpt_test** - Checkpoint testing with passage labeling
- **base_without_rules** - Baseline inference without specific rules
- **base_pure** - Pure baseline inference
- **base_sample** - Baseline inference with step-by-step reasoning for training data generation
- **rag_test** - RAG testing with different fact sources and prompt types

## Supported Datasets

- **StrategyQA** - Binary yes/no questions requiring multi-hop reasoning
- **HotpotQA** - Multi-hop question answering with supporting facts
- **Natural Questions (NQ)** - Open-domain question answering
- **Bamboogle** - Binary questions with Google search results

## Quick Start

### Prerequisites

- Python 3.x
- vLLM or compatible model serving infrastructure (for inference)
- OpenAI-compatible API access (for noise generation)
- See `environment.yml` for package dependencies

### 1. Environment Setup

```bash
# From NAACL/ directory
conda env create -f environment.yml
conda activate <env_name>
```

Or install packages manually:
```bash
pip install openai tqdm asyncio
# Additional packages as needed
```

### 2. Prepare Datasets

Ensure your datasets are in `datasets/original/` with the following structure:
```
datasets/original/
├── strategyqa/
│   └── test.json
├── hotpotqa/
│   ├── train.json
│   └── test.json
└── ...
```

### 3. Generate Noise Passages (Optional)

Generate synthetic noise passages for robustness testing:

```bash
# From NAACL/ directory
bash noise_generation/generate_noise.sh

# Or for a specific task:
python noise_generation/inference.py \
    --input_path datasets/original/strategyqa/test.json \
    --output_path datasets/noise_generated/strategyqa/test.json \
    --task gen_counterfactual \
    --max_concurrent_tasks 10
```

See [`noise_generation/README.md`](noise_generation/README.md) for details.

### 4. Run Model Inference

Generate model responses for a QA task:

```bash
# From NAACL/ directory
# Example: base_without_rules task
python inference/generator/budget_forcing.py \
    --input_file datasets/prepared/threePassages/strategyqa/test.json \
    --dataset strategyqa \
    --output_file output/base_without_rules_output.json \
    --task base_without_rules \
    --prompt_type vanilla \
    --question_type bi \
    --sample_num 1 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --temperature 0.0
```

See [`inference/README.md`](inference/README.md) for more examples.

### 5. Extract and Evaluate

Extract answers and evaluate results:

```bash
# From NAACL/ directory
# Auto-detect extractor and run extraction + evaluation
bash inference/eval_utils/.sh/eval.sh output/base_without_rules_output

# Or manually:
python inference/eval_utils/extractor.py \
    --input_path output \
    --output_path output/extracted \
    --extractor base_without_rules \
    --mode overwrite

python inference/eval_utils/evaluator.py \
    --input-dir output/extracted \
    --output-dir output/evaluated \
    --extractor base_without_rules \
    --mode overwrite
```

## Module Documentation

- **[`datasets/README.md`](datasets/README.md)** - Dataset structure and data formats
- **[`noise_generation/README.md`](noise_generation/README.md)** - Noise passage generation guide
- **[`inference/README.md`](inference/README.md)** - Model inference and evaluation guide

## Key Features

### Noise Types

The noise generation module supports four types of synthetic noise:

1. **Counterfactual** - Passages that contradict the answer while remaining relevant
2. **Relevant** - Passages that share topics but lack sufficient information
3. **Irrelevant** - Passages with no semantic connection to the question
4. **Consistent** - Passages that support the ground truth answer

### Evaluation Metrics

The evaluation module computes:

- **Accuracy** - Answer correctness
- **ECE (Expected Calibration Error)** - Calibration quality
- **AUROC/AUPRC** - Ranking quality of confidence scores
- **Label Accuracy** - Passage label correctness (for ckpt_test)
- **Reliability Diagrams** - Calibration visualization

### Prompt Types

For RAG testing, the module supports different prompting strategies:

- **vanilla** - Standard prompt without reasoning
- **cot (Chain-of-Thought)** - Step-by-step reasoning
- **multi-step** - Multi-step reasoning with per-step confidence

## Configuration

### Model Serving

For inference, start model servers using vLLM:

```bash
# Example: Serve a model on port 10000
vllm serve Qwen/Qwen2.5-7B-Instruct --port 10000
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --port 10001
```

### API Configuration

For noise generation, set API credentials:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="your-api-base-url"
```

Or modify `noise_generation/inference.py` directly.

## Data Formats

### Input Format

All datasets use JSON format with the following structure:

```json
[
  {
    "id": "sample_001",
    "question": "What is the capital of France?",
    "answer": "Paris",
    "passages": [
      {
        "content": "...",
        "type": "relevant"
      }
    ]
  }
]
```

### Output Format

Model inference outputs add `response` fields:

```json
{
  "id": "sample_001",
  "question": "...",
  "passages": [...],
  "response": {
    "task_name": {
      "prompt_type": ["response1", "response2", ...]
    }
  }
}
```

## Notes

- **Root Directory**: All paths in this codebase are relative to the `NAACL/` directory
- **Model Servers**: Ensure model servers are running before inference
- **API Limits**: Adjust `max_concurrent_tasks` based on your API rate limits
- **Extractor Detection**: The evaluation script automatically detects extractors from paths

## Citation

If you use this codebase in your research, please cite:

```bibtex
@article{your_article,
  title={Noise Confidence for Question Answering},
  author={...},
  year={2024}
}
```

## License

MIT

