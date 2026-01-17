#!/bin/bash

# Usage:
#   bash inference/generator/.sh/base_without_rules.sh
# Before running, make sure `base_serve.sh` is serving the base models on the
# same ports defined below.

set -euo pipefail

# Get the directory where this script is located (should be NAACL/inference/generator/.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Navigate to NAACL/ directory
cd "$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ---------------------------------------------------------------------------
# Base model configuration (keep in sync with base_serve.sh)
# ---------------------------------------------------------------------------
BASE_MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)

BASE_MODEL_NAMES=(
    "Llama-3.1-8B-Instruct"
    "Qwen2.5-7B-Instruct"
    "DeepSeek-R1-Distill-Qwen-7B"
    "DeepSeek-R1-Distill-Llama-8B"
)

PORTS=(40001 40002 40003 40004)

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
DATASET_NAME="strategyqa"  # Options: strategyqa, hotpotqa, nq, bamboogle
DATASET_SPLIT="test"  # Options: test, train
PASSAGE_COUNT="threePassages"  # Options: threePassages, fivePassage

INPUT_FILE="datasets/prepared/${PASSAGE_COUNT}/${DATASET_NAME}/${DATASET_SPLIT}.json"
DATASET="${DATASET_NAME}-${DATASET_SPLIT}"
QUESTION_TYPE="oe"  # Auto-detected from item ID ('s' prefix = bi, others = oe)

# ---------------------------------------------------------------------------
# Inference configuration
# ---------------------------------------------------------------------------
PROMPT_TYPE="vanilla"  # Options: vanilla, cot, self-probing, multi-step, top-k
TASK="base_without_rules"
SAMPLE_NUM=1
TEMPERATURE=0.0

# ---------------------------------------------------------------------------
# Output configuration
# ---------------------------------------------------------------------------
TS="$(date +"%m-%d-%H-%M")"  # month-day-hour-minute
OUTPUT_ROOT="inference/output_data/${TASK}"
OUTPUT_DIR="${OUTPUT_ROOT}/${TS}"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Running base_without_rules inference"
echo "Input file : $INPUT_FILE"
echo "Dataset    : $DATASET"
echo "Output dir : $OUTPUT_DIR"
echo "=========================================="

for idx in "${!BASE_MODELS[@]}"; do
    base_model="${BASE_MODELS[$idx]}"
    base_model_name="${BASE_MODEL_NAMES[$idx]}"
    port="${PORTS[$idx]}"
    output_file="${OUTPUT_DIR}/${DATASET}_${base_model_name}.json"

    echo ">> Base model: ${base_model_name}"
    echo "   Port      : ${port}"
    echo "   Output    : ${output_file}"

    mkdir -p "$(dirname "$output_file")"

    python inference/generator/budget_forcing.py \
        --input_file "$INPUT_FILE" \
        --dataset "$DATASET" \
        --output_file "$output_file" \
        --task "$TASK" \
        --prompt_type "$PROMPT_TYPE" \
        --question_type "$QUESTION_TYPE" \
        --sample_num "$SAMPLE_NUM" \
        --model_name "$base_model" \
        --temperature "$TEMPERATURE" \
        --port "$port" \
        --start_index 0 \
        --end_index 0 &

    if [ $? -eq 0 ]; then
        echo "   ✓ Started inference for ${base_model_name}"
    else
        echo "   ✗ Failed to start inference for ${base_model_name}"
    fi
    echo "---"
done

echo ""
echo "All base_without_rules jobs launched! Outputs will accumulate under: $OUTPUT_DIR"
