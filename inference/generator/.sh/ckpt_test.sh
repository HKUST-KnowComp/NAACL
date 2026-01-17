#!/bin/bash

# Usage:
#   bash inference/generator/.sh/ckpt_test.sh
# Before running, make sure `ckpt_serve.sh` (with the same MODEL_TO_RUN /
# CHECKPOINTS / PORTS settings) is already serving the LoRA checkpoints.

set -euo pipefail

# Get the directory where this script is located (should be NAACL/inference/generator/.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Navigate to NAACL/ directory
cd "$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ---------------------------------------------------------------------------
# Model / checkpoint configuration (keep in sync with ckpt_serve.sh)
# ---------------------------------------------------------------------------
BASE_MODELS=(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
)

BASE_MODEL_NAMES=(
    "DeepSeek-R1-Distill-Qwen-7B"
    "DeepSeek-R1-Distill-Llama-8B"
    "Llama-3.1-8B-Instruct"
    "Qwen2.5-7B-Instruct"
)

LORA_DIR_NAMES=(
    "hotpotqa-train_deepseek-r1-distill-qwen-7b-lora-sft"
    "hotpotqa-train_deepseek-r1-distill-llama-8b-lora-sft"
    "hotpotqa-train_llama-3_1-8b-instruct-lora-sft"
    "hotpotqa-train_qwen2_5-7b-instruct-lora-sft"
)

MODEL_TO_RUN=0  # Choose model index: 0-3 corresponding to BASE_MODELS
CHECKPOINTS=(20 40 60 80 100 110)
PORTS=(37000 37001 37002 37003 37004 37005)

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
TASK="ckpt_test"
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
echo "Running ckpt_test inference"
echo "Input file  : $INPUT_FILE"
echo "Dataset     : $DATASET"
echo "Model family: ${BASE_MODEL_NAMES[$MODEL_TO_RUN]}"
echo "Output dir  : $OUTPUT_DIR"
echo "=========================================="

BASE_MODEL="${BASE_MODELS[$MODEL_TO_RUN]}"
BASE_MODEL_NAME="${BASE_MODEL_NAMES[$MODEL_TO_RUN]}"
LORA_DIR="${LORA_DIR_NAMES[$MODEL_TO_RUN]}"

for idx in "${!CHECKPOINTS[@]}"; do
    ckpt="${CHECKPOINTS[$idx]}"
    port="${PORTS[$idx]}"
    lora_name="${LORA_DIR}_ckpt${ckpt}"
    output_file="${OUTPUT_DIR}/${DATASET}_${BASE_MODEL_NAME}_ckpt${ckpt}.json"

    echo ">> Checkpoint ${ckpt}"
    echo "   LoRA name : ${lora_name}"
    echo "   Base model: ${BASE_MODEL}"
    echo "   Port      : ${port}"
    echo "   Output    : ${output_file}"

    mkdir -p "$(dirname "$output_file")"

    python inference/generator/budget_forcing.py \
        --input_file "$INPUT_FILE" \
        --dataset "$DATASET" \
        --output_file "$output_file" \
        --task "$TASK" \
        --question_type "$QUESTION_TYPE" \
        --sample_num "$SAMPLE_NUM" \
        --model_name "$lora_name" \
        --temperature "$TEMPERATURE" \
        --port "$port" \
        --start_index 0 \
        --end_index 0 &

    if [ $? -eq 0 ]; then
        echo "   ✓ Started inference for checkpoint ${ckpt}"
    else
        echo "   ✗ Failed to start inference for checkpoint ${ckpt}"
    fi
    echo "---"
done

echo ""
echo "All ckpt_test jobs launched! Outputs will accumulate under: $OUTPUT_DIR"
