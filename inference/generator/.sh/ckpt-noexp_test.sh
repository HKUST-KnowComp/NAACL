#!/bin/bash

# Usage:
#   bash baseline_test/generator/.sh/ckpt-noexp_test.sh
# Before running, make sure `ckpt-noexp_serve.sh` (with the same MODEL_TO_RUN /
# CHECKPOINTS / PORTS settings) is already serving the LoRA checkpoints.

set -euo pipefail

cd /project/jiayujeff/noise_confidence/baseline_test

# ---------------------------------------------------------------------------
# Model / checkpoint configuration (keep in sync with ckpt-noexp_serve.sh)
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
    "hotpotqa-train-noexp_deepseek-r1-distill-qwen-7b-lora-sft"
    "hotpotqa-train-noexp_deepseek-r1-distill-llama-8b-lora-sft"
    "hotpotqa-train-noexp_llama-3_1-8b-instruct-lora-sft"
    "hotpotqa-train-noexp_qwen2_5-7b-instruct-lora-sft"
)

MODEL_TO_RUN=0 # 选择要运行的模型索引，0-3 对应上面的 BASE_MODELS 列表
CHECKPOINTS=(20 40 60 80 100 110) #  
PORTS=(37000 37001 37002 37003 37004 37005) #

# ---------------------------------------------------------------------------
# Inference configuration
# ---------------------------------------------------------------------------
INPUT_FILE="../datasets3/strategyqa/test2.json"
DATASET="strategyqa-test"
QUESTION_TYPE="oe"
TASK="ckpt_test"
SAMPLE_NUM=1
TEMPERATURE=0.0

# Output layout
OUTPUT_ROOT="../filter-test_12-17/output"
TS="$(date +"%m-%d-%H-%M")"
TS="12-25-01-12"
RUN_TAG="${TS}_noexp-models_ckpt-test-without-format"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_TAG}"
mkdir -p "$OUTPUT_DIR"

echo "Running ckpt_test inference"
echo "Dataset file : $INPUT_FILE"
echo "Model family : ${BASE_MODEL_NAMES[$MODEL_TO_RUN]}"
echo "Output dir   : $OUTPUT_DIR"
echo "---"

BASE_MODEL="${BASE_MODELS[$MODEL_TO_RUN]}"
BASE_MODEL_NAME="${BASE_MODEL_NAMES[$MODEL_TO_RUN]}"
LORA_DIR="${LORA_DIR_NAMES[$MODEL_TO_RUN]}"

for idx in "${!CHECKPOINTS[@]}"; do
    ckpt="${CHECKPOINTS[$idx]}"
    port="${PORTS[$idx]}"
    lora_name="${LORA_DIR}_ckpt${ckpt}"
    output_file="${OUTPUT_DIR}/${DATASET}_${BASE_MODEL_NAME}_ckpt${ckpt}.json"

    echo ">> Checkpoint ${ckpt}"
    echo "   LoRA name  : ${lora_name}"
    echo "   Base model : ${BASE_MODEL}"
    echo "   Port       : ${port}"
    echo "   Output     : ${output_file}"

    mkdir -p "$(dirname "$output_file")"

    python generator/budget_forcing.py \
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

echo "All ckpt_test jobs launched! Outputs will accumulate under: $OUTPUT_DIR"

