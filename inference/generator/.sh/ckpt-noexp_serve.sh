#!/bin/bash
# Checkpoint serving script for LoRA checkpoints (no explanation version)
# Usage: bash inference/generator/.sh/ckpt-noexp_serve.sh
# Note: Update the checkpoint paths according to your setup

# Get the directory where this script is located (should be NAACL/inference/generator/.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Navigate to NAACL/ directory
cd "$(cd "$SCRIPT_DIR/../../.." && pwd)"

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

# Actual LoRA directory names under CHECKPOINT_PATH
LORA_DIR_NAMES=(
    "hotpotqa-train-noexp_deepseek-r1-distill-qwen-7b-lora-sft"
    "hotpotqa-train-noexp_deepseek-r1-distill-llama-8b-lora-sft"
    "hotpotqa-train-noexp_llama-3_1-8b-instruct-lora-sft"
    "hotpotqa-train-noexp_qwen2_5-7b-instruct-lora-sft"
)

MODEL_TO_RUN=0  # Choose model index: 0-3 corresponding to BASE_MODELS
CHECKPOINTS=(20 40 60 80 100 110)
PORTS=(37000 37001 37002 37003 37004 37005)
GPUS=(0 1 2 3 4 5)
LOG_PATH="inference/logs/ckpt-noexp_serve"

# ---------------------------------------------------------------------------
# TODO: Update this path according to your checkpoint location
# ---------------------------------------------------------------------------
CHECKPOINT_PATH=""  # PLACEHOLDER: Replace with actual checkpoint path, e.g., "/path/to/LLaMA-Factory/models_noexp"

export VLLM_CONFIGURE_LOGGING=0

# Get selected model
BASE_MODEL="${BASE_MODELS[$MODEL_TO_RUN]}"
BASE_MODEL_NAME="${BASE_MODEL_NAMES[$MODEL_TO_RUN]}"

echo "Starting vLLM servers for model: $BASE_MODEL_NAME (no explanation)"
echo "Base model path: $BASE_MODEL"
echo "Checkpoint path: $CHECKPOINT_PATH"
echo "---"

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "ERROR: CHECKPOINT_PATH is not set. Please update ckpt-noexp_serve.sh with the actual checkpoint path."
    exit 1
fi

# Loop through all checkpoints
for i in "${!CHECKPOINTS[@]}"; do
    checkpoint="${CHECKPOINTS[$i]}"
    port="${PORTS[$i]}"
    gpu_device="${GPUS[$i]}"
    
    # Build LoRA path and name
    lora_dir="${LORA_DIR_NAMES[$MODEL_TO_RUN]}"
    lora_path="${CHECKPOINT_PATH}/${lora_dir}/checkpoint-${checkpoint}"
    lora_name="${lora_dir}_ckpt${checkpoint}"
    
    echo "Starting server for checkpoint-${checkpoint} on GPU ${gpu_device}, port ${port}"
    echo "LoRA path: $lora_path"
    
    # Check checkpoint exists
    if [ ! -d "$lora_path" ]; then
        echo "Warning: Checkpoint path does not exist: $lora_path"
        continue
    fi
    
    # Create log directory
    mkdir -p "$LOG_PATH"
    
    # Start vllm service
    CUDA_VISIBLE_DEVICES=$gpu_device vllm serve $BASE_MODEL \
        --enable-lora \
        --lora-modules $lora_name=$lora_path \
        --port $port \
        --tensor-parallel-size 1 \
        > $LOG_PATH/$lora_name.log 2>&1 &
    
    echo "Started server with PID: $!"
    echo "Log file: $LOG_PATH/$lora_name.log"
    echo "---"
    
    # Wait a bit to avoid starting too many processes at once
    sleep 2
done

echo ""
echo "All servers started. Check logs in $LOG_PATH for details."
