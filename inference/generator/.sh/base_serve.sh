#!/bin/bash
# Base model serving script
# Usage: bash inference/generator/.sh/base_serve.sh
# Note: Update the model paths and ports as needed

export VLLM_CONFIGURE_LOGGING=0

# Get the directory where this script is located (should be NAACL/inference/generator/.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Navigate to NAACL/ directory
cd "$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
# TODO: Update these paths according to your setup
# Placeholder paths - replace with actual model paths
BASE_MODEL_1="meta-llama/Llama-3.1-8B-Instruct"  # Replace with actual path if needed
BASE_MODEL_2="Qwen/Qwen2.5-7B-Instruct"  # Replace with actual path if needed
BASE_MODEL_3="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # Replace with actual path if needed
BASE_MODEL_4="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Replace with actual path if needed

PORTS=(40001 40002 40003 40004)
GPUS=(0 1 2 3)
LOG_DIR="inference/logs"

mkdir -p "$LOG_DIR"

echo "Starting vLLM servers for base models"
echo "Ports: ${PORTS[@]}"
echo "---"

# Start server 1
export CUDA_VISIBLE_DEVICES="${GPUS[0]}"
vllm serve "$BASE_MODEL_1" --port "${PORTS[0]}" > "${LOG_DIR}/llama.log" 2>&1 &
echo "Started server 1: ${BASE_MODEL_1} on port ${PORTS[0]}"

# Start server 2
export CUDA_VISIBLE_DEVICES="${GPUS[1]}"
vllm serve "$BASE_MODEL_2" --port "${PORTS[1]}" > "${LOG_DIR}/qwen.log" 2>&1 &
echo "Started server 2: ${BASE_MODEL_2} on port ${PORTS[1]}"

# Start server 3
export CUDA_VISIBLE_DEVICES="${GPUS[2]}"
vllm serve "$BASE_MODEL_3" --port "${PORTS[2]}" > "${LOG_DIR}/ds_qwen.log" 2>&1 &
echo "Started server 3: ${BASE_MODEL_3} on port ${PORTS[2]}"

# Start server 4
export CUDA_VISIBLE_DEVICES="${GPUS[3]}"
vllm serve "$BASE_MODEL_4" --port "${PORTS[3]}" > "${LOG_DIR}/ds_llama.log" 2>&1 &
echo "Started server 4: ${BASE_MODEL_4} on port ${PORTS[3]}"

echo ""
echo "All base model servers started. Check logs in $LOG_DIR for details."
