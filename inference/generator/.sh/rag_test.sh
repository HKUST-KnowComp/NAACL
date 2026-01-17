#!/bin/bash

# Usage:
#   bash inference/generator/.sh/rag_test.sh
# This script runs rag_test inference with different retrieval methods
# Supports both bm25-facts and Contriever-facts with multiple prompt types

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
# Note: For rag_test, input file should contain bm25-facts and Contriever-facts fields
INPUT_FILE="datasets/rag/${DATASET_NAME}/${DATASET_SPLIT}.json"  # Adjust path as needed
DATASET="${DATASET_NAME}"
QUESTION_TYPE="oe"  # Auto-detected from item ID ('s' prefix = bi, others = oe)

# ---------------------------------------------------------------------------
# Inference configuration
# ---------------------------------------------------------------------------
TASK="rag_test"
SAMPLE_NUM=1
TEMPERATURE=0.0

# Retrieval methods to test
FACT_TYPES=(
    "bm25-facts"
    "Contriever-facts"
)

# Prompt types to test
PROMPT_TYPES=(
    "vanilla"
    "cot"
    "multi-step"
)

# ---------------------------------------------------------------------------
# Output configuration
# ---------------------------------------------------------------------------
TS="$(date +"%m-%d-%H-%M")"  # month-day-hour-minute
OUTPUT_ROOT="inference/output_data/${TASK}"
OUTPUT_DIR="${OUTPUT_ROOT}/${TS}"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Running RAG test inference"
echo "Dataset    : $DATASET"
echo "Input file : $INPUT_FILE"
echo "Output dir : $OUTPUT_DIR"
echo "Note: Question type is auto-detected from item IDs"
echo "  - IDs starting with 's' -> binary (yes/no)"
echo "  - Other IDs -> open-ended"
echo ""
echo "Strategy: Each model runs tasks sequentially"
echo "          Different models run in parallel"
echo "=========================================="

# Function to run all tasks for a single model
run_model_tasks() {
    local base_model=$1
    local model_name=$2
    local port=$3
    
    echo ""
    echo "=========================================="
    echo "Starting tasks for: $model_name (Port: $port)"
    echo "=========================================="
    
    # Loop through all fact types
    for fact_type in "${FACT_TYPES[@]}"; do
        
        # Loop through all prompt types
        for prompt_type in "${PROMPT_TYPES[@]}"; do
            
            # Output path: model_dataset_facttype_prompttype.json
            output_file="$OUTPUT_DIR/${DATASET}_${model_name}.json"
            
            echo ""
            echo "[$model_name] Running: $fact_type + $prompt_type"
            echo "  Output: $output_file"
            
            # Run inference using budget_forcing.py (SYNCHRONOUSLY)
            python inference/generator/budget_forcing.py \
                --input_file "$INPUT_FILE" \
                --dataset "$DATASET" \
                --output_file "$output_file" \
                --task "$TASK" \
                --prompt_type "$prompt_type" \
                --question_type "$QUESTION_TYPE" \
                --sample_num "$SAMPLE_NUM" \
                --model_name "$base_model" \
                --temperature "$TEMPERATURE" \
                --port "$port" \
                --fact_used "$fact_type" \
                --start_index 0 \
                --end_index 0
            
            local exit_code=$?
            if [ $exit_code -eq 0 ]; then
                echo "  ✓ Completed: $model_name + $fact_type + $prompt_type"
            else
                echo "  ✗ Failed: $model_name + $fact_type + $prompt_type (exit code: $exit_code)"
            fi
        done
    done
    
    echo "=========================================="
    echo "Completed all tasks for: $model_name"
    echo "=========================================="
}

# Launch tasks for all models IN PARALLEL (each model runs its tasks sequentially)
for i in "${!BASE_MODELS[@]}"; do
    base_model="${BASE_MODELS[$i]}"
    model_name="${BASE_MODEL_NAMES[$i]}"
    port="${PORTS[$i]}"
    
    # Run all tasks for this model in the background
    run_model_tasks "$base_model" "$model_name" "$port" &
    
    echo "Launched task sequence for: $model_name"
done

# Wait for all background model tasks to complete
echo ""
echo "=========================================="
echo "Waiting for all models to complete..."
echo "=========================================="
wait

echo ""
echo "=========================================="
echo "All RAG test inference tasks completed!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To view results:"
echo "  ls -lh $OUTPUT_DIR/*.json"
echo "=========================================="
