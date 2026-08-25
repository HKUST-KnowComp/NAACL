#!/usr/bin/env bash

# Generate every supported noise type for the released three-passage datasets.
# Usage: bash noise_generation/generate_noise.sh [max_concurrent_tasks]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MAX_CONCURRENT_TASKS="${1:-64}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-inference/output_data/noise_generated}"

DATASETS=("bamboogle" "hotpotqa" "nq" "strategyqa")
TASKS=("gen_counterfactual" "gen_relevant" "gen_irrelevant" "gen_consistent")

run_task() {
    local dataset="$1"
    local task="$2"
    local input_file="$3"
    local output_file="${OUTPUT_ROOT}/${dataset}/$(basename "$input_file")"

    mkdir -p "$(dirname "$output_file")"
    echo "Processing ${dataset}/${task}: ${input_file} -> ${output_file}"

    "$PYTHON_BIN" noise_generation/inference.py \
        --input_path "$input_file" \
        --output_path "$output_file" \
        --task "$task" \
        --start_idx "$START_IDX" \
        --end_idx "$END_IDX" \
        --max_concurrent_tasks "$MAX_CONCURRENT_TASKS"
}

for dataset in "${DATASETS[@]}"; do
    found_input=false
    for input_file in "datasets/prepared/threePassages/${dataset}"/*.json; do
        if [[ ! -f "$input_file" ]]; then
            continue
        fi
        found_input=true
        for task in "${TASKS[@]}"; do
            run_task "$dataset" "$task" "$input_file"
        done
    done
    if [[ "$found_input" == false ]]; then
        echo "No JSON input found for dataset: $dataset" >&2
    fi
done

echo "Noise generation completed. Outputs are under ${OUTPUT_ROOT}/."
