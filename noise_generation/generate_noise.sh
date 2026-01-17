#!/bin/bash

# Script to generate all noise passage types for datasets
# Usage: bash noise_generation/generate_noise.sh [max_concurrent_tasks]
# Note: This script should be run from the root directory

MAX_CONCURRENT_TASKS=${1:-256}

# Create output directory if it doesn't exist
mkdir -p ./datasets2/

# List of datasets (excluding .raw_data)
DATASETS=("bamboogle" "hotpotqa" "nq" "strategyqa")

# List of noise generation tasks
TASKS=("gen_counterfactual" "gen_relevant" "gen_irrelevant" "gen_consistent")

# Number of samples to test
START_IDX=0
END_IDX=0

echo "Starting noise passage generation test for 3 samples from each dataset..."
echo "Max concurrent tasks: $MAX_CONCURRENT_TASKS"
echo "Datasets: ${DATASETS[*]}"
echo "Tasks: ${TASKS[*]}"
echo "Processing samples: $START_IDX to $((END_IDX-1))"
echo

# Function to run a single task
# Run a single task for a specific input file (e.g., train.json or test.json)
run_task() {
    local dataset=$1
    local task=$2
    local input_file=$3
    local base_name=$(basename "$input_file")
    local output_file="./datasets/noise_generated/${dataset}/${base_name}"

    mkdir -p "./datasets2/${dataset}/"

    echo "Processing: $dataset - $task - $base_name (samples $START_IDX-$((END_IDX-1)))"
    echo "Input: $input_file"
    echo "Output: $output_file"

    mkdir -p "$(dirname "$output_file")"

    python NAACL/noise_generation/inference.py \
        --input_path "$input_file" \
        --output_path "$output_file" \
        --task "$task" \
        --start_idx $START_IDX \
        --end_idx $END_IDX \
        --max_concurrent_tasks "$MAX_CONCURRENT_TASKS"

    wait

    if [ $? -eq 0 ]; then
        echo "✓ Completed: $dataset - $task - $base_name"

        # # Show summary of generated results
        if [ -f "$output_file" ]; then
            local item_count=$(python3 -c "import json,sys; data=json.load(open('$output_file')); print(len(data))")
            local passage_count=$(python3 -c "import json,sys; data=json.load(open('$output_file')); total=sum(len(item.get('passages', [])) for item in data); print(total)")
            echo "  → Generated $passage_count noise passages for $item_count items"
        fi
    else
        echo "✗ Failed: $dataset - $task - $base_name"
    fi
    echo
}

# Function to run all tasks sequentially for a single dataset
run_tasks_for_dataset() {
    local dataset=$1

    # Process all .json files inside the dataset folder (e.g., train.json, test.json)
    for input_file in datasets/original/${dataset}/*.json; do
        if [ ! -f "$input_file" ]; then
            echo "No .json files found for dataset: $dataset"
            break
        fi

        for task in "${TASKS[@]}"; do
            run_task "$dataset" "$task" "$input_file"
        done
    done
}

# Run all datasets concurrently, but tasks within each dataset sequentially
for dataset in "${DATASETS[@]}"; do
    run_tasks_for_dataset "$dataset" &
    # Run each dataset in the background
    sleep 1  # Optional: Add a slight delay to stagger dataset starts

done

# Wait for all background processes to complete
wait

echo "All test generations completed!"
echo "Test output files are saved in the 'datasets2/' directory."

# Optional: Display a summary of all generated files
echo
echo "=== Test Summary ==="
if [ -d "datasets2" ]; then
    echo "Generated files under datasets2/:"
    find datasets2 -type f -name "*.json" -print | while read line; do
        echo "  $line"
    done
else
    echo "No output files generated under datasets2/."
fi