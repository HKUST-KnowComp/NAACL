#!/bin/bash
# bash eval_utils/.sh/eval.sh
#
# Usage:
#   bash eval_utils/.sh/eval.sh <input_path> [options]
#
# Options:
#   --mode <mode>              Processing mode: 'add' or 'overwrite' (default: overwrite)
#   --output-base <path>       Base directory for output (default: auto-generated from input_path)
#   --original-data-dir <path> Directory containing original data files (required for ckpt_test)
#   --separate-ece             Enable separate ECE computation
#   --extractor <name>         Override auto-detected extractor (optional)
#
# Examples:
#   bash eval_utils/.sh/eval.sh output/12-01-17_base_without_rules
#   bash eval_utils/.sh/eval.sh output/12-17-22-06_ckpt_test --original-data-dir output
#   bash eval_utils/.sh/eval.sh output/rag_test_data --extractor rag_test

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the eval_utils directory (parent of .sh)
EVAL_UTILS_DIR="$(dirname "$SCRIPT_DIR")"
# Get the project root (assuming this is in NAACL/inference/eval_utils/.sh/)
PROJECT_ROOT="$(cd "$EVAL_UTILS_DIR/../.." && pwd)"

# Determine Python script paths
# Try relative path first (if running from NAACL/inference)
if [ -f "eval_utils/extractor.py" ]; then
    EXTRACTOR_SCRIPT="eval_utils/extractor.py"
    EVALUATOR_SCRIPT="eval_utils/evaluator.py"
elif [ -f "NAACL/inference/eval_utils/extractor.py" ]; then
    EXTRACTOR_SCRIPT="NAACL/inference/eval_utils/extractor.py"
    EVALUATOR_SCRIPT="NAACL/inference/eval_utils/evaluator.py"
else
    # Use absolute path
    EXTRACTOR_SCRIPT="$EVAL_UTILS_DIR/extractor.py"
    EVALUATOR_SCRIPT="$EVAL_UTILS_DIR/evaluator.py"
fi

# Detect extractor based on input path
detect_extractor() {
    local input_path="$1"
    local override_extractor="$2"
    
    # If extractor is explicitly provided, use it
    if [ -n "$override_extractor" ]; then
        echo "$override_extractor"
        return
    fi
    
    # Convert path to lowercase for pattern matching
    local path_lower=$(echo "$input_path" | tr '[:upper:]' '[:lower:]')
    
    # Pattern matching for each extractor
    if [[ "$path_lower" == *"ckpt_test"* ]] || [[ "$path_lower" == *"ckpt-test"* ]] || [[ "$path_lower" == *"ckpttest"* ]]; then
        echo "ckpt_test"
    elif [[ "$path_lower" == *"base_without_rules"* ]] || [[ "$path_lower" == *"base-without-rules"* ]] || [[ "$path_lower" == *"basewithoutrules"* ]]; then
        echo "base_without_rules"
    elif [[ "$path_lower" == *"base_pure"* ]] || [[ "$path_lower" == *"base-pure"* ]] || [[ "$path_lower" == *"basepure"* ]]; then
        echo "base_pure"
    elif [[ "$path_lower" == *"rag_test"* ]] || [[ "$path_lower" == *"rag-test"* ]] || [[ "$path_lower" == *"ragtest"* ]]; then
        echo "rag_test"
    else
        # Default fallback - try to infer from path
        echo "ckpt_test"  # Default fallback
    fi
}

# Process extraction and evaluation
process_data() {
    local input_path="$1"
    local extractor="$2"
    local mode="${3:-overwrite}"
    local output_base="${4:-}"
    local original_data_dir="${5:-}"
    local separate_ece="${6:-false}"
    
    # Determine output directories
    if [ -z "$output_base" ]; then
        # Auto-generate output base from input path
        # Extract directory containing the input path
        local input_dir=$(dirname "$input_path")
        local input_name=$(basename "$input_path")
        output_base="${input_dir}/eval_results/${input_name}"
    fi
    
    local extracted_dir="${output_base}/extracted"
    local evaluated_dir="${output_base}/evaluated"
    
    # Create output directories
    mkdir -p "$extracted_dir"
    mkdir -p "$evaluated_dir"
    
    echo "=========================================="
    echo "Processing: $input_path"
    echo "Extractor: $extractor"
    echo "Mode: $mode"
    echo "Output extracted: $extracted_dir"
    echo "Output evaluated: $evaluated_dir"
    echo "=========================================="
    
    # Step 1: Extract
    echo "Running extractor..."
    local extract_cmd="python $EXTRACTOR_SCRIPT --extractor $extractor --mode $mode --input_path \"$input_path\" --output_path \"$extracted_dir\""
    echo "Command: $extract_cmd"
    eval $extract_cmd
    
    if [ $? -ne 0 ]; then
        echo "Error: Extraction failed!"
        return 1
    fi
    
    # Step 2: Evaluate
    echo ""
    echo "Running evaluator..."
    local eval_cmd="python $EVALUATOR_SCRIPT --mode $mode --input-dir \"$extracted_dir\" --output-dir \"$evaluated_dir\" --extractor $extractor"
    
    # Add original-data-dir if provided (required for ckpt_test with labels)
    if [ -n "$original_data_dir" ]; then
        eval_cmd="$eval_cmd --original-data-dir \"$original_data_dir\""
    fi
    
    # Add separate-ece flag if requested
    if [ "$separate_ece" == "true" ]; then
        eval_cmd="$eval_cmd --separate-ece"
    fi
    
    echo "Command: $eval_cmd"
    eval $eval_cmd
    
    if [ $? -ne 0 ]; then
        echo "Error: Evaluation failed!"
        return 1
    fi
    
    echo ""
    echo "=========================================="
    echo "Completed processing: $input_path"
    echo "Results saved to: $evaluated_dir"
    echo "=========================================="
    echo ""
}

# Main script logic
main() {
    # Check if input path is provided
    if [ $# -eq 0 ]; then
        echo "Usage: $0 <input_path> [options]"
        echo ""
        echo "Options:"
        echo "  --mode <mode>              Processing mode: 'add' or 'overwrite' (default: overwrite)"
        echo "  --output-base <path>       Base directory for output (default: auto-generated)"
        echo "  --original-data-dir <path> Directory containing original data files (for ckpt_test)"
        echo "  --separate-ece             Enable separate ECE computation"
        echo "  --extractor <name>         Override auto-detected extractor"
        echo ""
        echo "Examples:"
        echo "  $0 output/12-01-17_base_without_rules"
        echo "  $0 output/12-17-22-06_ckpt_test --original-data-dir output"
        echo "  $0 output/rag_test_data --extractor rag_test"
        exit 1
    fi
    
    local input_path="$1"
    shift
    
    # Default values
    local mode="overwrite"
    local output_base=""
    local original_data_dir=""
    local separate_ece="false"
    local extractor_override=""
    
    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mode)
                mode="$2"
                shift 2
                ;;
            --output-base)
                output_base="$2"
                shift 2
                ;;
            --original-data-dir)
                original_data_dir="$2"
                shift 2
                ;;
            --separate-ece)
                separate_ece="true"
                shift
                ;;
            --extractor)
                extractor_override="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Detect extractor
    local extractor=$(detect_extractor "$input_path" "$extractor_override")
    
    if [ -z "$extractor" ]; then
        echo "Error: Could not detect extractor from path: $input_path"
        exit 1
    fi
    
    # Check if input path exists
    if [ ! -d "$input_path" ]; then
        echo "Error: Input path does not exist: $input_path"
        exit 1
    fi
    
    # Process the data
    process_data "$input_path" "$extractor" "$mode" "$output_base" "$original_data_dir" "$separate_ece"
}

# If script is being run directly, execute main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
