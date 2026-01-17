#!/usr/bin/env python3
"""
Script to parse bamboogle.json into the standardized format.
Converts from bamboogle format to the target format with id, question, gt_answer, and gt_passage fields.
"""

import json
import os

def parse_bamboogle():
    """Parse bamboogle.json and convert to standardized format."""
    
    # Input and output file paths
    input_file = "/project/jiayujeff/noise_confidence/datasets/.raw_data/bamboogle/bamboogle.json"
    output_file = "/project/jiayujeff/noise_confidence/datasets/bamboogle/test.json"
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the original bamboogle data
    with open(input_file, 'r', encoding='utf-8') as f:
        bamboogle_data = json.load(f)
    
    # Convert to target format
    converted_data = []
    
    for item in bamboogle_data:
        converted_item = {
            "id": item["id"],
            "question": item["question"],
            "gt_answer": item["gold_answers"],  # This is already a list
            "gt_passage": item["gold_text"]
        }
        converted_data.append(converted_item)
    
    # Write converted data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully converted {len(converted_data)} items from bamboogle.json")
    print(f"Output saved to: {output_file}")
    
    # Print a sample of the first converted item for verification
    if converted_data:
        print("\nSample converted item:")
        print(json.dumps(converted_data[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    parse_bamboogle()
