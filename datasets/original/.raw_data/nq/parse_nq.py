import json
import os

def parse_nq_data(input_file, output_file):
    """
    Parse Natural Questions data from the original JSON format to the desired format.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to the output JSON file
    """
    parsed_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        for idx, item in enumerate(data):
            parsed_item = {
                "id": idx,
                "question": item["question"],
                "gt_answer": item["gold_answers"],  # Already in list format
                "gt_passage": item["gold_text"]  # Empty string as gt_passages will be generated later
            }
            parsed_data.append(parsed_item)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(parsed_data, f, indent=2, ensure_ascii=False)
    
    print(f"Parsed {len(parsed_data)} examples and saved to {output_file}")

def main():
    # Define paths
    raw_data_dir = "/project/jiayujeff/noise_confidence/datasets/.raw_data/nq"
    output_dir = "/project/jiayujeff/noise_confidence/datasets/nq"
    
    # Parse nq.json data as test data
    input_file = os.path.join(raw_data_dir, "nq.json")
    test_output = os.path.join(output_dir, "test.json")
    parse_nq_data(input_file, test_output)
    
    print("Natural Questions parsing completed successfully!")

if __name__ == "__main__":
    main()