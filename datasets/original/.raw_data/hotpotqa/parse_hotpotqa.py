import json
import os

def parse_hotpotqa_data(input_file, output_file):
    """
    Parse HotpotQA data from the original format to the desired format.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to the output JSON file
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    parsed_data = []
    
    for index, item in enumerate(data):
        parsed_item = {
            "id": index,
            "question": item["question"],
            "gt_answer": [item["answer"]],  # Convert single answer to list
            "gt_passage": " ".join(item["facts"])  # Use facts as gt_passage
        }
        parsed_data.append(parsed_item)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(parsed_data, f, indent=2, ensure_ascii=False)
    
    print(f"Parsed {len(parsed_data)} examples and saved to {output_file}")

def main():
    # Define paths
    raw_data_dir = "/project/jiayujeff/noise_confidence/datasets/.raw_data/hotpotqa"
    output_dir = "/project/jiayujeff/noise_confidence/datasets/hotpotqa"
    
    # Parse train data
    train_input = os.path.join(raw_data_dir, "train.json")
    train_output = os.path.join(output_dir, "train.json")
    parse_hotpotqa_data(train_input, train_output)
    
    # Parse dev data as test data
    dev_input = os.path.join(raw_data_dir, "dev.json")
    test_output = os.path.join(output_dir, "test.json")
    parse_hotpotqa_data(dev_input, test_output)
    
    print("HotpotQA parsing completed successfully!")

if __name__ == "__main__":
    main()