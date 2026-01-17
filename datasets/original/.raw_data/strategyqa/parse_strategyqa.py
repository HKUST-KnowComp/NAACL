import json
import os
import random

def parse_strategyqa_data(input_file, output_file):
    """
    Parse StrategyQA data from the original format to the desired format.
    
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
            "gt_answer": ["no" if not item["answer"] else "yes"],  # Convert boolean to string and make lowercase
            "gt_passage": " ".join(item["facts"])  # Use facts as gt_passage
        }
        parsed_data.append(parsed_item)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(parsed_data, f, indent=2, ensure_ascii=False)
    
    print(f"Parsed {len(parsed_data)} examples and saved to {output_file}")

def sample_strategyqa_test2(input_file, output_file, sample_size=500, facts_lengths=[2, 3]):
    """
    Sample StrategyQA data from train.json with specific facts length constraints.
    
    Args:
        input_file (str): Path to the input JSON file (train.json)
        output_file (str): Path to the output JSON file (test2.json)
        sample_size (int): Number of items to sample (default: 500)
        facts_lengths (list): List of allowed facts lengths (default: [2, 3])
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter items where facts length is 2 or 3
    filtered_data = [item for item in data if len(item["facts"]) in facts_lengths]
    
    print(f"Found {len(filtered_data)} items with facts length {facts_lengths}")
    
    # Sample the requested number of items
    if len(filtered_data) < sample_size:
        print(f"Warning: Only {len(filtered_data)} items available, sampling all of them")
        sampled_data = filtered_data
    else:
        sampled_data = random.sample(filtered_data, sample_size)
    
    # Parse the sampled data
    parsed_data = []
    for index, item in enumerate(sampled_data):
        parsed_item = {
            "id": index,
            "question": item["question"],
            "gt_answer": ["no" if not item["answer"] else "yes"],
            "gt_passage": " ".join(item["facts"])
        }
        parsed_data.append(parsed_item)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(parsed_data, f, indent=2, ensure_ascii=False)
    
    print(f"Sampled {len(parsed_data)} examples and saved to {output_file}")

def main():
    # Define paths
    raw_data_dir = "/project/jiayujeff/noise_confidence/datasets/.raw_data/strategyqa"
    output_dir = "/project/jiayujeff/noise_confidence/datasets/strategyqa"
    
    # # Parse train data
    # train_input = os.path.join(raw_data_dir, "train.json")
    # train_output = os.path.join(output_dir, "train.json")
    # parse_strategyqa_data(train_input, train_output)
    
    # # Parse dev data as test data
    # dev_input = os.path.join(raw_data_dir, "dev.json")
    # test_output = os.path.join(output_dir, "test.json")
    # parse_strategyqa_data(dev_input, test_output)
    
    # Sample 500 items from train.json with facts length 2 or 3 for test2.json
    train_input = os.path.join(raw_data_dir, "train.json")
    test2_output = os.path.join(output_dir, "test2.json")
    sample_strategyqa_test2(train_input, test2_output, sample_size=500, facts_lengths=[2, 3])
    
    print("StrategyQA parsing completed successfully!")

if __name__ == "__main__":
    main()