import json
import os
import asyncio
import time
import argparse
import random
import re
from openai import AsyncOpenAI
from prompt_template import *
from tqdm.asyncio import tqdm_asyncio

# OpenAI API configuration
# TODO: Set your API key and base URL here, or set them as environment variables
# Priority: environment variables > these variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Your OpenAI API key (or compatible API key)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")  # API base URL (e.g., "https://api.openai.com/v1" or your custom endpoint)

def format_prompt(prompt_template: str, arguments: dict) -> str:
    """
    Formats the prompt template with the provided arguments.
    """
    return prompt_template.format(**arguments)


def parse_counterfactual_response(response_text: str) -> list:
    """
    Parses the counterfactual response into structured format.
    Expected format:
    Passage 1: [passage text]
    Counterfactual Answer 1: [answer]
    
    Returns list of dicts with 'content', 'type', and 'Counterfactual Answer' keys.
    """
    passages = []
    lines = response_text.strip().split('\n')
    
    current_passage = None
    current_answer = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('Passage '):
            if current_passage and current_answer:
                passages.append({
                    'content': current_passage,
                    'type': 'counterfactual',
                    'Counterfactual Answer': current_answer
                })
            # Extract passage text after "Passage X: "
            current_passage = line.split(': ', 1)[1] if ': ' in line else line
            current_answer = None
        elif line.startswith('Counterfactual Answer '):
            # Extract answer text after "Counterfactual Answer X: "
            current_answer = line.split(': ', 1)[1] if ': ' in line else line
    
    # Add the last passage if exists
    if current_passage and current_answer:
        passages.append({
            'content': current_passage,
            'type': 'counterfactual',
            'Counterfactual Answer': current_answer
        })
    
    return passages


def parse_relevant_response(response_text: str) -> list:
    """
    Parses the relevant noise response into structured format.
    Expected format:
    Passage 1: [passage text]
    Shared Topic/Keywords 1: [topic/keywords]
    
    Returns list of dicts with 'content', 'type', and 'Shared Topic/Keywords' keys.
    """
    passages = []
    lines = response_text.strip().split('\n')
    
    current_passage = None
    current_topic = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('Passage '):
            if current_passage and current_topic:
                passages.append({
                    'content': current_passage,
                    'type': 'relevant',
                    'Shared Topic/Keywords': current_topic
                })
            # Extract passage text after "Passage X: "
            current_passage = line.split(': ', 1)[1] if ': ' in line else line
            current_topic = None
        elif line.startswith('Shared Topic/Keywords '):
            # Extract topic text after "Shared Topic/Keywords X: "
            current_topic = line.split(': ', 1)[1] if ': ' in line else line
    
    # Add the last passage if exists
    if current_passage and current_topic:
        passages.append({
            'content': current_passage,
            'type': 'relevant',
            'Shared Topic/Keywords': current_topic
        })
    
    return passages


def parse_irrelevant_response(response_text: str) -> list:
    """
    Parses the irrelevant noise response into structured format.
    Expected format:
    Passage 1: [passage text]
    Topic 1: [topic]
    
    Returns list of dicts with 'content', 'type', and 'Topic' keys.
    """
    passages = []
    lines = response_text.strip().split('\n')
    
    current_passage = None
    current_topic = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('Passage '):
            if current_passage and current_topic:
                passages.append({
                    'content': current_passage,
                    'type': 'irrelevant',
                    'Topic': current_topic
                })
            # Extract passage text after "Passage X: "
            current_passage = line.split(': ', 1)[1] if ': ' in line else line
            current_topic = None
        elif line.startswith('Topic '):
            # Extract topic text after "Topic X: "
            current_topic = line.split(': ', 1)[1] if ': ' in line else line
    
    # Add the last passage if exists
    if current_passage and current_topic:
        passages.append({
            'content': current_passage,
            'type': 'irrelevant',
            'Topic': current_topic
        })
    
    return passages


def parse_consistent_response(response_text: str) -> list:
    """
    Parses the consistent response into structured format.
    Expected format:
    Passage 1: [passage text]
    Paraphrased Answer: [answer]

    Returns list of dicts with 'content', 'type', and 'Paraphrased Answer' keys.
    """
    passages = []
    lines = response_text.strip().split('\n')

    current_passage = None
    current_paraphrased = None

    for line in lines:
        line = line.strip()
        if line.startswith('Passage '):
            if current_passage and current_paraphrased:
                passages.append({
                    'content': current_passage,
                    'type': 'consistent',
                    'Alternative Expression': current_paraphrased
                })
            # Extract passage text after "Passage X: "
            current_passage = line.split(': ', 1)[1] if ': ' in line else line
            current_paraphrased = None
        elif line.startswith('Alternative Expression'):
            # Extract alternative expression text after "Alternative Expression: "
            current_paraphrased = line.split(': ', 1)[1] if ': ' in line else line

    # Add the last passage if exists
    if current_passage and current_paraphrased:
        passages.append({
            'content': current_passage,
            'type': 'consistent',
            'Alternative Expression': current_paraphrased
        })

    return passages



def _compute_word_range(word_count: int, start: int = 20, width: int = 40) -> str:
    """
    Compute the word_length range string for a given word_count.
    Produce a word range centered on `word_count` with ±20 words (width=40).
    Enforce a minimum lower bound of 15 words and a maximum upper bound of 220 words.
    If word_count is 0 or not provided, return the default range starting at the minimum lower bound.
    """
    # constants
    delta = 20
    min_lower = 10
    max_upper = 220
    width = 40

    if not isinstance(word_count, int) or word_count <= 0:
        lower = min_lower
        upper = min_lower + width
        return f"{lower}-{upper}"

    # start with centered bounds
    lower = word_count - delta
    upper = word_count + delta

    # enforce minimum lower bound
    if lower < min_lower:
        lower = min_lower
        upper = lower + width

    # enforce maximum upper bound
    if upper > max_upper:
        upper = max_upper
        lower = max(min_lower, upper - width)

    # Round to nearest 10
    lower = round(lower / 10) * 10
    upper = round(upper / 10) * 10

    return f"{lower}-{upper}"


def _count_sentences(text: str) -> int:
    """Rudimentary sentence splitter using punctuation. Returns >= 0."""
    import re
    if not text:
        return 0
    # split on sentence end punctuation (one or more), then filter empties
    parts = [p.strip() for p in re.split(r'[\.\!\?]+', text) if p.strip()]
    return len(parts)


async def run_single_inference(args, query: str, passage: str, gt_answer: str, client, model_name, generation_config, word_length: str = None, sentence_length: str = None):
    """
    Asynchronously gets inference result for a single item.
    """
    prompt_template = PROMPT_TEMPLATE[args.task]

    if args.task in ["gen_counterfactual", "gen_relevant", "gen_irrelevant", "gen_consistent"]:
        # Include the word_length and sentence_length into the prompt formatting if available.
        # For gen_consistent we also pass the ground-truth passage so the template can use it.
        format_setting = {
            "query": query,
            "gt_answer": gt_answer,
            "word_length": word_length if word_length is not None else "20-60",
            "sentence_length": sentence_length if sentence_length is not None else "1-2",
            "gt_passage": passage if args.task == "gen_consistent" else "",
        }
    elif args.task == "NLI":
        format_setting = {
            "query": query,
            "passage": passage,
            "gt_answer": gt_answer
        }

    final_prompt = format_prompt(prompt_template, format_setting)
    
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": final_prompt}],
            temperature=generation_config.get("temperature", 0.6),
            max_tokens=generation_config.get("max_output_tokens", 8192),
        )
        
        response_text = response.choices[0].message.content.strip()
                    
        return response_text

    except Exception as e:
        print(f"!!! An exception occurred for query '{query}': {e}")
        return f"Error in query {query}: {e}"

async def run_concurrent_inference(args, model_name: str, tasks_data: list, max_concurrent_tasks: int) -> list:
    """
    Initializes and runs all inference tasks concurrently with a limit on concurrency.

    Args:
        model_name: The name of the generative model to use (e.g., "gpt-4").
        tasks_data: A list of dictionaries, where each dictionary contains
                    'query', 'passage', and 'gt_answer'.
        max_concurrent_tasks: Maximum number of concurrent tasks.

    Returns:
        A list of dictionaries, with each dictionary containing the
        'answer', 'logprobs', and 'norm_probs' for a task.
    """
    # Initialize OpenAI client
    client_kwargs = {}
    if OPENAI_API_KEY:
        client_kwargs["api_key"] = OPENAI_API_KEY
    if OPENAI_BASE_URL:
        client_kwargs["base_url"] = OPENAI_BASE_URL
    
    if not client_kwargs:
        raise ValueError("Please set OPENAI_API_KEY and/or OPENAI_BASE_URL in inference.py")
    
    client = AsyncOpenAI(**client_kwargs)

    generation_config = Generation_Config

    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    async def run_with_semaphore(task):
        async with semaphore:
            return await task

    tasks = []
    for data in tasks_data:
        task = run_with_semaphore(run_single_inference(
            args=args,
            query=data["query"],
            passage=data["passage"],
            gt_answer=data["gt_answer"],
            client=client,
            model_name=model_name,
            generation_config=generation_config,
            word_length=data.get("word_length"),
            sentence_length=data.get("sentence_length"),
        ))
        tasks.append(task)

    results = await tqdm_asyncio.gather(*tasks, desc=f"Running {args.task} tasks")
    return results

def batch_inference(args):
    # Check if we should load from existing output file
    if os.path.exists(args.output_path):
        print(f"Output file {args.output_path} exists. Checking if it can be used as input...")
        try:
            with open(args.output_path, 'r') as f:
                existing_data = json.load(f)
            
            # Check if the existing file has the expected format (list with items having 'id', 'question', etc.)
            if (isinstance(existing_data, list) and len(existing_data) > 0 and 
                all(isinstance(item, dict) and 'id' in item and 'question' in item for item in existing_data[:5])):
                print(f"Using existing output file as input: {args.output_path}")
                data = existing_data
            else:
                print(f"Existing output file format doesn't match expected input format. Loading from input_path: {args.input_path}")
                with open(args.input_path, 'r') as infile:
                    data = json.load(infile)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error reading existing output file: {e}. Loading from input_path: {args.input_path}")
            with open(args.input_path, 'r') as infile:
                data = json.load(infile)
    else:
        with open(args.input_path, 'r') as infile:
            data = json.load(infile)

    TASKS_DATA = []
    information = []

    # Determine the range of items to process
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(data)
    if end_idx == 0:
        end_idx = len(data)
    
    # Ensure valid indices
    start_idx = max(0, start_idx)
    end_idx = min(len(data), end_idx)
    
    if start_idx >= end_idx:
        print(f"Warning: Invalid index range. start_idx({start_idx}) >= end_idx({end_idx})")
        print("Processing all items if possible.")
        if len(data) < 5:
            start_idx = 0
            end_idx = len(data)
        else:
            return []
    
    print(f"Processing items from index {start_idx} to {end_idx-1} (total: {end_idx-start_idx} items)")

    for index in range(start_idx, end_idx):
        item = data[index]

        # Handle different task types - all noise generation tasks don't need input passages
        if args.task in ["gen_counterfactual", "gen_relevant", "gen_irrelevant", "gen_consistent"]:
            # Determine word_length and sentence_length based on gt_passage
            gt_passage = item.get("gt_passage", "")
            # word count (simple whitespace split)
            word_count = len(gt_passage.split()) if gt_passage else 0
            word_range = _compute_word_range(word_count)

            # sentence count (use rudimentary splitter), ensure at least 1
            sent_count = _count_sentences(gt_passage)
            if sent_count <= 0:
                sent_count = 1
            elif sent_count > 8:
                sent_count = 8

            sent_lower = max(1, sent_count - 1)
            sent_upper = sent_count + 1
            sentence_range = f"{sent_lower}-{sent_upper}"

            TASKS_DATA.append({
                "query": item["question"],
                # For gen_consistent we pass the ground-truth passage so the template can reference it.
                "passage": gt_passage if args.task == "gen_consistent" else "",
                "gt_answer": item["gt_answer"][0] if isinstance(item["gt_answer"], list) else item["gt_answer"],
                "word_length": word_range,
                "sentence_length": sentence_range,
            })
        else:
            # For other tasks (supportive noise, NLI)
            passages = item[args.passage_name]
            if isinstance(passages, list):
                passages = random.sample(passages, min(EXAMPLE_NUM, len(passages)))
                passages = [f"{i+1}: {p}" for i, p in enumerate(passages)]
                passage = "  ".join(passages)
            else:
                passage = passages

            TASKS_DATA.append({
                "query": item["question"],
                "passage": passage,
                "gt_answer": item["gt_answer"][0] if isinstance(item["gt_answer"], list) else item["gt_answer"]
            })

        # Preserve original data structure
        result_item = item.copy()
        information.append(result_item)

    start_time = time.time()

    # To run the async function, we use asyncio.run()
    final_results = asyncio.run(run_concurrent_inference(
        args=args,
        model_name=MODEL_NAME,
        tasks_data=TASKS_DATA,
        max_concurrent_tasks=args.max_concurrent_tasks
    ))
    
    end_time = time.time()

    print(f"--- Completed {len(TASKS_DATA)} tasks in {end_time - start_time:.2f} seconds ---")
    
    results = []
    for record, result in zip(information, final_results):
        # Start with the original data structure
        result_entry = record.copy()

        # Use unified top-level 'passages' for all generated passages (each has a 'type')
        if "passages" not in result_entry:
            result_entry["passages"] = []

        # Parse the generated passages based on task type
        parsed_passages = []
        if args.task == "gen_counterfactual":
            parsed_passages = parse_counterfactual_response(result)
        elif args.task == "gen_relevant":
            parsed_passages = parse_relevant_response(result)
        elif args.task == "gen_irrelevant":
            parsed_passages = parse_irrelevant_response(result)
        elif args.task == "gen_consistent":
            parsed_passages = parse_consistent_response(result)
            # print(parsed_passages)
            # exit(0)

        # Only save the last 3 passages for diversity (if any passages were generated)
        if parsed_passages:
            if len(parsed_passages) >= 3:
                to_add = parsed_passages[-3:]
            else:
                to_add = parsed_passages
            result_entry["passages"].extend(to_add)
            # print(f"Added {len(to_add)} passages to item {result_entry.get('id', 'unknown')}")

        results.append(result_entry)

    save_results(results, args.output_path)
    return results

def save_results(results, output_path):
    # # Check if output file already exists
    # if os.path.exists(output_path):
    #     print(f"Output file {output_path} already exists. Loading and merging results...")
    #     with open(output_path, 'r') as f:
    #         existing_results = json.load(f)

    #     # Create a mapping of existing results by ID for efficient lookup
    #     existing_by_id = {item.get('id'): item for item in existing_results}

    #     for new_item in results:
    #         item_id = new_item.get('id')
    #         if item_id in existing_by_id:
    #             # 直接追加新的passages到现有passages后面
    #             existing_item = existing_by_id[item_id]
    #             existing_item["passages"].extend(new_item.get("passages", []))
    #             print(f"Merged passages for item {item_id}")
    #         else:
    #             # 新item，直接添加到结果列表
    #             existing_results.append(new_item)
    #             print(f"Added new item {item_id}")

    #     with open(output_path, 'w') as outfile:
    #         json.dump(existing_results, outfile, indent=2)
    #     print(f"Successfully merged results and saved to {output_path}")
    # else:
    
    # print(f"Creating new output file {output_path}...")
    with open(output_path, 'w') as outfile:
        json.dump(results, outfile, indent=2)
    print(f"Successfully saved {len(results)} items to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Generation and Classification using Gemini-2.5-Pro")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--task", type=str, required=True, choices=["NLI", "gen_supportive", "gen_counterfactual", "gen_relevant", "gen_irrelevant", "gen_consistent"], 
                        help="Task type: NLI, gen_supportive, gen_counterfactual, gen_relevant, gen_irrelevant, or gen_consistent.")
    parser.add_argument("--passage_name", type=str, default="facts", 
                        help="Key name for passages in the input JSON. Not used for noise generation tasks.")
    parser.add_argument("--max_concurrent_tasks", type=int, default=10, help="Maximum number of concurrent tasks.")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for processing items (inclusive).")
    parser.add_argument("--end_idx", type=int, default=None, help="End index for processing items (exclusive). If 0, process all items from start_idx.")
    args = parser.parse_args()

    batch_inference(args)

'''
Example commands:

# Generate counterfactual passages
python NAACL/noise_generation/inference.py \
--input_path datasets/hotpotqa/test.json \
--output_path output/hotpotqa_counterfactual.json \
--task gen_counterfactual \
--max_concurrent_tasks 5

# Generate relevant noise passages for items 0-3
python NAACL/noise_generation/inference.py \
--input_path datasets/hotpotqa/test.json \
--output_path output/hotpotqa_relevant_test.json \
--task gen_relevant \
--start_idx 0 \
--end_idx 3 \
--max_concurrent_tasks 5

# Generate irrelevant noise passages for items 10-20
# If output file exists, it will be used as input and passages will be appended
python NAACL/noise_generation/inference.py \
--input_path datasets/hotpotqa/test.json \
--output_path output/hotpotqa_irrelevant.json \
--task gen_irrelevant \
--start_idx 10 \
--end_idx 20 \
--max_concurrent_tasks 5

# NLI classification
python NAACL/noise_generation/inference.py \
--input_path input_data/test.json \
--output_path output/nli_results.json \
--task NLI \
--passage_name passages \
--max_concurrent_tasks 5

# Note: 
# - Model name is configured in prompt_template.py MODEL_NAME
# - API key and base URL should be set in inference.py (OPENAI_API_KEY, OPENAI_BASE_URL)
# - If output file exists, it will be loaded and new passages will be appended
# - If output file format doesn't match, input_path will be used instead
'''