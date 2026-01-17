import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

"""
Convert filtered base_sample responses to SFT (Supervised Fine-Tuning) format.

This script processes filtered base_sample outputs and converts them to a format
suitable for training. It:
1. Loads filtered JSON files from multiple models
2. Extracts common IDs across all models
3. Balances passage groups (counterfactual, consistent, irrelevant)
4. Selects best responses (correct with high confidence, or incorrect with low confidence)
5. Formats samples into instruction-input-output format

Usage:
    python inference/process_utils/sft_format.py \
        --input inference/output_data/base_sample/TIME_STAMP/filtered \
        --output inference/output_data/base_sample/TIME_STAMP/sft_formatted
"""

instruction = "You are a helpful assistant"
input_template = """You will be asked a question. You will be provided with 3 retrieved passages.
Each passage belongs to one of these 3 categories:
Highly Relevant: The passage direcly state an answer or strongly indicates an answer, regardless of whether the suggested answer is correct or not.
Relevant: The passage mentions some keywords or shares the same general topic as the question, but lacks information to answer the question.
Irrelevant: The passage has no shared topics or keywords with the question.

Rules:
1. If multiple passages are Highly Relevant, identify if there is a contradiction. 
  - If yes, you should not rely on the passages. Give your final answer based on your own knowledge and give a low confidence score.
  - If no, answer based on the consistent information from the passages and give high confidence score.
2. If exactly one passage is Highly Relevant, give your final answer based on that passage and give high confidence score.
3. If no passage is Highly Relevant, give your final answer based on your own knowledge and give a low confidence score.

Task: Think step by step, analyze the passages one by one and classify their types (Highly Relevant, Relevant, Irrelevant), then follow the rules above to give your final output, including passage classifications, your answer ({question_type}) and confidence score in your answer.

Response Format:
Step 1: ...
Step 2: ...
Step 3: ...
Step 4: ... (Think how to follow the rules)
Final Output (STRICTLY FOLLOW THIS FORMAT):
Passage Classifications:
1. [Type of passage 1]
2. [Type of passage 2]
3. [Type of passage 3]
Answer: [Your answer]
Confidence: [Your confidence score between 0% - 100%]

##
Question: {question}
Retrieved Passages:\n{facts}
##
Your response:"""


# Calculate NAACL root directory (assuming this file is in NAACL/inference/process_utils/)
NAACL_ROOT = Path(__file__).resolve().parents[2]
if str(NAACL_ROOT) not in sys.path:
    sys.path.insert(0, str(NAACL_ROOT))

# Import judge function from NAACL
from inference.eval_utils.judge import f1_judge  # noqa: E402

QUESTION_TYPE = "the most accurate and concise answer"
PASSAGE_GROUPS = ("counterfactual", "consistent", "irrelevant")
TIE_BREAKER = random.Random(42)

# Default paths: can be overridden by command-line arguments
# These are relative to NAACL_ROOT or can be absolute paths
DEFAULT_INPUT_DIR = NAACL_ROOT / "inference" / "output_data" / "base_sample" / "filtered"
DEFAULT_OUTPUT_DIR = NAACL_ROOT / "inference" / "output_data" / "base_sample" / "sft_formatted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert filtered base_sample responses to SFT format."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing filtered JSON files from multiple models.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where SFT-formatted JSON files will be written.",
    )
    return parser.parse_args()


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def load_datasets(input_dir: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load all JSON files from input directory."""
    datasets: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    # Load all JSON files in the directory
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in input directory: {input_dir}")
    
    for json_file in json_files:
        filename = json_file.name
        with json_file.open("r", encoding="utf-8") as fp:
            json_data = json.load(fp)
        # Handle both list format and dict format
        if isinstance(json_data, list):
            datasets[filename] = {sample["id"]: sample for sample in json_data if "id" in sample}
        elif isinstance(json_data, dict):
            # If it's a dict, try to find items with IDs
            datasets[filename] = {}
            for key, value in json_data.items():
                if isinstance(value, dict) and "id" in value:
                    datasets[filename][value["id"]] = value
        else:
            print(f"Warning: Skipping {filename} - unexpected format")
    
    if not datasets:
        raise ValueError(f"No valid data found in input directory: {input_dir}")
    
    print(f"Loaded {len(datasets)} model files from {input_dir}")
    return datasets


def extract_common_ids(datasets: Dict[str, Dict[str, Dict[str, Any]]]) -> List[str]:
    id_sets = [set(data.keys()) for data in datasets.values()]
    if not id_sets:
        return []
    common = set.intersection(*id_sets)
    return sorted(common)


def classify_prompt(item: Dict[str, Any]) -> str:
    passage_types = [
        passage.get("type")
        for passage in item.get("passages", [])
        if isinstance(passage, dict)
    ]
    has_counterfactual = any(t == "counterfactual" for t in passage_types)
    has_gt_or_consistent = any(t in {"gt_passage", "consistent"} for t in passage_types)
    has_only_rel_irrel = (
        len(passage_types) == 3
        and passage_types
        and all(t in {"relevant", "irrelevant"} for t in passage_types)
    )

    if has_counterfactual:
        return "counterfactual"
    if has_gt_or_consistent:
        return "consistent"
    if has_only_rel_irrel:
        return "irrelevant"
    return "unknown"


def summarize_groups(reference_dataset: Dict[str, Any], ids: List[str], label: str) -> None:
    group_counter = Counter(classify_prompt(reference_dataset[_id]) for _id in ids)
    print(f"{label} common items across all files: {len(ids)}")
    for group in PASSAGE_GROUPS:
        print(f"{label} {group.capitalize()} prompts: {group_counter.get(group, 0)}")
    if group_counter.get("unknown"):
        print(f"{label} Unknown prompts: {group_counter['unknown']}")


def balance_groups(common_ids: List[str], reference_dataset: Dict[str, Any]) -> List[str]:
    group_to_ids = {group: [] for group in PASSAGE_GROUPS}
    extra_ids: List[str] = []
    for sample_id in common_ids:
        group = classify_prompt(reference_dataset[sample_id])
        if group in group_to_ids:
            group_to_ids[group].append(sample_id)
        else:
            extra_ids.append(sample_id)

    consistent_ids = group_to_ids["consistent"]
    irrelevant_ids = group_to_ids["irrelevant"]
    target_consistent = len(irrelevant_ids)

    selected_consistent = consistent_ids
    if target_consistent and len(consistent_ids) > target_consistent:
        selected_consistent = TIE_BREAKER.sample(consistent_ids, target_consistent)
        print(
            f"Downsampling consistent prompts from {len(consistent_ids)} to {target_consistent}"
        )

    keep_ids = set(group_to_ids["counterfactual"] + irrelevant_ids + selected_consistent)
    keep_ids.update(extra_ids)
    return [sample_id for sample_id in common_ids if sample_id in keep_ids]


def print_group_metrics(filename: str, stats: Dict[str, Dict[str, float]]) -> None:
    print(f"Metrics for {filename}:")
    for group in PASSAGE_GROUPS:
        group_data = stats.get(group, {})
        total = group_data.get("total", 0)
        if total:
            accuracy = group_data.get("correct", 0) / total
            avg_conf = group_data.get("confidence_sum", 0.0) / total
        else:
            accuracy = 0.0
            avg_conf = 0.0
        print(
            f"  {group.capitalize():<14} | total: {total:4d} | accuracy: {accuracy:.3f} | avg_conf: {avg_conf:.2f}"
        )


def parse_confidence(raw_confidence: Any) -> float:
    if isinstance(raw_confidence, (int, float)):
        return float(raw_confidence)
    if isinstance(raw_confidence, str):
        match = re.search(r"-?\d+(?:\.\d+)?", raw_confidence)
        if match:
            return float(match.group())
        confidence_lower = raw_confidence.strip().lower()
        if "high" in confidence_lower and "low" not in confidence_lower:
            return 90.0
        if "medium" in confidence_lower or "moderate" in confidence_lower:
            return 50.0
        if "low" in confidence_lower:
            return 10.0
    return 0.0


def deduplicate_answers(answers: List[str]) -> List[str]:
    seen = {}
    for ans in answers:
        if ans:
            seen.setdefault(ans, True)
    return list(seen.keys())


def build_ground_truths(item: Dict[str, Any]) -> List[str]:
    """Extract ground truth answers from item. Supports multiple field names."""
    answers: List[str] = []
    # Try various field names (for different data sources)
    gt_fields = ["gt_answer", "gold_answers", "answer", "true_answer"]
    for field in gt_fields:
        field_value = item.get(field)
        if isinstance(field_value, list):
            answers.extend(str(ans) for ans in field_value if ans and ans != "N/A")
        elif isinstance(field_value, str) and field_value and field_value != "N/A":
            answers.append(field_value)
    # Also check for consistent_answer (for noise generation)
    consistent_answer = item.get("consistent_answer")
    if isinstance(consistent_answer, list):
        answers.extend(str(ans) for ans in consistent_answer if ans)
    elif isinstance(consistent_answer, str):
        answers.append(consistent_answer)
    return deduplicate_answers(answers)


def annotate_responses(
    responses: List[Dict[str, Any]], ground_truths: List[str]
) -> List[Dict[str, Any]]:
    annotated = []
    for response in responses:
        model_answer = response.get("model_answer") or response.get("content") or ""
        confidence = parse_confidence(response.get("confidence"))
        is_correct = f1_judge(model_answer, ground_truths) if ground_truths else False
        annotated.append(
            {
                "raw": response,
                "confidence": confidence,
                "is_correct": is_correct,
            }
        )
    return annotated


def select_response(
    annotated_responses: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if not annotated_responses:
        return None

    correct = [resp for resp in annotated_responses if resp["is_correct"]]
    incorrect = [resp for resp in annotated_responses if not resp["is_correct"]]
    candidates: List[Dict[str, Any]] = []

    if correct:
        max_conf = max(resp["confidence"] for resp in correct)
        tied = [resp for resp in correct if resp["confidence"] == max_conf]
        candidates.append(TIE_BREAKER.choice(tied))

    if incorrect:
        min_conf = min(resp["confidence"] for resp in incorrect)
        tied = [resp for resp in incorrect if resp["confidence"] == min_conf]
        candidates.append(TIE_BREAKER.choice(tied))

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    conf_a, conf_b = candidates[0]["confidence"], candidates[1]["confidence"]
    if conf_a == conf_b:
        return TIE_BREAKER.choice(candidates)
    chosen = candidates[0] if conf_a < conf_b else candidates[1]
    return chosen


def build_facts(item: Dict[str, Any]) -> str:
    passages = item.get("passages", [])
    formatted = [f"{idx + 1}: {passage.get('content', '')}\n" for idx, passage in enumerate(passages)]
    return "\n".join(formatted)


def format_sample(
    item: Dict[str, Any], response: Dict[str, Any], *, needs_think_prefix: bool
) -> Dict[str, str]:
    facts = build_facts(item)
    formatted_input = input_template.format(
        question_type=QUESTION_TYPE,
        question=item.get("question", ""),
        facts=facts,
    )
    output_text = (response.get("content") or "").strip()
    if needs_think_prefix and not output_text.startswith("<think>"):
        output_text = f"<think>{output_text}"
    return {
        "instruction": instruction,
        "input": formatted_input,
        "output": output_text,
    }


def process_file(
    filename: str,
    items: Dict[str, Dict[str, Any]],
    common_ids: List[str],
    group_map: Dict[str, str],
    output_dir: Path,
) -> int:
    sft_entries: List[Dict[str, str]] = []
    needs_think_prefix = "Distill" in filename
    group_stats: Dict[str, Dict[str, float]] = {
        group: {"total": 0, "correct": 0, "confidence_sum": 0.0}
        for group in PASSAGE_GROUPS
    }
    for item_id in common_ids:
        if item_id not in items:
            continue
        item = items[item_id]
        responses = item.get("response", {}).get("model_responses", [])
        if not responses:
            continue
        ground_truths = build_ground_truths(item)
        annotated = annotate_responses(responses, ground_truths)
        chosen_response = select_response(annotated)
        if not chosen_response:
            continue
        response_group = group_map.get(item_id, "unknown")
        if response_group in group_stats:
            group_stats[response_group]["total"] += 1
            group_stats[response_group]["confidence_sum"] += chosen_response["confidence"]
            if chosen_response["is_correct"]:
                group_stats[response_group]["correct"] += 1
        sft_entries.append(
            format_sample(
                item,
                chosen_response["raw"],
                needs_think_prefix=needs_think_prefix,
            )
        )

    output_path = output_dir / filename
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(sft_entries, fp, ensure_ascii=False, indent=2)
    print(f"Wrote {len(sft_entries)} SFT samples to {output_path}")
    print_group_metrics(filename, group_stats)
    return len(sft_entries)


def main() -> None:
    args = parse_args()
    
    # Convert relative paths to absolute if needed
    input_dir = args.input if args.input.is_absolute() else (NAACL_ROOT / args.input)
    output_dir = args.output if args.output.is_absolute() else (NAACL_ROOT / args.output)
    
    ensure_output_dir(output_dir)
    datasets = load_datasets(input_dir)
    common_ids = extract_common_ids(datasets)
    if not common_ids:
        print("No common ids found across all datasets.")
        return

    # Use first dataset as reference
    model_files = sorted(datasets.keys())
    reference_filename = model_files[0]
    reference_dataset = datasets[reference_filename]
    
    summarize_groups(reference_dataset, common_ids, "Initial")
    balanced_ids = balance_groups(common_ids, reference_dataset)
    summarize_groups(reference_dataset, balanced_ids, "Balanced")
    group_map = {
        sample_id: classify_prompt(reference_dataset[sample_id])
        for sample_id in balanced_ids
    }
    
    for filename, items in datasets.items():
        process_file(filename, items, balanced_ids, group_map, output_dir)
    
    print(f"\nFinished processing {len(datasets)} model files.")


if __name__ == "__main__":
    main()