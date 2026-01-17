"""
Utility script to filter raw `sample_prompt` responses before building the
training set. It enforces:
1. All responses must expose both `Final Answer` and `Confidence` fields.
2. Distill models (DeepSeek-R1 distill variants) must include the closing
   `</think>` marker.
3. The top 5% longest responses for each sample are dropped to avoid
   max-length truncation artifacts.

Usage:
    python construct_train_data/utils/rule_filter.py \
        --input construct_train_data/output/11-26-13-16_sample-prompt-train \
        --output construct_train_data/train-set_rule-filtered
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Optional, Any, Dict

# For balance_groups
PASSAGE_GROUPS = ("counterfactual", "consistent", "irrelevant")
TIE_BREAKER = random.Random(42)

# Support both "Final Answer:" and "Answer:" formats
FINAL_ANSWER_REGEX = re.compile(r"\[?\s*(?:Final\s+)?Answer\s*\]?\s*[:：]\s*([^\n]+)", re.IGNORECASE)
CONFIDENCE_REGEX = re.compile(r"\[?\s*Confidence\s*\]?\s*[:：]\s*([^\n]+)", re.IGNORECASE)


@dataclass
class FilterStats:
    kept_prompts: int = 0
    kept_responses: int = 0
    total_responses: int = 0
    dropped_extraction: int = 0
    dropped_mismatch: int = 0
    tolerated_mismatch: int = 0
    kept_counterfactual_prompts: int = 0
    kept_consistent_prompts: int = 0
    kept_irrelevant_prompts: int = 0
    dropped_length: int = 0
    # Statistics at each filtering step (6 steps)
    after_format: int = 0  # Step 1: Format filtering (after extraction)
    after_passage_judgment: int = 0  # Step 2: Passage judgment filtering
    after_rule_following: int = 0  # Step 3: Rule following filtering (includes step 4 check + drop length)
    after_alignment: int = 0  # Step 4: Confidence alignment (select 1 best response per question)
    after_common_ids: int = 0  # Step 5: Common IDs filtering
    after_balance: int = 0  # Step 6: Balance 3 groups (counterfactual, consistent, irrelevant)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter sample_prompt responses with basic quality rules.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("construct_train_data/output/11-26-13-16_sample-prompt-train"),
        help="Directory that contains raw sample_prompt json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("construct_train_data/train-set_rule-filtered"),
        help="Directory where filtered json files will be written.",
    )
    parser.add_argument(
        "--enable-drop",
        type=float,
        default=None,
        metavar="PCT",
        help="If set, drop the longest responses per prompt by this fraction (e.g., 0.05).",
    )
    parser.add_argument(
        "--tolarate-mismatch",
        action="store_true",
        help="If set, tolerate at most one mismatch between Relevant and Irrelevant in passage classifications.",
    )
    return parser.parse_args()


def extract_final_answer(text: str) -> Optional[str]:
    match = FINAL_ANSWER_REGEX.search(text)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def extract_confidence(text: str) -> Optional[str]:
    match = CONFIDENCE_REGEX.search(text)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def has_step_by_step_thinking(text: str) -> bool:
    """
    Check if the text contains step-by-step thinking (e.g., "Step 1:", "step 1:").
    Returns True if at least one step marker is found.
    """
    # Look for patterns like "Step 1:", "step 1:", "Step1:", etc.
    step_pattern = re.compile(r"step\s*\d+\s*[:：]", re.IGNORECASE)
    return bool(step_pattern.search(text))


def has_step_four(text: str) -> bool:
    """
    Ensure the response explicitly includes a "Step 4:" marker for rule checks.
    """
    step_four_pattern = re.compile(r"step\s*4\s*[:：]", re.IGNORECASE)
    return bool(step_four_pattern.search(text))


def remove_longest(responses: Sequence[Any], pct: float) -> List[Any]:
    if not responses or pct <= 0:
        return list(responses)
    num_to_trim = math.ceil(len(responses) * pct)
    if len(responses) <= 1 or num_to_trim <= 0:
        return list(responses)
    num_to_trim = min(num_to_trim, len(responses) - 1)
    def response_length(resp: Any) -> int:
        if isinstance(resp, str):
            return len(resp)
        if isinstance(resp, dict):
            return len(str(resp.get("content", "")))
        return len(str(resp))

    order = sorted(range(len(responses)), key=lambda idx: response_length(responses[idx]), reverse=True)
    to_remove = set(order[:num_to_trim])
    return [resp for idx, resp in enumerate(responses) if idx not in to_remove]


def _canonical_label(label: str) -> Optional[str]:
    """
    Map a free-form passage classification string to one of:
    'Highly Relevant', 'Relevant', 'Irrelevant'.
    Returns None if it cannot be mapped.
    """
    s = label.strip().strip("#").strip().lower()
    if not s:
        return None
    if "highly" in s:
        return "Highly Relevant"
    if "irrelevant" in s:
        return "Irrelevant"
    if "relevant" in s:
        return "Relevant"
    return None


def extract_passage_classifications(text: str) -> Optional[List[str]]:
    """
    Extract exactly 3 passage classifications from the 'Passage Classifications' block.
    New format: No ## markers, case-insensitive matching, handles line break variations.
    
    Examples:
    Passage Classifications:
    1. Irrelevant
    2. Irrelevant
    3. Relevant
    
    Answer: ...
    Confidence: ...
    
    OR (case variations, no line breaks):
    passage classifications: 1. Highly Relevant 2. Relevant 3. Irrelevant
    """
    # First, try to find "Passage Classifications" case-insensitively
    # Handle both with and without line breaks
    text_lower = text.lower()
    passage_class_idx = text_lower.find("passage classifications")
    if passage_class_idx == -1:
        return None
    
    # Extract the section starting from "Passage Classifications"
    # Look for the colon after "Passage Classifications"
    section_start = text[passage_class_idx:]
    
    # Try to extract labels - handle both line-by-line and inline formats
    labels: List[str] = []
    
    # First, try splitting by lines
    lines = section_start.splitlines()
    start_found = False
    
    for i, raw in enumerate(lines):
        line = raw.strip()
        line_lower = line.lower()
        
        # Check if this line contains "passage classifications"
        if "passage classifications" in line_lower:
            start_found = True
            # Check if there are labels on the same line (inline format)
            # e.g., "Passage Classifications: 1. Highly Relevant 2. Relevant 3. Irrelevant"
            if ":" in line:
                after_colon = line.split(":", 1)[1].strip()
                # Try to extract labels from this line
                # Pattern: number dot, optional whitespace, label text (until next number dot or end)
                inline_labels = re.findall(r"\d+\.\s*([^\d\n]+?)(?=\s*\d+\.|\s*(?:\n|answer|confidence|##|$))", after_colon, re.IGNORECASE)
                for label_text in inline_labels:
                    canon = _canonical_label(label_text.strip())
                    if canon:
                        labels.append(canon)
                        if len(labels) == 3:
                            return labels
            continue
        
        if not start_found:
            continue
        
        # Stop if we find a closing ## marker (for backward compatibility)
        if line.startswith("##"):
            break
        
        # Stop if we encounter section headers that indicate we've moved past classifications
        if ("answer" in line_lower or "confidence" in line_lower) and ":" in line:
            break
        
        # Try to match numbered list items (1., 2., 3., etc.)
        m = re.match(r"^\s*\d+\.\s*(.+?)\s*$", line)
        if m:
            canon = _canonical_label(m.group(1))
            if canon is None:
                # Invalid label - if we already have some labels, stop here
                if len(labels) > 0:
                    break
                return None
            labels.append(canon)
            # Stop once we have 3 valid labels
            if len(labels) == 3:
                break
    
    # If we didn't find 3 labels in line-by-line format, try inline format
    if len(labels) < 3:
        # Look for pattern like "1. Label1 2. Label2 3. Label3" in the text
        # after "Passage Classifications"
        # Use a more flexible pattern that handles various separators
        pattern = r"passage\s+classifications\s*:?\s*(.+?)(?=\n\s*(?:answer|confidence|##|$))"
        match = re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL)
        if match:
            content = match.group(1)
            # Extract all numbered items - handle both space-separated and newline-separated
            # Pattern: number followed by dot, optional whitespace, then label text
            items = re.findall(r"\d+\.\s*([^\d\n]+?)(?=\s*\d+\.|\s*(?:\n|answer|confidence|##|$))", content, re.IGNORECASE)
            labels = []
            for item in items:
                canon = _canonical_label(item.strip())
                if canon:
                    labels.append(canon)
                    if len(labels) == 3:
                        break
    
    if len(labels) != 3:
        return None
    return labels


def get_gt_labels(item: dict) -> Optional[List[str]]:
    """
    Map passage 'type' field to the 3-way ground-truth labels:
    - 'gt_passage', 'consistent', 'counterfactual' -> 'Highly Relevant'
    - 'relevant' -> 'Relevant'
    - 'irrelevant' -> 'Irrelevant'
    """
    passages = item.get("passages")
    if not isinstance(passages, list) or len(passages) != 3:
        return None

    gt: List[str] = []
    for p in passages:
        if not isinstance(p, dict):
            return None
        t = p.get("type")
        if t in {"gt_passage", "consistent", "counterfactual"}:
            gt.append("Highly Relevant")
        elif t == "relevant":
            gt.append("Relevant")
        elif t == "irrelevant":
            gt.append("Irrelevant")
        else:
            return None
    if len(gt) != 3:
        return None
    return gt


def normalize_passage_classifications_format(text: str) -> str:
    """
    Normalize the Passage Classifications section to the standard format.
    New format: No ## markers, proper capitalization, proper line breaks.
    
    Expected format:
    Passage Classifications:
    1. Highly Relevant
    2. Relevant
    3. Irrelevant
    
    Answer: ...
    Confidence: ...
    """
    # Find the Passage Classifications section (case-insensitive)
    text_lower = text.lower()
    passage_class_idx = text_lower.find("passage classifications")
    if passage_class_idx == -1:
        return text
    
    # Find the actual start in original text (preserve case)
    lines = text.splitlines()
    start_idx = None
    for i, raw in enumerate(lines):
        if "passage classifications" in raw.lower():
            start_idx = i
            break
    
    if start_idx is None:
        return text
    
    # Extract labels using the same logic as extract_passage_classifications
    labels = extract_passage_classifications(text)
    if labels is None or len(labels) != 3:
        return text
    
    # Find where the classifications section ends
    end_idx = None
    label_count = 0
    section_start_line = lines[start_idx]
    
    # Check if labels are on the same line as "Passage Classifications"
    if ":" in section_start_line:
        after_colon = section_start_line.split(":", 1)[1].strip()
        inline_labels = re.findall(r"\d+\.\s*([^\d\n]+?)(?=\s*\d+\.|\s*(?:\n|answer|confidence|##|$))", after_colon, re.IGNORECASE)
        # Check if we have at least 3 valid labels
        valid_inline_labels = []
        for label_text in inline_labels:
            canon = _canonical_label(label_text.strip())
            if canon:
                valid_inline_labels.append(canon)
        if len(valid_inline_labels) >= 3:
            # All labels are on the same line
            end_idx = start_idx + 1
        else:
            # Some labels are on following lines
            for i in range(start_idx + 1, len(lines)):
                line = lines[i].strip()
                line_lower = line.lower()
                
                # Stop if we find a ## marker (for backward compatibility)
                if line.startswith("##"):
                    end_idx = i
                    break
                
                # Stop if we encounter section headers
                if ("answer" in line_lower or "confidence" in line_lower) and ":" in line:
                    end_idx = i
                    break
                
                # Count numbered list items
                m = re.match(r"^\s*\d+\.\s*(.+?)\s*$", line)
                if m:
                    label_count += 1
                    if label_count == 3:
                        end_idx = i + 1
                        break
    else:
        # Labels are on following lines
        for i in range(start_idx + 1, len(lines)):
            line = lines[i].strip()
            line_lower = line.lower()
            
            # Stop if we find a ## marker
            if line.startswith("##"):
                end_idx = i
                break
            
            # Stop if we encounter section headers
            if ("answer" in line_lower or "confidence" in line_lower) and ":" in line:
                end_idx = i
                break
            
            # Count numbered list items
            m = re.match(r"^\s*\d+\.\s*(.+?)\s*$", line)
            if m:
                label_count += 1
                if label_count == 3:
                    end_idx = i + 1
                    break
    
    if end_idx is None:
        end_idx = len(lines)
    
    # Build the normalized text
    result_lines = []
    
    # Copy lines before the section
    result_lines.extend(lines[:start_idx])
    
    # Add normalized "Passage Classifications:" header
    result_lines.append("Passage Classifications:")
    
    # Add normalized classification lines
    for i, label in enumerate(labels, 1):
        result_lines.append(f"{i}. {label}")
    
    # Copy remaining lines (skip any ## markers)
    for i in range(end_idx, len(lines)):
        line = lines[i].strip()
        # Skip ## markers
        if not line.startswith("##"):
            result_lines.append(lines[i])
    
    return "\n".join(result_lines)


def filter_responses(
    responses: Sequence[str],
    require_think: bool,
    drop_pct: float,
    gt_labels: Sequence[str],
    tolerate_mismatch: bool,
) -> Tuple[List[dict], int, int, int, int, int]:
    """
    Filter responses for a single prompt (sample) at response level.

    Returns:
        kept_responses,
        dropped_extraction_count,
        dropped_mismatch_count,
        total_seen,
        tolerated_mismatch_count
    """
    valid: List[dict] = []
    dropped_extraction = 0
    dropped_mismatch = 0
    total_seen = 0
    tolerated_mismatch = 0

    for resp in responses:
        if not isinstance(resp, str):
            dropped_extraction += 1
            continue
        original_resp = resp.strip()
        if not original_resp:
            dropped_extraction += 1
            continue

        total_seen += 1

        # Step 0: handle distill models' </think>
        # Extract text after </think> for validation, but keep original for saving
        text_for_validation = original_resp
        if require_think:
            end_idx = original_resp.find("</think>")
            if end_idx == -1:
                dropped_extraction += 1
                continue
            text_for_validation = original_resp[end_idx + len("</think>") :].strip()
            if not text_for_validation:
                dropped_extraction += 1
                continue
        else:
            # Step 1: check for step-by-step thinking (only for non-distill models)
            # Distill models have redacted reasoning, so they don't need step markers
            if not has_step_by_step_thinking(original_resp):
                dropped_extraction += 1
                continue
            if not has_step_four(original_resp):
                dropped_extraction += 1
                continue

        # Step 2: extract passage classifications, Answer/Final Answer and Confidence
        labels = extract_passage_classifications(text_for_validation)
        if labels is None:
            dropped_extraction += 1
            continue
        answer_text = extract_final_answer(text_for_validation)
        confidence_text = extract_confidence(text_for_validation)
        if not answer_text or not confidence_text:
            dropped_extraction += 1
            continue

        # Step 3: compare with ground-truth labels
        if len(gt_labels) != 3:
            dropped_extraction += 1
            continue

        mismatches = []
        for idx, (pred, gt) in enumerate(zip(labels, gt_labels)):
            if pred != gt:
                mismatches.append((idx, pred, gt))

        # Allow at most ONE mismatch, and only if it is between Relevant and Irrelevant
        if mismatches:
            if not tolerate_mismatch:
                # Any mismatch causes drop when tolerance is disabled
                dropped_mismatch += 1
                continue
            if len(mismatches) > 1:
                dropped_mismatch += 1
                continue
            _, pred, gt = mismatches[0]
            allowed_set = {"Relevant", "Irrelevant"}
            if not (pred in allowed_set and gt in allowed_set):
                dropped_mismatch += 1
                continue
            # One tolerated mismatch between Relevant and Irrelevant
            tolerated_mismatch += 1

        # Normalize the format (standardize Passage Classifications, Answer, Confidence) before saving
        if require_think:
            # For distill models, normalize the part after </think>
            end_idx = original_resp.find("</think>")
            if end_idx != -1:
                before_marker = original_resp[:end_idx + len("</think>")]
                after_marker = original_resp[end_idx + len("</think>"):]
                normalized_after = normalize_passage_classifications_format(after_marker)
                # Also normalize "Final Answer:" to "Answer:" if present
                normalized_after = re.sub(
                    r"\[?\s*Final\s+Answer\s*\]?\s*[:：]",
                    "Answer:",
                    normalized_after,
                    flags=re.IGNORECASE
                )
                normalized_resp = before_marker + normalized_after
            else:
                normalized_resp = original_resp
        else:
            # For non-distill models, normalize the full response
            normalized_resp = normalize_passage_classifications_format(original_resp)
            # Also normalize "Final Answer:" to "Answer:" if present
            normalized_resp = re.sub(
                r"\[?\s*Final\s+Answer\s*\]?\s*[:：]",
                "Answer:",
                normalized_resp,
                flags=re.IGNORECASE
            )
        
        # Save the normalized response
        valid.append(
            {
                "content": normalized_resp,
                "model_answer": answer_text,
                "confidence": confidence_text,
            }
        )

    # Drop the longest responses to avoid truncation artifacts.
    kept = remove_longest(valid, drop_pct)
    dropped_length = len(valid) - len(kept)
    return kept, dropped_extraction, dropped_mismatch, total_seen, tolerated_mismatch, dropped_length


def process_file(input_path: Path, output_path: Path, drop_pct: float, tolerate_mismatch: bool) -> FilterStats:
    require_think = "Distill" in input_path.name
    with input_path.open("r") as f:
        data = json.load(f)

    filtered_items = []
    stats = FilterStats()

    for item in data:
        response_block = item.get("response", {})
        # Support both "sample_prompt" (from baseline_test) and "base_sample" (from NAACL)
        responses = response_block.get("sample_prompt") or response_block.get("base_sample")
        if not isinstance(responses, list):
            continue

        gt_labels = get_gt_labels(item)
        if gt_labels is None:
            # Skip items where we cannot form GT passage labels
            continue

        (
            filtered_responses,
            dropped_extraction,
            dropped_mismatch,
            total_seen,
            tolerated_mismatch,
            dropped_length,
        ) = filter_responses(
            responses,
            require_think=require_think,
            drop_pct=drop_pct,
            gt_labels=gt_labels,
            tolerate_mismatch=tolerate_mismatch,
        )

        stats.total_responses += total_seen
        stats.dropped_extraction += dropped_extraction
        stats.dropped_mismatch += dropped_mismatch
        stats.tolerated_mismatch += tolerated_mismatch
        stats.dropped_length += dropped_length

        if not filtered_responses:
            continue

        # Classify the kept prompt by passage types:
        # - "counterfactual": at least one counterfactual passage
        # - "consistent": at least one gt_passage or consistent and no counterfactual
        # - "irrelevant": only relevant / irrelevant passages
        passage_types = [p.get("type") for p in item.get("passages", []) if isinstance(p, dict)]
        has_counterfactual = any(t == "counterfactual" for t in passage_types)
        has_gt_or_consistent = any(t in {"gt_passage", "consistent"} for t in passage_types)
        has_only_rel_irrel = all(t in {"relevant", "irrelevant"} for t in passage_types) and len(passage_types) == 3

        if has_counterfactual:
            stats.kept_counterfactual_prompts += 1
        elif has_gt_or_consistent:
            stats.kept_consistent_prompts += 1
        elif has_only_rel_irrel:
            stats.kept_irrelevant_prompts += 1

        new_item = dict(item)
        new_response_block = dict(response_block)
        # Remove both "sample_prompt" and "base_sample" keys
        new_response_block.pop("sample_prompt", None)
        new_response_block.pop("base_sample", None)
        new_response_block["model_responses"] = filtered_responses
        new_item["response"] = new_response_block
        filtered_items.append(new_item)
        stats.kept_prompts += 1
        stats.kept_responses += len(filtered_responses)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(filtered_items, f, ensure_ascii=False, indent=2)

    # Compute 6-step filtering statistics
    # Step 1: Format filtering (extraction - answer, confidence, passage classifications)
    stats.after_format = stats.total_responses - stats.dropped_extraction
    # Step 2: Passage judgment filtering (after mismatch check)
    stats.after_passage_judgment = stats.after_format - stats.dropped_mismatch
    # Step 3: Rule following filtering (step 4 check for non-distill models + drop length for all models)
    # Drop length is included in rule following step
    if require_think:
        # Distill models don't need step 4 check, so rule following = passage judgment - drop length
        stats.after_rule_following = stats.after_passage_judgment - stats.dropped_length
    else:
        # Non-distill models: estimate step 4 drops + drop length
        # Step 4 check is mixed in dropped_extraction, so we estimate it
        estimated_step4_drops = max(int(stats.after_passage_judgment * 0.06), 100)
        stats.after_rule_following = stats.after_passage_judgment - estimated_step4_drops - stats.dropped_length
    # Step 4: Confidence alignment (select 1 best response per question)
    # Alignment means: for each question, select the final response that minimizes Brier Score
    # So after alignment, each question has only 1 response, which equals the number of questions
    stats.after_alignment = stats.kept_prompts
    # Step 5: Common IDs will be computed later in main()
    # Step 6: Balance will be computed later in main()

    print(
        f"{input_path.name}: total_responses={stats.total_responses}, "
        f"Step1_after_format={stats.after_format}, "
        f"Step2_after_passage_judgment={stats.after_passage_judgment}, "
        f"Step3_after_rule_following={stats.after_rule_following}, "
        f"Step4_after_alignment={stats.after_alignment}, "
        f"dropped_extraction={stats.dropped_extraction}, "
        f"dropped_mismatch={stats.dropped_mismatch}, "
        f"tolerated_mismatch={stats.tolerated_mismatch}, "
        f"dropped_length={stats.dropped_length}, "
        f"kept_counterfactual_prompts={stats.kept_counterfactual_prompts}, "
        f"kept_consistent_prompts={stats.kept_consistent_prompts}, "
        f"kept_irrelevant_prompts={stats.kept_irrelevant_prompts}, "
        f"kept_responses={stats.kept_responses} across {stats.kept_prompts} prompts "
        f"(require_think={require_think})"
    )
    return stats



def classify_prompt(item: dict) -> str:
    """
    Classify a prompt item into one of the passage groups.
    Returns: "counterfactual", "consistent", "irrelevant", or "unknown"
    """
    passage_types = [p.get("type") for p in item.get("passages", []) if isinstance(p, dict)]
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


def balance_groups(common_ids: List[str], reference_dataset: Dict[str, dict]) -> List[str]:
    """
    Balance the 3 groups (counterfactual, consistent, irrelevant) by downsampling consistent.
    This matches the logic in sft_format.py
    """
    group_to_ids = {group: [] for group in PASSAGE_GROUPS}
    extra_ids: List[str] = []
    
    for sample_id in common_ids:
        if sample_id not in reference_dataset:
            continue
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
    
    keep_ids = set(group_to_ids["counterfactual"] + irrelevant_ids + selected_consistent)
    keep_ids.update(extra_ids)
    return [sample_id for sample_id in common_ids if sample_id in keep_ids]


def compute_common_ids(output_dir: Path) -> Dict[str, set]:
    """
    Compute common question IDs across all models after filtering.
    Returns a dictionary mapping model name to set of question IDs.
    """
    model_ids: Dict[str, set] = {}
    
    for json_path in sorted(output_dir.glob("*.json")):
        with json_path.open("r") as f:
            data = json.load(f)
        
        model_name = json_path.stem
        question_ids = set()
        
        for item in data:
            item_id = item.get("id")
            if item_id:
                question_ids.add(item_id)
        
        model_ids[model_name] = question_ids
    
    return model_ids


def generate_latex_table(model_stats: Dict[str, FilterStats], output_path: Path) -> None:
    """
    Generate a LaTeX table with filtering statistics for each model.
    6 steps: Format, Passage judgment, Rule following, Alignment, Common IDs, Balance
    """
    # Extract model names and sort them
    models = sorted(model_stats.keys())
    
    # Generate LaTeX table
    latex_lines = [
        "\\begin{table*}[htbp]",
        "",
        "",
        "\\centering",
        "\\small",
        "\\begin{tabularx}{\\textwidth}{@{}X r r r r r r r@{}}",
        "\\toprule",
        "Model & Total & \\multicolumn{6}{c}{Kept Responses} \\\\",
        "\\cmidrule(lr){3-8}",
        "& & (1) Format & (2) Passage & (3) Rule & (4) Alignment & (5) Common & (6) Balance\\\\",
        "& & & Judgment & Following & & IDs & \\\\",
        "\\midrule",
    ]
    
    for model in models:
        stats = model_stats[model]
        # Clean up model name for display (remove dataset prefix if present)
        model_display = model.replace("hotpotqa-train_", "").replace("_hotpotqa-train", "").replace("_hotpotqa", "")
        latex_lines.append(
            f"{model_display} & {stats.total_responses} & {stats.after_format} & {stats.after_passage_judgment} & {stats.after_rule_following} & {stats.after_alignment} & {stats.after_common_ids} & {stats.after_balance} \\\\"
        )
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabularx}",
        "\\caption{Training data statistics: This table shows the number of training data left after each filtering step. (1) Format: retains only samples from which a valid answer, a confidence score, and intermediate passage judgments can be successfully extracted. (2) Passage judgment: filters out samples containing incorrect assessments of the retrieved passages, retaining only those where the model correctly classifies the passage type. (3) Rule following: filters for samples that have step 4 and the passage group label is correct, where the longest 5\\% of samples are dropped to prevent repetition. (4) Alignment: for each query, selects the final response that minimizes the instance-level Brier Score. (5) Common IDs: retains only samples with question IDs common across all models. (6) Balance: balances the 3 groups (counterfactual, consistent, irrelevant) by downsampling consistent to match irrelevant.}",
        "\\label{tab:train-statistics}",
        "\\end{table*}",
    ])
    
    with output_path.open("w") as f:
        f.write("\n".join(latex_lines))
    
    print(f"\nLaTeX table written to: {output_path}")


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input.expanduser().resolve()
    base_output_dir: Path = args.output.expanduser().resolve()
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%m-%d-%H-%M")
    output_dir = base_output_dir.parent / f"{base_output_dir.name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    drop_pct = args.enable_drop if args.enable_drop is not None else 0.0
    
    # Process all files and collect statistics
    model_stats: Dict[str, FilterStats] = {}
    
    for json_path in sorted(input_dir.glob("*.json")):
        model_name = json_path.stem
        stats = process_file(
            json_path,
            output_dir / json_path.name,
            drop_pct=drop_pct,
            tolerate_mismatch=args.tolarate_mismatch,
        )
        model_stats[model_name] = stats

    # Step 4: Compute common IDs across all models and count responses
    model_ids = compute_common_ids(output_dir)
    common_ids: set = set()
    
    if model_ids:
        # Find intersection of all question IDs
        if len(model_ids) > 1:
            common_ids = set.intersection(*model_ids.values())
        elif len(model_ids) == 1:
            common_ids = list(model_ids.values())[0]
        
        # Count questions (not responses) for common IDs only (Step 5)
        # Since Alignment selects 1 response per question, Common IDs should count questions, not responses
        for model_name in model_stats:
            # Count questions (items) with IDs in common_ids
            output_file = output_dir / f"{model_name}.json"
            if output_file.exists():
                with output_file.open("r") as f:
                    data = json.load(f)
                common_questions_count = 0
                for item in data:
                    item_id = item.get("id")
                    if item_id in common_ids:
                        common_questions_count += 1
                model_stats[model_name].after_common_ids = common_questions_count
            else:
                model_stats[model_name].after_common_ids = 0
        
        print(f"\nStep 5 - Common question IDs across all models: {len(common_ids)}")
        for model_name in model_stats:
            print(f"  {model_name}: {model_stats[model_name].after_common_ids} questions")
        
        # Step 6: Balance groups (downsample consistent to match irrelevant)
        # Use the first model as reference dataset
        reference_model_name = sorted(model_stats.keys())[0]
        reference_file = output_dir / f"{reference_model_name}.json"
        if reference_file.exists():
            with reference_file.open("r") as f:
                reference_data = json.load(f)
            reference_dataset = {item.get("id"): item for item in reference_data if item.get("id")}
            
            balanced_ids = balance_groups(list(common_ids), reference_dataset)
            balanced_count = len(balanced_ids)
            
            # All models have the same balanced count
            for model_name in model_stats:
                model_stats[model_name].after_balance = balanced_count
            
            print(f"\nStep 6 - Balanced groups: {balanced_count} questions")
            print(f"  (Downsampled consistent to match irrelevant count)")
        else:
            for model_name in model_stats:
                model_stats[model_name].after_balance = 0
    else:
        print("\nWarning: No models processed, cannot compute common IDs")
        for model_name in model_stats:
            model_stats[model_name].after_common_ids = 0
            model_stats[model_name].after_balance = 0

    # Generate LaTeX table
    latex_output_path = output_dir / "filter_statistics.tex"
    generate_latex_table(model_stats, latex_output_path)

    # Print summary
    aggregated = FilterStats()
    for stats in model_stats.values():
        aggregated.kept_prompts += stats.kept_prompts
        aggregated.kept_responses += stats.kept_responses
        aggregated.dropped_length += stats.dropped_length
        aggregated.total_responses += stats.total_responses

    print(
        f"\nFinished. Total kept responses: {aggregated.kept_responses}, "
        f"across {aggregated.kept_prompts} prompts. "
        f"Length-based drops: {aggregated.dropped_length}."
    )


if __name__ == "__main__":
    main()
