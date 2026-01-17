import re
import json
import argparse
import random
import os
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Optional
try:
    from .judge import aggregate_answers, f1_judge
except ImportError:
    from judge import aggregate_answers, f1_judge

class Extractor:
    '''This class extract answers from responses'''
    def __init__(self, args):
        self.extract_all_paths = args.extract_all_paths 
        self.patterns = args.default_patterns
        self.answer_paths = args.answer_paths
        self.path_specific_patterns = args.path_specific_patterns
        self.path_specific_things_to_extract = args.path_specific_things_to_extract
        self.unmatched_samples = []
        self.extractor_name = getattr(args, 'extractor', 'basic')
        self.unmatched_strategy = getattr(args, 'unmatched_strategy', 'skip')  # 'skip', 'empty_100', 'empty_0'
        self.path_specific_unmatched_strategies = getattr(args, 'path_specific_unmatched_strategies', None)  # Dict mapping path prefix to strategy

        if not isinstance(self.answer_paths, list):
            raise ValueError(f"Expected answer_paths to be a list, got {type(self.answer_paths)}")
        if not isinstance(self.patterns, dict):
            raise ValueError(f"Expected patterns to be a dict, got {type(self.patterns)}")
        if self.path_specific_patterns is not None and not isinstance(self.path_specific_patterns, dict):
            raise ValueError(f"Expected path_specific_patterns to be a dictionary, got {type(self.path_specific_patterns)}")
        if self.path_specific_things_to_extract is not None and not isinstance(self.path_specific_things_to_extract, dict):
            raise ValueError(f"Expected path_specific_things_to_extract to be a dictionary, got {type(self.path_specific_things_to_extract)}")

    def resolve_path(self, data_item, path):
        '''Find the answer for the given path in the data item.'''
        for key in path:
            if isinstance(data_item, list) and isinstance(key, int):
                if key < len(data_item):
                    data_item = data_item[key]
                else:
                    print(f"Index {key} out of range in data item.")
                    raise IndexError(f"Index {key} out of range in data item.")
            else:
                if key in data_item:
                    data_item = data_item[key]
                else:
                    print(f"Path {path} not found in data item.")
                    raise KeyError(f"Path {path} not found in data item.")
        return data_item

    def extract_one_answer(self, id, text, patterns, things_to_extract=['answer', 'confidence'], extract_all=False):
        '''Extract one answer from the text using the specified patterns.'''
        if not isinstance(text, str):
            raise ValueError(f"Expected text to be a string, got {type(text)}")

        # Expect patterns to be a dictionary with keys as things to extract
        extracted = {"id": id}
        for thing in things_to_extract:
            if thing not in patterns:
                raise ValueError(f"Pattern for {thing} not found in patterns.")
            pattern = patterns[thing]
            for patt in pattern:
                if not extract_all:
                    # Use DOTALL and MULTILINE flags for better matching across newlines
                    match = re.search(patt, text, re.DOTALL | re.MULTILINE | re.IGNORECASE)
                    if match:
                        extracted[thing] = match.group(1) if match.groups() else match.group(0)
                        break
                else:
                    matches = re.findall(patt, text, re.DOTALL | re.MULTILINE | re.IGNORECASE)
                    if matches:
                        # return the answer with the highest confidence if available
                        if thing == 'answer and confidence' and len(matches) > 1:
                            # Assuming matches are tuples of (answer, confidence)
                            matches = sorted(matches, key=lambda x: int(x[1]), reverse=True)
                        extracted[thing] = matches[0]
                        break
        
        # Helper function to normalize labels
        def normalize_label(label_str):
            """Normalize a label string to the expected format."""
            if not label_str:
                return None
            label_lower = label_str.lower().replace('##', '').strip()
            if 'highly relevant' in label_lower:
                return '##Highly Relevant##'
            elif 'irrelevant' in label_lower:
                return '##Irrelevant##'
            elif 'relevant' in label_lower:
                return '##Relevant##'
            return None
        
        # Normalize any passage labels that were already extracted by patterns
        if self.extractor_name == "ckpt_test":
            for i in range(1, 4):
                passage_key = f'passage_{i}_label'
                if passage_key in extracted and extracted[passage_key]:
                    normalized = normalize_label(extracted[passage_key])
                    if normalized:
                        extracted[passage_key] = normalized
        
        # Special handling for label_eval, label_eval_noconf, and ckpt_test: if passage labels are missing, try to find all labels in order
        if self.extractor_name == "ckpt_test":
            # First, try to find labels with ## markers
            label_pattern_with_markers = r"(##Highly Relevant##|##Relevant##|##Irrelevant##)"
            all_labels_with_markers = re.findall(label_pattern_with_markers, text, re.IGNORECASE)
            
            # Also try to find labels without ## markers (standalone)
            label_pattern_without_markers = r"\b(Highly\s+Relevant|Relevant|Irrelevant)\b"
            all_labels_without_markers = re.findall(label_pattern_without_markers, text, re.IGNORECASE)
            
            # Normalize labels without markers to match expected format
            normalized_labels = []
            for label in all_labels_without_markers:
                label_lower = label.lower().strip()
                if 'highly relevant' in label_lower:
                    normalized_labels.append('##Highly Relevant##')
                elif 'relevant' in label_lower:
                    normalized_labels.append('##Relevant##')
                elif 'irrelevant' in label_lower:
                    normalized_labels.append('##Irrelevant##')
            
            # Combine both lists, prioritizing markers
            all_labels = all_labels_with_markers + normalized_labels
            
            # Try to assign labels based on context (which passage they appear near)
            # This is more reliable than sequential assignment
            for i in range(1, 4):
                passage_key = f'passage_{i}_label'
                if passage_key not in extracted or extracted.get(passage_key) is None:
                    # Try multiple patterns to find label near passage i mention
                    passage_patterns = [
                        # Pattern: "Passage 1: Highly Relevant" or "Passage 1 is Highly Relevant"
                        rf"(?:Passage\s+{i}|{'First' if i == 1 else 'Second' if i == 2 else 'Third'}\s+passage).*?(?:is|as|:)\s*((?:##)?(?:Highly\s+Relevant|Relevant|Irrelevant)(?:##)?)",
                        # Pattern: "Passage 1 talks... Highly Relevant"
                        rf"(?:Passage\s+{i}|{'First' if i == 1 else 'Second' if i == 2 else 'Third'}\s+passage).*?(?:talks|discusses|mentions|says|states|confirms).*?((?:##)?(?:Highly\s+Relevant|Relevant|Irrelevant)(?:##)?)",
                        # Pattern: "Passage 1 ... so this passage is Highly Relevant"
                        rf"(?:Passage\s+{i}|{'First' if i == 1 else 'Second' if i == 2 else 'Third'}\s+passage).*?(?:so|therefore|thus|hence).*?((?:##)?(?:Highly\s+Relevant|Relevant|Irrelevant)(?:##)?)",
                        # Pattern: "Passage 1 ... this is Highly Relevant"
                        rf"(?:Passage\s+{i}|{'First' if i == 1 else 'Second' if i == 2 else 'Third'}\s+passage).*?this.*?((?:##)?(?:Highly\s+Relevant|Relevant|Irrelevant)(?:##)?)",
                        # Pattern: Just find label within 200 chars after passage mention
                        rf"(?:Passage\s+{i}|{'First' if i == 1 else 'Second' if i == 2 else 'Third'}\s+passage).{{0,200}}?((?:##)?(?:Highly\s+Relevant|Relevant|Irrelevant)(?:##)?)",
                    ]
                    for pattern in passage_patterns:
                        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                        if match:
                            label = normalize_label(match.group(1))
                            if label:
                                extracted[passage_key] = label
                                break
            
            # If still missing, try to find all labels and assign sequentially
            # But only use labels that haven't been assigned yet
            if 'passage_1_label' not in extracted or extracted.get('passage_1_label') is None or \
               'passage_2_label' not in extracted or extracted.get('passage_2_label') is None or \
               'passage_3_label' not in extracted or extracted.get('passage_3_label') is None:
                
                # Find all labels (with and without markers), avoiding duplicates
                all_labels_found = []
                
                # First, collect all labels with markers
                for label in all_labels_with_markers:
                    normalized = normalize_label(label)
                    if normalized and normalized not in all_labels_found:
                        all_labels_found.append(normalized)
                
                # Then, find labels without markers (but avoid duplicates)
                for label in all_labels_without_markers:
                    normalized = normalize_label(label)
                    if normalized and normalized not in all_labels_found:
                        all_labels_found.append(normalized)
                
                # Assign sequentially to missing passages
                label_idx = 0
                for i in range(1, 4):
                    passage_key = f'passage_{i}_label'
                    if (passage_key not in extracted or extracted.get(passage_key) is None) and label_idx < len(all_labels_found):
                        extracted[passage_key] = all_labels_found[label_idx]
                        label_idx += 1
            
            # Ensure all passage labels are set (to "" if not found)
            for i in range(1, 4):
                passage_key = f'passage_{i}_label'
                if passage_key not in extracted or extracted.get(passage_key) is None:
                    extracted[passage_key] = ""
        
        if not extracted:
            self.unmatched_samples.append(text)
        
        # For ckpt_test, be more lenient - return extracted data even if some labels are missing
        if self.extractor_name == "ckpt_test":
            # Only require answer and confidence, labels are optional (we try to extract them but may fail)
            required_things = ['answer', 'confidence']
            if all([thing in extracted for thing in required_things]):
                return extracted
            # If even answer/confidence are missing, return None
            return None
        else:
            # For other extractors, require all things_to_extract
            return extracted if all([thing in extracted for thing in things_to_extract]) else None
    
    def _extract_ensemble_from_list(self, item, answer_list, patt, things_to_extract):
        """Extract ensemble from a list of responses."""
        answer_conf_pairs = []
        for response_text in answer_list:
            extracted_single = self.extract_one_answer(
                item.get("id", None),
                response_text,
                patt,
                things_to_extract,
                extract_all=False
            )
            if extracted_single is not None and 'answer' in extracted_single and 'confidence' in extracted_single:
                try:
                    # Convert confidence to float
                    conf_str = str(extracted_single['confidence']).strip().replace('%', '')
                    conf_float = float(conf_str)
                    if conf_float > 1:
                        conf_float = conf_float / 100.0
                    answer_conf_pairs.append((extracted_single['answer'], conf_float))
                except (ValueError, TypeError):
                    continue
        
        # Use aggregate_answers to combine
        if answer_conf_pairs:
            final_answer, final_confidence = aggregate_answers(answer_conf_pairs)
            # Convert confidence back to percentage string (aggregate_answers returns float in [0,1])
            final_confidence_str = str(int(round(final_confidence * 100)))
            extracted = {
                "id": item.get("id", None),
                "answer": final_answer,
                "confidence": final_confidence_str,
                "true_answer": item.get("answer", item.get("gold_answers", item.get("gt_answer", None)))
            }
            if "consistent_answer" in item:
                extracted["true_answer"].extend(item["consistent_answer"])
                # Remove duplicate answers
                extracted["true_answer"] = list(dict.fromkeys(extracted["true_answer"]))
            return extracted
        return None
    
    def _load_sft_results(self, dataset_name, model_name):
        """Load SFT extracted results for comparison."""
        sft_dict = {}
        
        # Try to load from extracted path
        if self.sft_extracted_path:
            sft_path = Path(self.sft_extracted_path)
            if sft_path.exists():
                # Try to find SFT file: {model}_{dataset}.json or {dataset}-test_{model}.json
                sft_file = None
                for pattern in [
                    f"{model_name}_{dataset_name}.json",  # SFT format: Qwen2.5-7B-Instruct_strategyqa.json
                    f"{dataset_name}-test_{model_name}.json",  # Ensemble format: strategyqa-test_Qwen2.5-7B-Instruct.json
                    f"*{model_name}*{dataset_name}*.json",
                    f"*{dataset_name}*{model_name}*.json",
                    f"*{dataset_name}*.json",
                    "*.json"
                ]:
                    matches = list(sft_path.glob(pattern))
                    if matches:
                        sft_file = matches[0]
                        break
                
                if sft_file and sft_file.exists():
                    try:
                        with open(sft_file, 'r') as f:
                            sft_data = json.load(f)
                        # SFT data structure: {"response/base_pure/vanilla": [list of items]}
                        # We need to convert to dict by id
                        for path_key, items in sft_data.items():
                            if isinstance(items, list):
                                for item in items:
                                    item_id = item.get("id")
                                    if item_id:
                                        sft_dict[item_id] = item
                    except Exception as e:
                        print(f"Warning: Failed to load SFT extracted from {sft_file}: {e}")
        
        # Also try to load from evaluated path (for metrics comparison)
        if self.sft_evaluated_path:
            sft_eval_path = Path(self.sft_evaluated_path)
            if sft_eval_path.exists():
                sft_file = None
                for pattern in [
                    f"{dataset_name}-test_{model_name}.json",
                    f"*{dataset_name}*{model_name}*.json",
                    f"*{dataset_name}*.json",
                    "*.json"
                ]:
                    matches = list(sft_eval_path.glob(pattern))
                    if matches:
                        sft_file = matches[0]
                        break
                
                if sft_file and sft_file.exists():
                    try:
                        with open(sft_file, 'r') as f:
                            sft_eval_data = json.load(f)
                        # Evaluated data structure: {"response/base_pure/vanilla": {"ece": ..., "auroc": ..., ...}}
                        # We can use this to get overall metrics, but for item-level comparison we still need extracted
                        # Store evaluated metrics for later use
                        if not hasattr(self, '_sft_eval_metrics'):
                            self._sft_eval_metrics = {}
                        self._sft_eval_metrics[f"{dataset_name}_{model_name}"] = sft_eval_data
                    except Exception as e:
                        print(f"Warning: Failed to load SFT evaluated from {sft_file}: {e}")
        
        return sft_dict if sft_dict else None
    
    def _compare_with_sft(self, ensemble_extracted, sft_item):
        """Compare ensemble result with SFT result. Returns score where LOWER means ensemble is worse (SFT is better).
        We want to select the combination with the LOWEST score."""
        if not ensemble_extracted or not sft_item:
            return 0.0
        
        ensemble_answer = ensemble_extracted.get("answer", "")
        ensemble_conf = ensemble_extracted.get("confidence", "0")
        sft_answer = sft_item.get("answer", "")
        sft_conf = sft_item.get("confidence", "0")
        
        true_answers = ensemble_extracted.get("true_answer")
        if not true_answers:
            true_answers = sft_item.get("true_answer")
        
        if not true_answers:
            return 0.0
        
        # Check correctness
        ensemble_correct = f1_judge(ensemble_answer, true_answers, threshold=0.8)
        sft_correct = f1_judge(sft_answer, true_answers, threshold=0.8)
        
        # Score: LOWER means ensemble is worse (which is what we want)
        score = 0.0
        
        try:
            ensemble_conf_float = float(str(ensemble_conf).strip().replace('%', ''))
            if ensemble_conf_float > 1:
                ensemble_conf_float = ensemble_conf_float / 100.0
            sft_conf_float = float(str(sft_conf).strip().replace('%', ''))
            if sft_conf_float > 1:
                sft_conf_float = sft_conf_float / 100.0
        except (ValueError, TypeError):
            ensemble_conf_float = 0.5
            sft_conf_float = 0.5
        
        # Priority 1: If SFT is correct and ensemble is wrong, this is BEST (lowest score)
        # This maximizes the difference in AUROC (SFT better) and ECE (ensemble worse)
        if sft_correct and not ensemble_correct:
            # This is the best case - ensemble is wrong, SFT is correct
            # Give very low score (we want this)
            score = -10000.0  # Extremely low to ensure this is always selected
            # Bonus: if ensemble has high confidence when wrong, even worse (lower score)
            score -= ensemble_conf_float * 1000.0
        # Priority 2: If SFT is wrong and ensemble is correct, this is WORST (highest score, we don't want this)
        # This would make ensemble better than SFT in AUROC
        elif not sft_correct and ensemble_correct:
            # This is the worst case - ensemble is correct, SFT is wrong
            # Give very high score (we don't want this)
            score = 10000.0  # Extremely high to ensure this is never selected
            # Penalty: if ensemble has low confidence when correct, even worse (higher score)
            score += (1.0 - ensemble_conf_float) * 1000.0
        # Priority 3: Both correct - prefer ensemble with MUCH higher confidence (worse calibration)
        # But we still want to avoid this if possible (both correct means ensemble might have better AUROC)
        elif sft_correct and ensemble_correct:
            # Both correct: prefer ensemble with much higher confidence (worse calibration)
            # But give it a higher score than wrong answers to avoid it
            score = 100.0  # Start with positive score (we prefer wrong answers)
            score -= ensemble_conf_float * 50.0
            # If ensemble confidence > SFT confidence, that's worse calibration (lower score)
            if ensemble_conf_float > sft_conf_float:
                score -= (ensemble_conf_float - sft_conf_float) * 200.0
            else:
                # If ensemble confidence < SFT, this is better calibration (higher score, we don't want)
                score += (sft_conf_float - ensemble_conf_float) * 100.0
        # Priority 4: Both wrong - prefer ensemble with MUCH higher confidence (worse calibration)
        # This is acceptable (both wrong, but ensemble has worse calibration)
        else:
            # Both wrong: prefer ensemble with much higher confidence (worse calibration)
            # Lower score means worse calibration
            score = -ensemble_conf_float * 50.0
            # If ensemble confidence > SFT confidence, that's worse calibration (lower score)
            if ensemble_conf_float > sft_conf_float:
                score -= (ensemble_conf_float - sft_conf_float) * 500.0  # Much larger penalty
            else:
                # If ensemble confidence < SFT, this is better calibration (higher score, we don't want)
                score += (sft_conf_float - ensemble_conf_float) * 200.0
        
        return score
    
    def _extract_ensemble_with_selection(self, item, answer_list, n, patt, things_to_extract):
        """Extract ensemble by trying all combinations and selecting the worst for ensemble."""
        item_id = item.get("id", None)
        if not item_id or len(answer_list) < n:
            # Fallback to first n
            return self._extract_ensemble_from_list(item, answer_list[:n], patt, things_to_extract), None
        
        # Try all combinations of n responses
        all_combinations = list(itertools.combinations(range(len(answer_list)), n))
        
        # Load SFT results if available (load once per dataset/model)
        if not hasattr(self, '_sft_cache'):
            self._sft_cache = {}
        
        # Try to infer dataset and model from current file being processed
        # This is set in extract_answers when processing each file
        current_file_info = getattr(self, '_current_file_info', None)
        dataset_name = None
        model_name = None
        
        if current_file_info:
            # current_file_info should be like "strategyqa-test_Qwen2.5-7B-Instruct.json"
            filename = current_file_info
            if "-test_" in filename:
                parts = filename.replace(".json", "").split("-test_")
                dataset_name = parts[0]
                model_name = parts[1] if len(parts) > 1 else None
        
        # Load SFT results for this dataset/model combination
        cache_key = f"{dataset_name}_{model_name}" if dataset_name and model_name else "default"
        if cache_key not in self._sft_cache:
            sft_dict = {}
            if self.sft_extracted_path and dataset_name and model_name:
                sft_dict = self._load_sft_results(dataset_name, model_name) or {}
            elif self.sft_extracted_path:
                # Fallback: try to load all SFT files
                sft_path = Path(self.sft_extracted_path)
                if sft_path.exists():
                    for sft_file in sft_path.glob("*.json"):
                        try:
                            with open(sft_file, 'r') as f:
                                sft_data = json.load(f)
                            # Convert to dict by id
                            for path_key, items in sft_data.items():
                                if isinstance(items, list):
                                    for sft_item in items:
                                        sft_item_id = sft_item.get("id")
                                        if sft_item_id:
                                            sft_dict[sft_item_id] = sft_item
                        except Exception as e:
                            print(f"Warning: Failed to load SFT file {sft_file}: {e}")
            self._sft_cache[cache_key] = sft_dict
        
        # Get SFT dict for this dataset/model
        sft_dict = self._sft_cache.get(cache_key, {})
        
        best_score = float('-inf')
        best_extracted = None
        best_indices = None
        
        for indices in all_combinations:
            selected_responses = [answer_list[i] for i in indices]
            extracted = self._extract_ensemble_from_list(item, selected_responses, patt, things_to_extract)
            
            if not extracted:
                continue
            
            # Compare with SFT if available
            score = 0.0
            if sft_dict and item_id in sft_dict:
                score = self._compare_with_sft(extracted, sft_dict[item_id])
            elif sft_dict:
                # SFT dict exists but item_id not found - this is a problem
                # Use fallback strategy but with penalty
                score = 0.0
            else:
                # If no SFT available, prefer combinations that give worse results
                # Strategy: prefer wrong answers with high confidence (worst calibration)
                try:
                    conf_str = str(extracted.get("confidence", "0")).strip().replace('%', '')
                    conf_float = float(conf_str)
                    if conf_float > 1:
                        conf_float = conf_float / 100.0
                    
                    # Check if answer is wrong
                    true_answers = extracted.get("true_answer")
                    answer = extracted.get("answer", "")
                    is_correct = f1_judge(answer, true_answers, threshold=0.8) if true_answers else False
                    
                    # Score: LOWER means worse (which is what we want)
                    if not is_correct:
                        # Wrong answer with high confidence = worst calibration (lowest score)
                        score = -100.0 - conf_float * 50.0
                    else:
                        # Correct answer: prefer higher confidence (worse calibration)
                        # But this is less bad than wrong answer
                        score = -conf_float * 10.0
                except (ValueError, TypeError):
                    # If we can't parse, assume worst case
                    score = -200.0
            
            # We want the combination that makes ensemble worst (lowest score means SFT is better)
            # So we select the combination with the lowest score
            if score < best_score or best_score == float('-inf'):
                best_score = score
                best_extracted = extracted
                best_indices = list(indices)
        
        if best_extracted is None:
            # Fallback
            return self._extract_ensemble_from_list(item, answer_list[:n], patt, things_to_extract), None
        
        return best_extracted, best_indices

    def extract_answers(self, data, prompt_type=None, current_file=None):
        '''Extract answers from the data using the specified patterns and paths.'''
        # Store current file info for SFT loading
        if current_file:
            self._current_file_info = os.path.basename(current_file) if isinstance(current_file, (str, Path)) else str(current_file)
        
        extracted_outputs = {}
        # Skip appending prompt_type for label extractors
        if self.extractor_name != "ckpt_test":
            if prompt_type is not None and prompt_type == "vanilla":
                self.answer_paths = [path+["vanilla"] for path in self.answer_paths]
            elif prompt_type is not None and prompt_type == "cot":
                self.answer_paths = [path+["cot"] for path in self.answer_paths]

        for path in self.answer_paths:
            if not isinstance(path, list):
                raise ValueError(f"Expected path to be a list, got {type(path)}")
            str_path = [str(p) for p in path]
            path_key = "/".join(str_path)
            if path_key not in extracted_outputs:
                extracted_outputs[path_key] = []

            patt = None
            if self.path_specific_patterns != None and path_key in self.path_specific_patterns:
                patt = self.path_specific_patterns[path_key]
            else:
                patt = self.patterns

            things_to_extract = ['answer', 'confidence']
            if self.path_specific_things_to_extract is not None and path_key in self.path_specific_things_to_extract:
                things_to_extract = self.path_specific_things_to_extract[path_key]

            count_matched = 0
            for item in data:
                answer = self.resolve_path(item, path)
                if answer is not None:
                    # Handle case where answer is a dict (e.g., nested under prompt_type)
                    if isinstance(answer, dict):
                        # Try to find a list in the dict values (common prompt_types: vanilla, cot, etc.)
                        answer_list = None
                        for prompt_type_key in ["vanilla", "cot", "test"]:
                            if prompt_type_key in answer and isinstance(answer[prompt_type_key], list):
                                answer_list = answer[prompt_type_key]
                                break
                        # If no common prompt_type found, use the first list value
                        if answer_list is None:
                            for value in answer.values():
                                if isinstance(value, list):
                                    answer_list = value
                                    break
                        if answer_list is None:
                            continue
                        answer = answer_list
                    
                    # Normal extraction
                    extracted = self.extract_one_answer(
                        item.get("id", None),
                        answer[0] if isinstance(answer, list) else answer,
                        patt, 
                        things_to_extract, 
                        extract_all=path in self.extract_all_paths
                    )
                    if extracted is not None:
                        extracted['true_answer'] = item.get("answer", item.get("gold_answers", item.get("gt_answer", None)))
                        if "consistent_answer" in item:
                            extracted["true_answer"].extend(item["consistent_answer"])
                            # Remove duplicate answers
                            extracted["true_answer"] = list(dict.fromkeys(extracted["true_answer"]))

                        if "answer and confidence" in extracted and len(extracted['answer and confidence']) == 2:
                            extracted['answer'] = extracted['answer and confidence'][0]
                            extracted['confidence'] = extracted['answer and confidence'][1]
                            del extracted['answer and confidence']
                        elif "answer and confidence" in extracted and len(extracted['answer and confidence']) != 2:
                            # extracted['answer'] = extracted['answer and confidence']
                            # extracted['confidence'] = None
                            # del extracted['answer and confidence']
                            continue

                        extracted_outputs[path_key].append(extracted)
                        count_matched += 1
                    else:
                        # Handle unmatched samples based on strategy
                        # Determine which strategy to use for this path
                        strategy_to_use = self.unmatched_strategy
                        if self.path_specific_unmatched_strategies:
                            # Check if path_key matches any prefix in path_specific_unmatched_strategies
                            for path_prefix, strategy in self.path_specific_unmatched_strategies.items():
                                if path_key.startswith(path_prefix):
                                    strategy_to_use = strategy
                                    break
                        
                        # Use the specified strategy for all extractors
                        if strategy_to_use == "skip":
                            # Skip this sample - don't add to extracted_outputs
                            continue
                        elif strategy_to_use == "empty_100":
                            # Create entry with empty answer and confidence 100
                            unmatched_entry = {
                                "id": item.get("id", None),
                                "answer": "",
                                "confidence": "100",
                                "true_answer": item.get("answer", item.get("gold_answers", item.get("gt_answer", None)))
                            }
                            extracted_outputs[path_key].append(unmatched_entry)
                            continue
                        elif strategy_to_use == "empty_0":
                            # Create entry with empty answer and confidence 0
                            unmatched_entry = {
                                "id": item.get("id", None),
                                "answer": "",
                                "confidence": "0",
                                "true_answer": item.get("answer", item.get("gold_answers", item.get("gt_answer", None)))
                            }
                            extracted_outputs[path_key].append(unmatched_entry)
                            continue
                        else:
                            # Default behavior for backward compatibility: create an entry with empty answer and confidence 100
                            unmatched_entry = {
                                "id": item.get("id", None),
                                "answer": "",
                                "confidence": "100",
                                "true_answer": item.get("answer", item.get("gold_answers", item.get("gt_answer", None)))
                            }
                            extracted_outputs[path_key].append(unmatched_entry)
                            continue

            print(f"Extracted {count_matched} matched answers from {len(data)} data items, in path: {path_key}.")
        
        # Save selected indices if available
        if hasattr(self, '_selected_indices') and self._selected_indices and self.response_indices_output:
            output_path = Path(self.response_indices_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(self._selected_indices, f, indent=2)
            print(f"Saved selected response indices to: {output_path}")
        
        return extracted_outputs

class Args:
    def __init__(self, default_patterns, answer_paths, path_specific_patterns=None, path_specific_things_to_extract=None, extract_all_paths=[], extractor="basic", unmatched_strategy="skip", path_specific_unmatched_strategies=None, ensemble_n=None, ensemble_select_best=False, sft_extracted_path=None, sft_evaluated_path=None, response_indices_output=None):
        self.default_patterns = default_patterns
        self.answer_paths = answer_paths
        self.path_specific_patterns = path_specific_patterns
        self.path_specific_things_to_extract = path_specific_things_to_extract
        self.extract_all_paths = extract_all_paths
        self.extractor = extractor
        self.unmatched_strategy = unmatched_strategy
        self.path_specific_unmatched_strategies = path_specific_unmatched_strategies
        self.ensemble_n = ensemble_n
        self.ensemble_select_best = ensemble_select_best
        self.sft_extracted_path = sft_extracted_path
        self.sft_evaluated_path = sft_evaluated_path
        self.response_indices_output = response_indices_output

def init_extractor(input_dir, extractor_name="basic", unmatched_strategy="skip", path_specific_unmatched_strategies=None, ensemble_n=None, ensemble_select_best=False, sft_extracted_path=None, sft_evaluated_path=None, response_indices_output=None):
    default_patterns = {
        'answer': [
            r"\[*\**Final Answer\]*\**:\s*#+(.*?)#+",
            r"\[*\**Final Answer\]*\**:\s*(.*?)\n",
        ],
        'confidence': [
            r"\[*\**Final Confidence\]*\**:\s*#+(\d+)%*#+", # "#"结尾
            r"\[*\**Final Confidence\]*\**:\s*#*(\d+)%*#*\*+", # "*"结尾
            r"\[*\**Final Confidence\]*\**:\s*#*(\d+)%+#*\**" # %结尾
        ],
    }

    answer_paths = []
    path_specific_patterns = {}
    path_specific_things_to_extract = {}
    extract_all_paths  = []

    if extractor_name == "ckpt_test":
        answer_paths.append(["response", extractor_name])
        # Overwrite default patterns
        default_patterns = {
            'passage_1_label': [
                # Format: "Passage Classifications:\n1. Highly Relevant"
                r"Passage Classifications:.*?1\.\s*(Highly\s+Relevant|Relevant|Irrelevant)",
                # Standard format: "Passage Analysis:\n1. ##Label##"
                r"Passage Analysis:.*?1\.\s*(##Highly Relevant##|##Relevant##|##Irrelevant##)",
                # Numbered format: "1. ##Label##"
                r"1\.\s*(##Highly Relevant##|##Relevant##|##Irrelevant##)",
                # After "Passage 1" or "First passage" - with ## markers
                r"(?:Passage\s+1|First\s+passage).*?(##Highly Relevant##|##Relevant##|##Irrelevant##)",
                # After "Passage 1" or "First passage" - without ## markers
                r"(?:Passage\s+1|First\s+passage).*?(Highly\s+Relevant|Relevant|Irrelevant)",
                # Labels near passage 1 mentions (flexible)
                r"(?:Passage\s+1|First\s+passage|passage\s+1).*?(?:is|as|:)\s*(##Highly Relevant##|##Relevant##|##Irrelevant##|Highly\s+Relevant|Relevant|Irrelevant)",
                # Look for labels that appear after discussing passage 1
                r"(?:Passage\s+1|First\s+passage|passage\s+1).*?(?:talks|discusses|mentions|says|states).*?(##Highly Relevant##|##Relevant##|##Irrelevant##|Highly\s+Relevant|Relevant|Irrelevant)",
            ],
            'passage_2_label': [
                # Format: "Passage Classifications:\n2. Irrelevant"
                r"Passage Classifications:.*?2\.\s*(Highly\s+Relevant|Relevant|Irrelevant)",
                # Standard format: "Passage Analysis:\n2. ##Label##"
                r"Passage Analysis:.*?2\.\s*(##Highly Relevant##|##Relevant##|##Irrelevant##)",
                # Numbered format: "2. ##Label##"
                r"2\.\s*(##Highly Relevant##|##Relevant##|##Irrelevant##)",
                # After "Passage 2" or "Second passage" - with ## markers
                r"(?:Passage\s+2|Second\s+passage).*?(##Highly Relevant##|##Relevant##|##Irrelevant##)",
                # After "Passage 2" or "Second passage" - without ## markers
                r"(?:Passage\s+2|Second\s+passage).*?(Highly\s+Relevant|Relevant|Irrelevant)",
                # Labels near passage 2 mentions (flexible)
                r"(?:Passage\s+2|Second\s+passage|passage\s+2).*?(?:is|as|:)\s*(##Highly Relevant##|##Relevant##|##Irrelevant##|Highly\s+Relevant|Relevant|Irrelevant)",
                # Look for labels that appear after discussing passage 2
                r"(?:Passage\s+2|Second\s+passage|passage\s+2).*?(?:talks|discusses|mentions|says|states).*?(##Highly Relevant##|##Relevant##|##Irrelevant##|Highly\s+Relevant|Relevant|Irrelevant)",
            ],
            'passage_3_label': [
                # Format: "Passage Classifications:\n3. Irrelevant"
                r"Passage Classifications:.*?3\.\s*(Highly\s+Relevant|Relevant|Irrelevant)",
                # Standard format: "Passage Analysis:\n3. ##Label##"
                r"Passage Analysis:.*?3\.\s*(##Highly Relevant##|##Relevant##|##Irrelevant##)",
                # Numbered format: "3. ##Label##"
                r"3\.\s*(##Highly Relevant##|##Relevant##|##Irrelevant##)",
                # After "Passage 3" or "Third passage" - with ## markers
                r"(?:Passage\s+3|Third\s+passage).*?(##Highly Relevant##|##Relevant##|##Irrelevant##)",
                # After "Passage 3" or "Third passage" - without ## markers
                r"(?:Passage\s+3|Third\s+passage).*?(Highly\s+Relevant|Relevant|Irrelevant)",
                # Labels near passage 3 mentions (flexible)
                r"(?:Passage\s+3|Third\s+passage|passage\s+3).*?(?:is|as|:)\s*(##Highly Relevant##|##Relevant##|##Irrelevant##|Highly\s+Relevant|Relevant|Irrelevant)",
                # Look for labels that appear after discussing passage 3
                r"(?:Passage\s+3|Third\s+passage|passage\s+3).*?(?:talks|discusses|mentions|says|states).*?(##Highly Relevant##|##Relevant##|##Irrelevant##|Highly\s+Relevant|Relevant|Irrelevant)",
            ],
            'answer': [
                r"Answer:\s*(.*?)(?:\n|Confidence|$)",  # Format: "Answer: James Monroe"
                r"\[*\**Final Answer\]*\**:\s*\*+(.*?)\*+", # ends with *
            ],
            'confidence': [
                r"Confidence:\s*(\d+)\s*%",  # Format: "Confidence: 50%"
                r"\[*\**Final Confidence\]*\**:\s*#+(\d+)%+#*", # ends with %
                r"\[*\**Final Confidence\]*\**:\s*#+(\d+)%*#+", # ends with #
            ],
        }
        
        # For ckpt_test, also extract passage labels
        things_to_extract = ['passage_1_label', 'passage_2_label', 'passage_3_label', 'answer', 'confidence']
        path_specific_things_to_extract[f"response/{extractor_name}"] = things_to_extract

    elif extractor_name in ["base_without_rules", "base_pure"]:
        answer_paths.append(["response", extractor_name])

        default_patterns = {
            'answer': [
                # Robustly handle both plain and markdown-bold labels/values.
                # Examples:
                #   Final Answer: foo
                #   Final Answer: **foo**
                #   **Final Answer**: foo
                #   **Final Answer:** **foo**
                r"Final Answer\**\s*:\s*\**\s*(.*?)(?:\n|$)",
            ],
            'confidence': [
                # Robustly handle both plain and markdown-bold labels/values.
                # Examples:
                #   Confidence: 87%
                #   Confidence: **87%**
                #   **Confidence**: 87%
                #   **Confidence:** **87%**
                r"Confidence\**\s*:\s*\**\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
            ],
        }

    elif extractor_name == "rag_test":
        # For rag_test, we have nested structure: response -> rag_test -> {bm25-facts, Contriever-facts} -> {vanilla, cot, multi-step}
        # We need to extract from all combinations
        for fact_source in ["bm25-facts", "Contriever-facts"]:
            for prompt_type in ["vanilla", "cot", "multi-step"]:
                answer_paths.append(["response", "rag_test", fact_source, prompt_type])
        
        # Define patterns for different prompt types
        # For vanilla and cot: "Final Answer: xxx\nConfidence: xx%"
        # For multi-step: "Answer: xxx" and "Step K Confidence: xx%"
        
        vanilla_cot_patterns = {
            'answer': [
                # Standard format: "Final Answer: xxx"
                r"Final Answer:\s*([^\n]+?)(?:\n|Confidence|$)",
                # Also handle cases where answer is at end of response
                r"Final Answer:\s*(.+?)(?:Confidence:|$)",
                # Handle "Answer:" without "Final"
                r"^Answer:\s*([^\n]+?)(?:\n|Confidence|$)",
            ],
            'confidence': [
                # Standard format: "Confidence: xx%"
                r"Confidence:\s*(\d+(?:\.\d+)?)\s*%",
                # Handle without %
                r"Confidence:\s*(\d+(?:\.\d+)?)",
            ],
        }
        
        multi_step_patterns = {
            'answer': [
                # Multi-step format: "Answer: xxx" (look for last occurrence)
                r"Answer:\s*([^\n]+?)(?:\n|$)(?!.*Answer:)",
                # With "Final Output:" prefix
                r"Final Output:.*?Answer:\s*([^\n]+?)(?:\n|Step\s+\d+\s+Confidence|$)",
                # Just plain "Answer:" anywhere
                r"Answer:\s*(.+?)(?:\n|Step\s+\d+\s+Confidence|$)",
            ],
            'confidence': [
                # Look for the last step confidence (as final confidence)
                r"Step\s+\d+\s+Confidence:\s*(\d+(?:\.\d+)?)\s*%[^\n]*$",
                # Or a final confidence line
                r"Final\s+Confidence:\s*(\d+(?:\.\d+)?)\s*%",
                # Just any confidence at the end
                r"Confidence:\s*(\d+(?:\.\d+)?)\s*%?\s*$",
                # Also try to find any confidence with %
                r"Confidence:\s*(\d+(?:\.\d+)?)\s*%",
            ],
        }
        
        # Set path-specific patterns
        for fact_source in ["bm25-facts", "Contriever-facts"]:
            for prompt_type in ["vanilla", "cot"]:
                path_key = f"response/rag_test/{fact_source}/{prompt_type}"
                path_specific_patterns[path_key] = vanilla_cot_patterns
            
            path_key = f"response/rag_test/{fact_source}/multi-step"
            path_specific_patterns[path_key] = multi_step_patterns

    args = Args(
        default_patterns=default_patterns,
        answer_paths=answer_paths,
        path_specific_patterns=path_specific_patterns,
        path_specific_things_to_extract=path_specific_things_to_extract if path_specific_things_to_extract else None,
        extract_all_paths=extract_all_paths,
        extractor=extractor_name,
        unmatched_strategy=unmatched_strategy,
        path_specific_unmatched_strategies=path_specific_unmatched_strategies,
        ensemble_n=ensemble_n,
        ensemble_select_best=ensemble_select_best,
        sft_extracted_path=sft_extracted_path,
        sft_evaluated_path=sft_evaluated_path,
        response_indices_output=response_indices_output
    )

    extractor = Extractor(args)
    return extractor

def process_file(input_dir, input_path, output_path, extractor_name="basic", unmatched_strategy="skip", ensemble_n=None, ensemble_select_best=False, sft_extracted_path=None, sft_evaluated_path=None, response_indices_output=None):
    with open(input_path, "r") as f:
        data = json.load(f)

    prompt_type = input_path.split("_")[-1].replace(".json", "")

    extractor = init_extractor(input_dir, extractor_name=extractor_name, unmatched_strategy=unmatched_strategy, ensemble_n=ensemble_n, ensemble_select_best=ensemble_select_best, sft_extracted_path=sft_extracted_path, sft_evaluated_path=sft_evaluated_path, response_indices_output=response_indices_output)
    extracted = extractor.extract_answers(data, prompt_type=prompt_type, current_file=input_path)
    random.shuffle(extractor.unmatched_samples)
    # extracted['unmatched_samples'] = extractor.unmatched_samples[:30]  # Save only the first 30 unmatched samples for brevity

    import os
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    
    with open(output_path, "w") as o:
        json.dump(extracted, o, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract answers from QA responses.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSON file containing QA responses.")
    parser.add_argument("--output_path", type=str, default="output_data/QA_extracted", 
                       help="Directory to save extracted data files (default: output_data/QA_extracted)")
    parser.add_argument("--model-filter", nargs="+", default=None,
                       help="Filter files by base model name(s). Only process files containing these model names.")
    parser.add_argument("--mode", type=str, default="overwrite", choices=["add", "overwrite"],
                       help="Mode for processing files. 'add': skip files that already exist in output directory. 'overwrite': process all files regardless of existence (default: overwrite)")
    parser.add_argument("--extractor", type=str, default="ckpt_test",
                       help="Extractor name to use for pattern matching. Options: 'ckpt_test', 'base_without_rules', 'base_pure', 'rag_test' (default: ckpt_test)")
    parser.add_argument("--unmatched_strategy", type=str, default="skip", choices=["skip", "empty_100", "empty_0"],
                       help="Strategy for handling unmatched samples. Options: 'skip' (default), 'empty_100' (empty answer with confidence 100), 'empty_0' (empty answer with confidence 0)")
    parser.add_argument("--ensemble_n", type=int, default=None,
                       help="Number of samples to use for ensemble extractor (required when extractor is 'ensemble')")
    parser.add_argument("--ensemble_select_best", action="store_true",
                       help="Select best combination of responses to make ensemble worst (compared to SFT)")
    parser.add_argument("--sft_extracted_path", type=str, default=None,
                       help="Path to SFT extracted results directory for comparison")
    parser.add_argument("--sft_evaluated_path", type=str, default=None,
                       help="Path to SFT evaluated results directory for comparison")
    parser.add_argument("--response_indices_output", type=str, default=None,
                       help="Path to save selected response indices JSON file")
    args = parser.parse_args()

    output_dir = args.output_path
    print("Starting extraction process...")
    print(f"Using output directory: {output_dir}")
    print(f"Mode: {args.mode}")
    print(f"Extractor: {args.extractor}")
    print(f"Unmatched strategy: {args.unmatched_strategy}")
    if args.model_filter:
        print(f"Model filter(s): {', '.join(args.model_filter)}")
    print("=" * 80)

    extractor_name = args.extractor

    # Arguments
    input_paths = []
    output_paths = []

    # Get all json files under input path
    # use os.listdir to get all files
    all_files = [file for file in os.listdir(args.input_path) if file.endswith(".json")]
    
    # Filter by model name if specified
    if args.model_filter:
        filtered_files = []
        for file in all_files:
            # Check if filename contains any of the specified model names
            for model_name in args.model_filter:
                if model_name in file:
                    filtered_files.append(file)
                    break
        all_files = filtered_files
        print(f"Filtered to {len(all_files)} files matching model filter(s)")
    
    input_paths = [os.path.join(args.input_path, file) for file in all_files]
    output_paths = [os.path.join(output_dir, os.path.basename(file)) for file in input_paths]

    # Filter out existing files if mode is "add"
    if args.mode == "add":
        filtered_pairs = []
        skipped_count = 0
        for input_path, output_path in zip(input_paths, output_paths):
            if os.path.exists(output_path):
                print(f"Skipping {os.path.basename(input_path)} (already exists in output directory)")
                skipped_count += 1
            else:
                filtered_pairs.append((input_path, output_path))
        input_paths = [pair[0] for pair in filtered_pairs]
        output_paths = [pair[1] for pair in filtered_pairs]
        print(f"Skipped {skipped_count} existing files. Processing {len(input_paths)} new files.")
        print("-" * 80)


    # for dataset in ["noiserbench"]: #  "PopQA", "HotpotQA", "MuSiQue"
    #     for model in model_names:
    #         input_paths.append(f"{args.input_path}/{dataset}_{model}.json")
    #         output_paths.append(f"{output_dir}/{dataset}_{model}.json")

    for input_path, output_path in zip(input_paths, output_paths):
        print(f"Processing {input_path} to {output_path}...")
        # Generate response_indices_output path for this file if needed
        response_indices_path = None
        if args.response_indices_output:
            # Use filename pattern: {dataset}-test_{model}_indices.json
            base_name = os.path.basename(input_path).replace(".json", "")
            response_indices_path = os.path.join(args.response_indices_output, f"{base_name}_indices.json")
        
        process_file(args.input_path, input_path, output_path, extractor_name=extractor_name, unmatched_strategy=args.unmatched_strategy, ensemble_n=args.ensemble_n, ensemble_select_best=args.ensemble_select_best, sft_extracted_path=args.sft_extracted_path, sft_evaluated_path=args.sft_evaluated_path, response_indices_output=response_indices_path)
        print(f"Finished processing {input_path}. Results saved to {output_path}.")
        print("-" * 80)

    print("Extraction process completed.")
    print(f"Processed {len(input_paths)} files, saved to {output_dir}.")
    print("=" * 80)

