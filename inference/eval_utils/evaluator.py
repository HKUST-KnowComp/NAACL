import json
import numpy as np
import random
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
# from metrics.answer import compute_f1
try:
    from .judge import f1_judge
except ImportError:
    from judge import f1_judge
import os


class Evaluator:
    def __init__(self, metrics, judge_func, model_name=None, extractor_name=None, original_data_path=None, has_confidence=True, has_labels=False):
        self.metrics = metrics # 'accuracy', 'ece', 'auroc' ...
        self.judge_func = judge_func  # Function to judge the correctness of answers
        self.decimal_places = 3
        self.model_name = model_name
        self.extractor_name = extractor_name  # Extractor name to differentiate tasks
        self.original_data_path = original_data_path  # Path to original data for label evaluation
        self.passage_type_map = None  # Will be populated if extractor_name is "ckpt_test" (has_labels=True)
        self.has_confidence = has_confidence
        self.has_labels = has_labels

    def evaluate(self, data):
        # Load original data if needed for label evaluation
        if self.has_labels and self.original_data_path:
            self._load_passage_type_map()
        
        # ensure all item has 'answer', 'confidence', 'true answer' keys
        for key in data.keys():
            if key == 'unmatched_samples':
                continue
            for item in data[key]:
                required_keys = ['answer', 'true_answer']
                if self.has_confidence:
                    required_keys.append('confidence')
                if self.has_labels:
                    label_keys = ['passage_1_label', 'passage_2_label', 'passage_3_label']
                    for label_key in label_keys:
                        if label_key not in item or item[label_key] is None:
                            item[label_key] = ""
                missing_keys = [k for k in required_keys if k not in item]
                if missing_keys:
                    raise ValueError(f"Missing keys in task {key}, data item {item.get('id', 'unknown')}: {missing_keys}")

        results = {}
        if self.model_name:
            results['model_name'] = self.model_name

        for task in data:
            if task == 'unmatched_samples':
                continue
            results[task] = {}
            for metric in self.metrics:
                results[task][metric] = self.process_metric(data[task], metric)

        return results
        
    def round_float_result(self, value):
        """
        Rounds a float value to the specified number of decimal places.
        Returns NaN values unchanged.
        """
        if isinstance(value, float):
            if np.isnan(value):
                return value
            return round(value, self.decimal_places)
        return value
        
    def process_metric(self, data, metric):
        if metric == 'accuracy':
            return self.round_float_result(self.compute_accuracy(data))
        elif metric == "label_accuracy":
            return self.round_float_result(self.compute_label_accuracy(data))
        elif metric == 'ece':
            return self.round_float_result(self.compute_ece(data))
        elif metric == 'auroc':
            return self.round_float_result(self.compute_auroc(data))
        elif metric == 'auprc':
            return self.round_float_result(self.compute_auprc(data))
        elif metric == "valid_sample_portion":
            return self.round_float_result(self.compute_sample_portion(data))
        elif metric == "label_extraction_portion":
            return self.compute_label_extraction_portion(data)
        elif metric == "ave_conf":
            return self.round_float_result(self.compute_ave_conf(data))
        elif metric == "reliability_diagram":
            return self.compute_reliability_diagram(data)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    def compute_reliability_diagram(self, data):
        """
        Returns a 2D list for reliability diagram:
        [ [#correct, #total] for each confidence bin: 0-10, 10-20, ..., 90-100 ]
        Boundaries: 0 in first bin, 100 in last bin.
        """
        num_bins = 10
        bins = [[0, 0] for _ in range(num_bins)]  # [correct, total] for each bin
        for item in data:
            pred = item.get('answer', None)
            truth = item.get('true_answer', None)
            confidence_str = item.get('confidence', None)
            if pred is None or truth is None or confidence_str is None:
                continue
            try:
                clean_str = str(confidence_str).strip().replace('%', '')
                confidence = float(clean_str)
                if confidence > 1:
                    confidence = min(confidence, 100.0)
                else:
                    confidence = confidence * 100.0
                # Clamp to [0, 100]
                confidence = min(max(confidence, 0.0), 100.0)
            except (ValueError, TypeError):
                continue
            # Bin index: 0 for [0,10), 1 for [10,20), ..., 9 for [90,100]
            if confidence == 100.0:
                bin_idx = num_bins - 1
            else:
                bin_idx = int(confidence // 10)
            bins[bin_idx][1] += 1  # total
            if self.judge_func(pred, truth):
                bins[bin_idx][0] += 1  # correct
        return bins

    def compute_sample_portion(self, data):
        total = len(data)
        count = 0
        for item in data:
            answer = item.get('answer', None)
            if answer:
                count += 1
        return f"{count}/{total} ({(count/total*100) if total > 0 else 0:.1f}%)"
    
    def compute_label_extraction_portion(self, data):
        """
        Compute the proportion of passage labels that were successfully extracted.
        Returns a string in the format "extracted/total (percentage%)".
        """
        if not self.has_labels:
            return "N/A"
        
        total_labels = 0
        extracted_labels = 0
        
        for item in data:
            # Each item should have 3 passage labels
            for i in range(1, 4):
                label_key = f'passage_{i}_label'
                total_labels += 1
                label_value = item.get(label_key, None)
                if label_value is not None and label_value != "":
                    extracted_labels += 1
        
        percentage = (extracted_labels / total_labels * 100) if total_labels > 0 else 0.0
        return f"{extracted_labels}/{total_labels} ({percentage:.1f}%)"

    def compute_ave_conf(self, data):
        """
        Compute average confidence for the given data.
        Average confidence is the mean of all confidence scores in the data.
        """
        total_confidence = 0.0
        count = 0
        
        for item in data:
            confidence_str = item.get('confidence', None)
            
            # Handle unmatched samples with None confidence
            if confidence_str is None:
                # Skip samples with no confidence value
                continue
            else:
                try:
                    # Clean and parse confidence string
                    clean_str = str(confidence_str).strip().replace('%', '')
                    confidence = float(clean_str)
                    
                    # Normalize to [0,1] range
                    if confidence > 1:  # Assume it's percentage if >1
                        confidence /= 100.0
                    
                    # Ensure confidence is within [0, 1]
                    confidence = min(max(confidence, 0.0), 1.0)
                except (ValueError, TypeError):
                    # Skip invalid confidence values
                    continue
            
            total_confidence += confidence
            count += 1
            
        return total_confidence / count if count > 0 else 0.0

    def compute_accuracy(self, data):
        """
        Compute accuracy for the given data.
        Accuracy is the proportion of correct predictions to the total number of predictions.
        """
        correct_count = 0
        total_count = 0
        
        for item in data:
            pred = item['answer']
            truth = item['true_answer']
            
            # Skip if either prediction or truth is missing
            if pred is None or truth is None:
                continue
                
            if self.judge_func(pred, truth):
                correct_count += 1
                
            total_count += 1
            
        return correct_count / total_count if total_count > 0 else 0.0
        

    def _prepare_ece_samples(self, data):
        """
        Prepare calibrated samples for ECE computation.
        Returns (samples, invalid_samples) where samples is a list of dicts with id/confidence/correct.
        """
        samples = []
        invalid_samples = 0

        for item in data:
            pred = item.get('answer', None)
            truth = item.get('true_answer', None)
            confidence_str = item.get('confidence', None)

            # Skip if pred or truth is missing
            if pred is None or truth is None:
                invalid_samples += 1
                continue

            if confidence_str is None:
                invalid_samples += 1
                continue

            try:
                clean_str = str(confidence_str).strip().replace('%', '')
                confidence = float(clean_str)
                if confidence > 1:
                    confidence /= 100.0
                confidence = min(max(confidence, 0.0), 1.0)
            except (ValueError, TypeError):
                invalid_samples += 1
                continue

            correct = 1 if self.judge_func(pred, truth) else 0
            samples.append({
                "confidence": confidence,
                "correct": correct,
                "id": item.get('id')
            })

        return samples, invalid_samples

    def _compute_ece_from_samples(self, samples, num_bins=10):
        if not samples:
            return float('nan')

        samples = sorted(samples, key=lambda x: x["confidence"])
        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

        bin_weights = np.zeros(len(bin_edges) - 1)
        bin_confidences = np.zeros(len(bin_edges) - 1)
        bin_accuracies = np.zeros(len(bin_edges) - 1)

        for sample in samples:
            conf = sample["confidence"]
            correct = sample["correct"]
            bin_index = np.searchsorted(bin_edges, conf, side='right') - 1
            bin_index = min(max(bin_index, 0), len(bin_edges) - 2)
            bin_weights[bin_index] += 1
            bin_confidences[bin_index] += conf
            bin_accuracies[bin_index] += correct

        ece = 0.0
        total_samples = len(samples)
        for i in range(len(bin_weights)):
            if bin_weights[i] > 0:
                avg_confidence = bin_confidences[i] / bin_weights[i]
                avg_accuracy = bin_accuracies[i] / bin_weights[i]
                ece += (bin_weights[i] / total_samples) * abs(avg_confidence - avg_accuracy)

        return ece

    def compute_ece(self, data, num_bins=10):
        """
        Compute Expected Calibration Error (ECE) for the given data using 10-bin fixed binning.
        """
        samples, invalid_samples = self._prepare_ece_samples(data)
        if invalid_samples > 0:
            print(f"Warning: Skipped {invalid_samples} samples due to missing or invalid data")

        if not samples:
            print("Error: No valid samples for ECE computation")
            return 0.0

        print(f"Collected {len(samples)} valid samples out of {len(data)} for ECE computation.")
        return self._compute_ece_from_samples(samples, num_bins=num_bins)

    def compute_auroc(self, data):
        """
        Compute Area Under the Receiver Operating Characteristic Curve (AUROC) for the given data.
        AUROC is a measure of how well the model distinguishes between positive and negative classes.
        """
        # Prepare data: list of tuples (confidence, correct)
        samples = []
        
        for item in data:
            pred = item['answer']
            truth = item['true_answer']
            confidence_str = item.get('confidence', None)
            
            # Skip if pred or truth is missing
            if pred is None or truth is None:
                continue
                
            # Handle unmatched samples with None confidence
            if confidence_str is None:
                confidence = 0.0
            else:
                try:
                    # Handle percentages or regular numbers
                    if '%' in confidence_str:
                        confidence = float(confidence_str.strip('%')) / 100.0
                    else:
                        confidence = float(confidence_str) / 100.0
                    # Ensure confidence is within [0, 1]
                    confidence = min(max(confidence, 0.0), 1.0)
                except (ValueError, TypeError):
                    confidence = 0.0
                
            correct = 1 if self.judge_func(pred, truth) else 0
            samples.append((confidence, correct))
            
        if not samples:
            return 0.0

        print(f"Collected {len(samples)} valid samples out of {len(data)} for AUROC computation.")

        # Separate confidence scores and labels
        confidences = np.array([conf for conf, correct in samples])
        labels = np.array([correct for conf, correct in samples])
        
        # Calculate AUROC

        unique_classes = np.unique(labels)
        if len(unique_classes) < 2:
            # Only one class present, return NaN
            print("Warning: Only one class present in data for AUROC computation.")
            return float('nan')

        return roc_auc_score(labels, confidences)

    def compute_auprc(self, data):
        """
        Compute Area Under the Precision-Recall Curve (AUPRC) for the given data.
        AUPRC is a measure of how well the model distinguishes between positive and negative classes,
        particularly useful for imbalanced datasets as it focuses on the positive class.
        """
        # Prepare data: list of tuples (confidence, correct)
        samples = []
        
        for item in data:
            pred = item['answer']
            truth = item['true_answer']
            confidence_str = item.get('confidence', None)
            
            # Skip if pred or truth is missing
            if pred is None or truth is None:
                continue
                
            # Handle unmatched samples with None confidence
            if confidence_str is None:
                confidence = 0.0
            else:
                try:
                    # Handle percentages or regular numbers
                    if '%' in confidence_str:
                        confidence = float(confidence_str.strip('%')) / 100.0
                    else:
                        confidence = float(confidence_str) / 100.0
                    # Ensure confidence is within [0, 1]
                    confidence = min(max(confidence, 0.0), 1.0)
                except (ValueError, TypeError):
                    confidence = 0.0
                
            correct = 1 if self.judge_func(pred, truth) else 0
            samples.append((confidence, correct))
            
        if not samples:
            return 0.0

        print(f"Collected {len(samples)} valid samples out of {len(data)} for AUPRC computation.")

        # Separate confidence scores and labels
        confidences = np.array([conf for conf, correct in samples])
        labels = np.array([correct for conf, correct in samples])
        
        # Calculate AUPRC
        unique_classes = np.unique(labels)
        if len(unique_classes) < 2:
            # Only one class present, return NaN
            print("Warning: Only one class present in data for AUPRC computation.")
            return float('nan')

        # Compute precision-recall curve
        precision, recall, _ = precision_recall_curve(labels, confidences)
        
        # Compute area under the precision-recall curve
        auprc = auc(recall, precision)
        
        return auprc

    def _load_passage_type_map(self):
        """Load original data and create a mapping from item id to passage types."""
        if self.passage_type_map is not None:
            return
        
        self.passage_type_map = {}
        
        if not self.original_data_path or not os.path.exists(self.original_data_path):
            print(f"Warning: Original data path not found: {self.original_data_path}")
            return
        
        try:
            with open(self.original_data_path, 'r') as f:
                original_data = json.load(f)
            
            # Original data is a list of items
            for item in original_data:
                item_id = item.get('id')
                if item_id is None:
                    continue
                
                passages = item.get('passages', [])
                if len(passages) >= 3:
                    # Store passage types in order
                    passage_types = [p.get('type', '').lower() for p in passages[:3]]
                    self.passage_type_map[item_id] = passage_types
            
            print(f"Loaded passage type map for {len(self.passage_type_map)} items from {self.original_data_path}")
        except Exception as e:
            print(f"Error loading original data: {e}")
            self.passage_type_map = {}
    
    def _get_expected_label(self, passage_type):
        """Map passage type to expected label (case-insensitive)."""
        passage_type_lower = passage_type.lower() if passage_type else ""
        
        # Map passage types to expected labels
        if passage_type_lower in ['counterfactual', 'gt_passage', 'consistent']:
            return '##Highly Relevant##'
        elif passage_type_lower == 'relevant':
            return '##Relevant##'
        elif passage_type_lower == 'irrelevant':
            return '##Irrelevant##'
        else:
            return None  # Unknown type
    
    def _normalize_label(self, label):
        """Normalize label for comparison (case-insensitive, strip whitespace)."""
        if label is None or label == "":
            return None
        return label.strip().lower()
    
    def compute_label_accuracy(self, data):
        """Compute overall label accuracy across all passages."""
        if not self.has_labels or self.passage_type_map is None:
            return float('nan')
        
        correct_count = 0
        total_count = 0
        
        for item in data:
            item_id = item.get('id')
            if item_id not in self.passage_type_map:
                continue
            
            passage_types = self.passage_type_map[item_id]
            # Get labels, defaulting to empty string if missing
            predicted_labels = [
                item.get('passage_1_label', ''),
                item.get('passage_2_label', ''),
                item.get('passage_3_label', '')
            ]
            
            # Check all three passages
            for i, (passage_type, predicted_label) in enumerate(zip(passage_types, predicted_labels)):
                expected_label = self._get_expected_label(passage_type)
                if expected_label is None:
                    continue
                
                # Normalize both labels for case-insensitive comparison
                # Empty strings will be normalized to None
                predicted_normalized = self._normalize_label(predicted_label)
                expected_normalized = self._normalize_label(expected_label)
                
                # If predicted label is empty/None, it's incorrect
                if predicted_normalized is not None and predicted_normalized == expected_normalized:
                    correct_count += 1
                total_count += 1
        
        return correct_count / total_count if total_count > 0 else 0.0

    def _build_label_confusion(self, data):
        """
        Build a confusion matrix for passage labels using expected labels from original data.
        Returns (matrix, label_names) where matrix is n x n numpy array.
        """
        if self.passage_type_map is None:
            return None, None

        label_display = ['Highly Relevant', 'Relevant', 'Irrelevant']
        canonical_labels = ['##Highly Relevant##', '##Relevant##', '##Irrelevant##']
        normalized_labels = [self._normalize_label(label) for label in canonical_labels]
        label_to_idx = {label: idx for idx, label in enumerate(normalized_labels)}
        matrix = np.zeros((len(canonical_labels), len(canonical_labels)), dtype=int)
        missing_predictions = 0

        for item in data:
            item_id = item.get('id')
            if item_id not in self.passage_type_map:
                continue
            passage_types = self.passage_type_map[item_id]
            predicted_labels = [
                item.get('passage_1_label', ''),
                item.get('passage_2_label', ''),
                item.get('passage_3_label', '')
            ]

            for passage_type, predicted_label in zip(passage_types, predicted_labels):
                expected_label = self._get_expected_label(passage_type)
                normalized_expected = self._normalize_label(expected_label)
                if normalized_expected not in label_to_idx:
                    continue
                normalized_predicted = self._normalize_label(predicted_label)
                if normalized_predicted not in label_to_idx:
                    missing_predictions += 1
                    continue
                matrix[label_to_idx[normalized_expected], label_to_idx[normalized_predicted]] += 1

        if missing_predictions > 0:
            print(f"Warning: {missing_predictions} label predictions were missing or invalid and were excluded from the confusion matrix.")

        return matrix, label_display

    def save_label_confusion_matrix(self, data, output_png_path):
        """
        Save the aggregated label confusion matrix across all tasks into a PNG.
        """
        if not self.has_labels:
            return
        if self.passage_type_map is None:
            print("Warning: Cannot generate confusion matrix without original passage types.")
            return

        combined_matrix = None
        label_names = None

        for task, items in data.items():
            if task == 'unmatched_samples' or not isinstance(items, list):
                continue
            matrix, labels = self._build_label_confusion(items)
            if matrix is None:
                continue
            if combined_matrix is None:
                combined_matrix = matrix
                label_names = labels
            else:
                combined_matrix += matrix

        if combined_matrix is None or label_names is None:
            print("Warning: No label data available to build confusion matrix.")
            return

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(combined_matrix, cmap='Blues')

        ax.set_xticks(range(len(label_names)))
        ax.set_yticks(range(len(label_names)))
        ax.set_xticklabels(label_names, rotation=45, ha='right')
        ax.set_yticklabels(label_names)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Passage Label Confusion Matrix')

        for i in range(combined_matrix.shape[0]):
            for j in range(combined_matrix.shape[1]):
                value = combined_matrix[i, j]
                ax.text(j, i, str(value), ha='center', va='center', color='black')

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
        fig.savefig(output_png_path)
        plt.close(fig)
    
    def compute_passage_label_accuracy(self, data, passage_num):
        """Compute label accuracy for a specific passage (1, 2, or 3)."""
        if not self.has_labels or self.passage_type_map is None:
            return float('nan')
        
        if passage_num not in [1, 2, 3]:
            raise ValueError(f"passage_num must be 1, 2, or 3, got {passage_num}")
        
        correct_count = 0
        total_count = 0
        
        for item in data:
            item_id = item.get('id')
            if item_id not in self.passage_type_map:
                continue
            
            passage_types = self.passage_type_map[item_id]
            if len(passage_types) < passage_num:
                continue
            
            passage_type = passage_types[passage_num - 1]
            predicted_label = item.get(f'passage_{passage_num}_label')
            expected_label = self._get_expected_label(passage_type)
            
            if expected_label is None:
                continue
            
            # Normalize both labels for case-insensitive comparison
            predicted_normalized = self._normalize_label(predicted_label)
            expected_normalized = self._normalize_label(expected_label)
            
            if predicted_normalized == expected_normalized:
                correct_count += 1
            total_count += 1
        
        return correct_count / total_count if total_count > 0 else 0.0

def _detect_data_features(extracted_data):
    """
    Inspect extracted data to determine whether confidence scores or passage labels are present.
    """
    label_keys = ['passage_1_label', 'passage_2_label', 'passage_3_label']
    has_confidence = False
    has_labels = False

    for task, items in extracted_data.items():
        if task == 'unmatched_samples':
            continue
        if not isinstance(items, list):
            continue
        for item in items:
            if not has_confidence and 'confidence' in item:
                has_confidence = True
            if not has_labels and any(label_key in item for label_key in label_keys):
                has_labels = True
            if has_confidence and has_labels:
                return has_confidence, has_labels

    return has_confidence, has_labels


def evaluate_file(input_path, output_path, extractor_name=None, original_data_dir=None):
    with open(input_path, 'r') as f:
        extracted_data = json.load(f)
    
    has_confidence, has_labels = _detect_data_features(extracted_data)
    
    metrics = ['accuracy']
    if has_labels:
        metrics.append('label_accuracy')
    if has_confidence:
        metrics.extend(['ave_conf', 'ece', 'auroc', 'auprc'])
    metrics.append('valid_sample_portion')
    if has_labels:
        metrics.append('label_extraction_portion')
    if has_confidence:
        metrics.append('reliability_diagram')
    
    # Find original data file if needed for label evaluation
    original_data_path = None
    if has_labels and original_data_dir:
        filename = os.path.basename(input_path)
        original_data_path = os.path.join(original_data_dir, filename)
        if not os.path.exists(original_data_path):
            print(f"Warning: Original data file not found: {original_data_path}")
            original_data_path = None
    elif has_labels and not original_data_dir:
        print("Warning: Original data directory required for label evaluation, but not provided.")
    
    evaluator = Evaluator(
        metrics=metrics,
        judge_func=f1_judge,
        model_name=input_path.split('/')[-1].replace('.json', '').split("_")[-1],
        extractor_name=extractor_name,
        original_data_path=original_data_path,
        has_confidence=has_confidence,
        has_labels=has_labels,
    )

    results = evaluator.evaluate(extracted_data)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    if has_labels:
        confusion_png_path = output_path.replace('.json', '_label_confusion.png')
        evaluator.save_label_confusion_matrix(extracted_data, confusion_png_path)


if __name__ == "__main__":
    import sys
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate model outputs for QA tasks')
    parser.add_argument('--input-dir', type=str, default='output_data/QA_extracted',
                        help='Directory containing extracted data files (default: output_data/QA_extracted)')
    parser.add_argument('--output-dir', type=str, default='output_data/QA_evaluation',
                        help='Directory to save evaluation results (default: output_data/QA_evaluation)')
    parser.add_argument('--model-filter', nargs="+", default=None,
                       help='Filter files by base model name(s). Only process files containing these model names.')
    parser.add_argument('--mode', type=str, default='overwrite', choices=['add', 'overwrite'],
                       help="Mode for processing files. 'add': skip files that already exist in output directory. 'overwrite': process all files regardless of existence (default: overwrite)")
    parser.add_argument('--extractor', type=str, default=None,
                       help="Extractor name to use for pattern matching. Options: 'ckpt_test', 'base_without_rules', 'base_pure', 'rag_test'")
    parser.add_argument('--original-data-dir', type=str, default=None,
                       help="Directory containing original data files (required for ckpt_test extractor when has_labels=True).")
    
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    input_paths = []
    output_paths = []

    all_files = [file for file in os.listdir(args.input_dir) if file.endswith(".json")]
    
    if args.model_filter:
        filtered_files = []
        for file in all_files:
            for model_name in args.model_filter:
                if model_name in file:
                    filtered_files.append(file)
                    break
        all_files = filtered_files
        print(f"Filtered to {len(all_files)} files matching model filter(s)")
    
    input_paths = [os.path.join(args.input_dir, file) for file in all_files]
    output_paths = [os.path.join(output_dir, os.path.basename(file)) for file in input_paths]

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

    print("Starting evaluation process...")
    print("=" * 80)
    print(f"Using input directory: {input_dir}")
    print(f"Using output directory: {output_dir}")
    print(f"Mode: {args.mode}")
    
    os.makedirs(output_dir, exist_ok=True)

    for input_path, output_path in zip(input_paths, output_paths):
        print(f"Processing {input_path} to {output_path}...")
        evaluate_file(input_path, output_path, extractor_name=args.extractor, original_data_dir=args.original_data_dir)
        print(f"Finished processing {input_path}. Results saved to {output_path}.")
        print("-" * 80)
    
    print("Evaluation process completed.")
    print("=" * 80)


