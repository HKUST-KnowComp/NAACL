#!/usr/bin/env python3

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    from inference.process_utils.sft_format import select_response

    selected = select_response(
        [
            {"raw": {"content": "correct"}, "confidence": 90.0, "is_correct": True},
            {"raw": {"content": "wrong"}, "confidence": 20.0, "is_correct": False},
        ]
    )
    if not selected or not selected["is_correct"]:
        raise AssertionError("Brier-score response selection is incorrect")

    fixture = [
        {
            "id": "h-smoke-1",
            "question": "What is the capital of France?",
            "gt_answer": ["Paris"],
            "response": {
                "base_pure": {
                    "vanilla": ["Final Answer: Paris\nConfidence: 90%"]
                }
            },
        },
        {
            "id": "h-smoke-2",
            "question": "What is 2 + 2?",
            "gt_answer": ["4"],
            "response": {
                "base_pure": {
                    "vanilla": ["Final Answer: 5\nConfidence: 20%"]
                }
            },
        },
    ]

    with tempfile.TemporaryDirectory(prefix="nova-smoke-") as temp_dir:
        temp = Path(temp_dir)
        input_dir = temp / "base_pure"
        output_dir = temp / "results"
        input_dir.mkdir()
        input_file = input_dir / "smoke.json"
        input_file.write_text(json.dumps(fixture), encoding="utf-8")

        env = os.environ.copy()
        env["MPLCONFIGDIR"] = str(temp / "matplotlib")
        env["XDG_CACHE_HOME"] = str(temp / "cache")
        subprocess.run(
            [
                "bash",
                str(REPO_ROOT / "inference/eval_utils/.sh/eval.sh"),
                str(input_dir),
                "--extractor",
                "base_pure",
                "--output-base",
                str(output_dir),
            ],
            cwd=REPO_ROOT,
            env=env,
            check=True,
        )

        result_file = output_dir / "evaluated/smoke.json"
        results = json.loads(result_file.read_text(encoding="utf-8"))
        metrics = results["response/base_pure"]
        if metrics["accuracy"] != 0.5:
            raise AssertionError(f"Expected accuracy 0.5, got {metrics['accuracy']}")
        if metrics["valid_sample_portion"] != "2/2 (100.0%)":
            raise AssertionError("The evaluator did not retain both smoke samples")

        response = """Step 1: Inspect passage 1.
Step 2: Inspect passage 2.
Step 3: Inspect passage 3.
Step 4: Apply the NOVA rules.
Passage Classifications:
1. Highly Relevant
2. Relevant
3. Irrelevant
Answer: Paris
Confidence: 90%"""
        training_fixture = [
            {
                "id": "h-smoke-train",
                "question": "What is the capital of France?",
                "gt_answer": ["Paris"],
                "passages": [
                    {"content": "Paris is the capital of France.", "type": "consistent"},
                    {"content": "France is in Europe.", "type": "relevant"},
                    {"content": "Saturn has rings.", "type": "irrelevant"},
                ],
                "response": {"base_sample": [response]},
            }
        ]
        raw_dir = temp / "training_raw"
        filtered_dir = temp / "training_filtered"
        sft_dir = temp / "training_sft"
        raw_dir.mkdir()
        training_file = raw_dir / "hotpotqa-train_Qwen2.5-7B-Instruct.json"
        training_file.write_text(json.dumps(training_fixture), encoding="utf-8")

        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "inference/process_utils/filter_rule.py"),
                "--input",
                str(raw_dir),
                "--output",
                str(filtered_dir),
            ],
            cwd=REPO_ROOT,
            env=env,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "inference/process_utils/sft_format.py"),
                "--input",
                str(filtered_dir),
                "--output",
                str(sft_dir),
            ],
            cwd=REPO_ROOT,
            env=env,
            check=True,
        )
        sft_records = json.loads((sft_dir / training_file.name).read_text(encoding="utf-8"))
        if len(sft_records) != 1 or "Passage Classifications" not in sft_records[0]["output"]:
            raise AssertionError("Training filtering/SFT formatting smoke test failed")

    print("NOVA smoke test passed.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
