# Noise generation

This module generates the four controlled passage types used by NOVA with Gemini 2.5 Pro through an OpenAI-compatible API.

## Setup

Install the root requirements and set credentials:

```bash
python3 -m pip install -r requirements.txt
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://your-endpoint.example/v1"  # optional for OpenAI
```

The model and decoding parameters are defined in `prompt_template.py`. The paper uses Gemini 2.5 Pro, temperature 0.6, and asks for five candidates; the parser retains the last three candidates.

## One task

```bash
python3 noise_generation/inference.py \
  --input_path datasets/prepared/threePassages/hotpotqa/test.json \
  --output_path inference/output_data/noise/hotpotqa-test.json \
  --task gen_counterfactual \
  --start_idx 0 \
  --end_idx 100 \
  --max_concurrent_tasks 10
```

Valid generation tasks are `gen_counterfactual`, `gen_relevant`, `gen_irrelevant`, and `gen_consistent`. `--end_idx 0` processes through the end. Partial runs preserve records outside the selected interval, and an existing output can be resumed to add another passage type.

## Batch generation

```bash
bash noise_generation/generate_noise.sh 64
```

The batch script processes all released three-passage splits and four noise types. It writes to `inference/output_data/noise_generated/` and resumes existing files. Set `START_IDX` and `END_IDX` to restrict the interval, or `OUTPUT_ROOT` to choose another output directory:

```bash
START_IDX=0 END_IDX=10 OUTPUT_ROOT=/tmp/nova-noise \
  bash noise_generation/generate_noise.sh 10
```

Generated files preserve all records outside a partial run's selected interval.
