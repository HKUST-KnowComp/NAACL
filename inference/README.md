# Inference and evaluation

The inference module has three stages: generation through an OpenAI-compatible endpoint, response extraction, and metric computation.

## Tasks

| Task | Purpose | Response layout |
|---|---|---|
| `base_pure` | vanilla QA and confidence baseline | `response/base_pure/<prompt>` |
| `base_without_rules` | passage-label prompt without NOVA rules | `response/base_without_rules/<prompt>` |
| `base_sample` | Best-of-N training response generation | `response/base_sample` |
| `ckpt_test` | LoRA evaluation | `response/ckpt_test` |
| `rag_test` | pre-retrieved BM25/Contriever evaluation | `response/rag_test/<retriever>/<prompt>` |

The supported backbones are Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, DeepSeek-R1-Distill-Qwen-7B, and DeepSeek-R1-Distill-Llama-8B. The default maximum generation length is 2048 tokens.

## Generation

Start a compatible server:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 40002
```

Then run:

```bash
python3 inference/generator/budget_forcing.py \
  --input_file datasets/prepared/threePassages/strategyqa/test.json \
  --dataset strategyqa-test \
  --output_file inference/output_data/base_pure/run/qwen.json \
  --task base_pure \
  --prompt_type vanilla \
  --question_type bi \
  --sample_num 1 \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --temperature 0 \
  --port 40002
```

Dataset-specific question types are inferred from prepared record IDs. Shell wrappers under `generator/.sh/` expose the multi-model runs. The serving scripts assume NVIDIA GPUs; adjust the GPU and port arrays for the target machine.

For `rag_test`, set `RAG_INPUT_FILE` to JSON records containing `bm25-facts` and `Contriever-facts` lists:

```bash
RAG_INPUT_FILE=/path/to/retrieved.json bash inference/generator/.sh/rag_test.sh
```

For LoRA evaluation, pass the parent directory through `CHECKPOINT_PATH`, for example `CHECKPOINT_PATH=/path/to/models bash inference/generator/.sh/ckpt_serve.sh`.

## Extraction and metrics

```bash
bash inference/eval_utils/.sh/eval.sh \
  inference/output_data/base_pure/run \
  --extractor base_pure
```

Available extractors are `base_pure`, `base_without_rules`, `ckpt_test`, and `rag_test`. The evaluator reports accuracy, average confidence, 10-bin ECE, AUROC, AUPRC, valid sample portion, and reliability-diagram counts. Checkpoint outputs additionally support passage-label accuracy.

Run `python3 tests/smoke_test.py` to exercise extraction and evaluation offline.

## NOVA training data

Generate 16 responses per prompt at temperature 1.0 with `generator/.sh/base_sample.sh`, then run:

```bash
python3 inference/process_utils/filter_rule.py \
  --input <raw-output-dir> \
  --output <filtered-output-dir> \
  --enable-drop 0.05 \
  --tolerate-mismatch

python3 inference/process_utils/sft_format.py \
  --input <filtered-output-dir> \
  --output <sft-output-dir>
```

The formatter intersects IDs across model files, balances retrieval scenarios, selects confidence-aligned responses, and writes LLaMA-Factory `instruction`/`input`/`output` records. Training configs are under `../training/`.
