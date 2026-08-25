# LoRA training

These configs reproduce the final NOVA SFT settings reported in the paper: LoRA rank 16, maximum sequence length 2048, learning rate `5e-5`, and 2 epochs.

1. Generate and filter SFT JSON files as described in the root README.
2. Place the four JSON files in the `data/` directory of a LLaMA-Factory checkout.
3. Merge the entries from `dataset_info.json` into LLaMA-Factory's `data/dataset_info.json`.
4. Run the matching config from the LLaMA-Factory environment.

```bash
llamafactory-cli train /path/to/NAACL/training/qwen2_5_7b_lora_sft.yaml
```

Output directories are relative to the directory where `llamafactory-cli` is launched. The reported experiments used four NVIDIA L20 GPUs.
