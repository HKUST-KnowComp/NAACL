from base import BaseGenerator
from typing import List, Tuple, Optional
from tqdm import tqdm
import argparse
import json
import os
from prompts import *
from inference_utils import *   

HOME_DIR = os.getcwd()

def process_input_file(args, batch):
    # here, batches refer to the whole input data
    prompts = load_prompt(args, batch)  # Load the prompt template based on the task and question type
    return prompts

def get_max_think_len(model_name: str) -> int:
    return THINK_LEN_TEMPLATE.get(model_name, 1024)

class ForceBudgeGenerator(BaseGenerator):
    def __init__(self, args, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"):
        """Initialize the generator with configuration."""
        super().__init__(args, model_name)
        self.thinking_start = "<think>"
        self.thinking_end = "</think>"
        self.max_thinking_len = get_max_think_len(model_name)
        self.temperature = args.temperature

    def _prepare_prompt(self, input_text: str) -> str:
        """Prepare the prompt for the model."""
        return input_text

    def _prepare_prompts_for_batch(self, input_texts: List[str]) -> List[str]:
        """Prepare messages for the different API call."""
        return [(self._prepare_prompt(text), idx) for idx, text in enumerate(input_texts)]

    def _generate_single(
        self, prompt: str, request_id: Optional[str] = None
    ) -> Tuple[str, Optional[str]]:
        return super()._generate_single(prompt, request_id, max_tokens=self.max_thinking_len, temperature=self.temperature)[0]
        
    def generate(self, **kwargs) -> List[Tuple[str, Optional[str]]]:
        pass
    
def batch_inference(args):
    input_file = args.input_file
    output_path = args.output_file
    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists, loading it as the input file.")
        input_file = output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # check with the inputs
    print("input_file: ", input_file)
    print("model_name: ", args.model_name)

    input_data = json.load(open(input_file, "r"))[args.start_index:args.end_index] if args.end_index > 0 else json.load(open(input_file, "r"))
    print("the length of input data is: ", len(input_data))
    
    # turn in into the format of list of dict (including the sample nums if needed)
    prompts = process_input_file(args, input_data) 

    generator = ForceBudgeGenerator(
        args=args,
        model_name=args.model_name,
    )
    # pack the prompts with the request_id
    prompts = generator._prepare_prompts_for_batch(prompts)
    
    output = generator.generate_batch(prompts)
    
    if args.task in ["base_without_rules", "base_pure"]:
        for item in input_data:
            data_index = input_data.index(item)
            if "response" not in item:
                item["response"] = {}
            if args.task not in item["response"]:
                item["response"][args.task] = {}
            if args.prompt_type not in item["response"][args.task]:
                item["response"][args.task][args.prompt_type] = []
            for sample_id in range(args.sample_num):
                item["response"][args.task][args.prompt_type].append(output[data_index * args.sample_num + sample_id][0])
    
    elif args.task == "base_sample":
        for item in input_data:
            data_index = input_data.index(item)
            if "response" not in item:
                item["response"] = {}
            if args.task not in item["response"]:
                item["response"][args.task] = []
            for sample_id in range(args.sample_num):
                item["response"][args.task].append(output[data_index * args.sample_num + sample_id][0])
    
    elif args.task == "ckpt_test":
        for item in input_data:
            data_index = input_data.index(item)
            if "response" not in item:
                item["response"] = {}
            if args.task not in item["response"]:
                item["response"][args.task] = []
            for sample_id in range(args.sample_num):
                item["response"][args.task].append(output[data_index * args.sample_num + sample_id][0])
    
    elif args.task == "rag_test":
        for item in input_data:
            data_index = input_data.index(item)
            if "response" not in item:
                item["response"] = {}
            if args.task not in item["response"]:
                item["response"][args.task] = {}
            if args.fact_used not in item["response"][args.task]:
                item["response"][args.task][args.fact_used] = {}
            if args.prompt_type not in item["response"][args.task][args.fact_used]:
                item["response"][args.task][args.fact_used][args.prompt_type] = []
            for sample_id in range(args.sample_num):
                item["response"][args.task][args.fact_used][args.prompt_type].append(output[data_index * args.sample_num + sample_id][0])
            
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(input_data, f, ensure_ascii=False, indent=4)
    
    print("output_path:", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--task", type=str, required=True, choices=["base_without_rules", "base_pure", "base_sample", "ckpt_test", "rag_test"])
    parser.add_argument("--prompt_type", type=str, default="vanilla", choices=["vanilla", "cot", "self-probing", "multi-step", "top-k"])
    parser.add_argument("--question_type", type=str, required=True, choices=["fact-mem-cont", "mem-intra-cont", "non-relevant", "bi", "mc", "oe"])
    parser.add_argument("--sample_num", type=int, default=1, help="Number of samples to generate for each prompt.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=0)
    parser.add_argument("--port", type=int, default=0, help="Port for the model server. Use 0 for default port.")
    parser.add_argument("--fact_used", type=str, default=None, help="Used fact type for rag_test: 'bm25-facts' or 'Contriever-facts'")
    args = parser.parse_args()

    if args.task == "rag_test":
        if args.fact_used is None:
            parser.error("--fact_used is required when --task is rag_test. Use 'bm25-facts' or 'Contriever-facts'")
        if args.fact_used not in ["bm25-facts", "Contriever-facts"]:
            parser.error("--fact_used must be 'bm25-facts' or 'Contriever-facts' for rag_test task")

    batch_inference(args)
    
"""
serve commands:

vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --port 10000
vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B --port 10001
vllm serve Qwen/Qwen2.5-7B-Instruct --port 10002 
vllm serve meta-llama/Llama-3.1-8B --port 10003
"""
    
"""
example usage:

rag_test:
python generator/budget_forcing.py \
    --input_file datasets3_raged/strategyqa/test.json \
    --dataset StrategyQA \
    --output_file output_data/rag_test/StrategyQA_test.json \
    --task rag_test \
    --prompt_type vanilla \
    --question_type bi \
    --sample_num 1 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --fact_used bm25-facts \
    --start_index 0 \
    --end_index 10

python generator/budget_forcing.py \
    --input_file datasets3_raged/strategyqa/test.json \
    --dataset StrategyQA \
    --output_file output_data/rag_test/StrategyQA_test.json \
    --task rag_test \
    --prompt_type cot \
    --question_type bi \
    --sample_num 1 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --fact_used Contriever-facts \
    --start_index 0 \
    --end_index 10
"""