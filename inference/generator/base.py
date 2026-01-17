from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image
import base64
from io import BytesIO
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
import logging
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from dotenv import load_dotenv
import os
load_dotenv()

logger = logging.getLogger(__name__)

url_dic = {
    "gpt-4o-mini": {
        "api_key": os.getenv('NUWA_API_KEY'),
        "base_url": os.getenv("NUWA_BASE_URL")  
    },
    "deepseek-reasoner":{
        "api_key": os.getenv('DEEPSEEK_R1_API_KEY'),
        "base_url": os.getenv('DEEPSEEK_R1_URL'),
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
        "api_key": "EMPTY",
        "base_url": "http://0.0.0.0:31000/v1",
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
        "api_key": "EMPTY",
        "base_url": "http://0.0.0.0:31001/v1",
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "api_key": "EMPTY",
        "base_url": "http://0.0.0.0:40001/v1",
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "api_key": "EMPTY",
        "base_url": "http://0.0.0.0:40002/v1",
    },
    "Qwen/Qwen3-8B": {
        "api_key": "EMPTY",
        "base_url": "http://0.0.0.0:10002/v1",
    },
    "Qwen/Qwen2.5-32B-Instruct": { # GPU 6,7
        "api_key": "EMPTY",
        "base_url": "http://0.0.0.0:10005/v1",
    },
    "Qwen/Qwen3-32B": { # GPU 6,7
        "api_key": "EMPTY",
        "base_url": "http://0.0.0.0:10006/v1",
    }
}

class BaseGenerator(ABC):
    def __init__(self, args, model_name):
        """Initialize the generator with configuration."""
        self.model = model_name
        self.client = OpenAI(
            api_key=url_dic[model_name]["api_key"] if model_name in url_dic else "EMPTY",
            base_url=url_dic[model_name]["base_url"] if args.port == 0 else f"http://0.0.0.0:{args.port}/v1",
            timeout=3600,
        )
        self.batch_size = 256

        # if args.task == "llm_filter":
        #     self.batch_size = 2048 # for filtering, larger batch size

    @abstractmethod
    def _prepare_prompts_for_batch(self, **kwargs) -> Any:
        """Prepare messages for the different API call."""
        pass
    
    def _generate_single(self, prompt: str, request_id: Optional[str], **kwargs ) -> Tuple[str, Optional[str]]:
        """Generate a single response with optional request ID for tracking.
        
        If n parameter is provided in kwargs and n > 1, returns multiple results.
        Otherwise, returns a single result.
        """        
        try:
            if self.model:
                if isinstance(prompt, list):
                    # Treat prompt as messages
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=prompt,
                        stream=False,
                        **kwargs
                    )
                    return (response.choices[0].message.content, request_id), response.usage.completion_tokens if response.usage else 0
                elif isinstance(prompt, str):
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "user", 
                                "content": prompt
                            }
                        ],
                        stream=False,
                        **kwargs
                    )
                    return (response.choices[0].message.content, request_id), response.usage.completion_tokens if response.usage else 0
            else:
                raise ValueError("Model name is not specified.")
        except Exception as e:
            logger.error(f"Error generating response for request {request_id}: {str(e)}")
            return (str(e), request_id), 0
    
    def generate_batch(self, prompt_list: List[Tuple[str, Optional[str]]]) -> List[Tuple[str, Optional[str]]]:
        """Generate responses for a batch of messages using ThreadPoolExecutor with context manager.
        
        If _generate_single returns a list (batch_decode mode), results will be expanded.
        """
        results = []
        future_to_idx = {}
        
        # with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
        # with ProcessPoolExecutor(max_workers=self.batch_size) as executor:
            # Submit all tasks to the executor
            for idx, (prompt, request_id) in enumerate(prompt_list):
                future = executor.submit(self._generate_single, prompt, request_id)
                future_to_idx[future] = idx
            
            # Process futures as they complete with progress bar
            with tqdm(total=len(future_to_idx), desc="Generating responses") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        # If result is a list (batch_decode mode), expand it
                        if isinstance(result, list):
                            results.extend(result)
                        else:
                            results.append(result)
                    except Exception as e:
                        request_id = prompt_list[idx][1]
                        logger.error(f"Error in batch generation for request {request_id}: {str(e)}")
                        results.append((str(e), request_id))
                    pbar.update(1)
        
        # Sort results: first by request_id, then by original order within same request_id
        if results and all(len(r) >= 2 and r[1] is not None for r in results):
            # Create a mapping to track results by request_id and their order
            request_id_to_results = {}
            
            # Group results by request_id, maintaining the order they appear in results
            for result in results:
                rid = result[1]
                if rid not in request_id_to_results:
                    request_id_to_results[rid] = []
                request_id_to_results[rid].append(result)
            
            # Sort by request_id (using prompt_list order), then by position within request_id
            sorted_results = []
            seen_request_ids = set()
            
            # Iterate through prompt_list to maintain the order of request_ids
            for idx, (_, rid) in enumerate(prompt_list):
                if rid in request_id_to_results and rid not in seen_request_ids:
                    # Get results for this request_id
                    req_results = request_id_to_results[rid]
                    # Add all results for this request_id, maintaining their order
                    sorted_results.extend(req_results)
                    seen_request_ids.add(rid)
            
            results = sorted_results
            
        return results
    
    @abstractmethod
    def generate(self, **kwargs) -> Any:
        """Generate content based on input."""
        pass