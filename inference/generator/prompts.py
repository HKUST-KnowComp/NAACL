THINK_LEN_TEMPLATE = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 2048,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 2048,
    "Qwen/Qwen2.5-7B-Instruct": 2048,
    "meta-llama/Llama-3.1-8B-Instruct": 2048,
    "meta-llama/Llama-3.2-3B-Instruct": 1024,
    "deepseek-reasoner": 2048,
    "Qwen/Qwen3-8B": 2048,
    "Qwen/Qwen2.5-32B-Instruct": 2048,
    "Qwen/Qwen3-32B": 2048
}



BASE_WITHOUT_RULES = """You will be asked a question. You will be provided with 3 retrieved passages.
Each passage belongs to one of these 3 categories:
Highly Relevant: The passage direcly state an answer or strongly indicates an answer, regardless of whether the suggested answer is correct or not.
Relevant: The passage mentions some keywords or shares the same general topic as the question, but lacks information to answer the question.
Irrelevant: The passage has no shared topics or keywords with the question.

Task: Think step by step, analyze the passages one by one and classify their types (Highly Relevant, Relevant, Irrelevant), then give your your final answer ({question_type}) and confidence score in your answer.

Response Format:
Final Answer: [Your final answer]
Confidence: [Your confidence score between 0% - 100%]

##
Question: {question}
Retrieved Passages:\n{facts}
##
Your response:"""

BASE_PURE = """You will be asked a question. You will be provided with some retrieved passages.

Task: Think step by step, give your final answer ({question_type}) and confidence score in your answer.

Response Format:
Final Answer: [Your final answer]
Confidence: [Your confidence score between 0% - 100%]

##
Question: {question}
Retrieved Passages:\n{facts}
##
Your response:"""

CKPT_TEST_WITHOUT_RULES = """You will be asked a question. You will be provided with 3 retrieved passages.
Each passage belongs to one of these 3 categories:
Highly Relevant: The passage direcly state an answer or strongly indicates an answer, regardless of whether the suggested answer is correct or not.
Relevant: The passage mentions some keywords or shares the same general topic as the question, but lacks information to answer the question.
Irrelevant: The passage has no shared topics or keywords with the question.

Task: Think step by step, analyze the passages one by one and classify their types (Highly Relevant, Relevant, Irrelevant), then give your your final answer ({question_type}) and confidence score in your answer.

##
Question: {question}
Retrieved Passages:\n{facts}
##
Your response:"""

CKPT_TEST_NOEXP = """You will be asked a question. You will be provided with 3 retrieved passages.

Task: Give your answer ({question_type}) and confidence score in your answer.

##
Question: {question}
Retrieved Passages:\n{facts}
##
Your response:"""

VANILLA = """You will be asked a question. You will be provided with some retrieved passages.

Task: Give your final answer ({question_type}) and confidence score in your answer.

Response Format:
Final Answer: [Your final answer]
Confidence: [Your confidence score between 0% - 100%]

##
Question: {question}
Retrieved Passages:\n{facts}
##
Your response:"""

COT = """You will be asked a question. You will be provided with some retrieved passages.

Task: Analyze step by step, then give your final answer ({question_type}) and confidence score in your answer.

Response Format:
Final Answer: [Your final answer]
Confidence: [Your confidence score between 0% - 100%]

##
Question: {question}
Retrieved Passages:\n{facts}
##
Your response:"""

MULTI_STEP = """You will be asked a question. You will be provided with 3 retrieved passages.

Task: Analyze step by step, give your confidence score (0% - 100%) in each step, then give your final output, including your confidence score in each step and your answer ({question_type}).

Response Format:
Step 1: ...
Step 2: ...
...
Step K: ...
Final Output:
Step 1 Confidence: [Your confidence score between 0% - 100%]
Step 2 Confidence: [Your confidence score between 0% - 100%]
...
Step K Confidence: [Your confidence score between 0% - 100%]
Answer: [Your final answer]

##
Question: {question}
Retrieved Passages:\n{facts}
##
Your response:"""

BASE_SAMPLE = """You will be asked a question. You will be provided with 3 retrieved passages.
Each passage belongs to one of these 3 categories:
Highly Relevant: The passage direcly state an answer or strongly indicates an answer, regardless of whether the suggested answer is correct or not.
Relevant: The passage mentions some keywords or shares the same general topic as the question, but lacks information to answer the question.
Irrelevant: The passage has no shared topics or keywords with the question.

Rules:
1. If multiple passages are Highly Relevant, identify if there is a contradiction. 
  - If yes, you should not rely on the passages. Give your final answer based on your own knowledge and give corresponding confidence score.
  - If no, answer based on the consistent information from the passages and give corresponding confidence score.
2. If exactly one passage is Highly Relevant, give your final answer based on that passage and give corresponding confidence score.
3. If no passage is Highly Relevant, give your final answer based on your own knowledge and give corresponding confidence score.

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

PROMPT_TYPE_TEMPLATE = {
    "vanilla": "Give your response, then provide your final answer ({question_type}) and your confidence in this answer.",
    "cot": "Analyze step by step, give your reasoning response, then provide your answer ({question_type}) and your confidence in this answer.",
    "self-probing": "Read and answer the question.\nAnswer Format: You MUST use a pair of two asterisk to wrap your final answer ({question_type}), like this: [Final Answer]: **answer**.",
    "multi-step": "Break down the problem into K steps, think step by step, and give your confidence in each step by: [Step i Confidence]: your_confidence, where i = 1, 2, ..., k. Note that the confidence indicates how likely you think this step is correct.\nAnswer Format: your final answer MUST be prefixed with [Final Answer], and you MUST use a pair of two asterisk to wrap your final answer, like this:\n[Final Answer]: **your answer**.\nConfidence Format: you MUST provide your final confidence score (0%% - 100%%) between a pair of two hashes, prefixed with [Final Confidence], like this:\n[Final Confidence]: ##your confidence##.\nNote that the final confidence score should be the multiplication of all steps confidence scores. Do NOT do calculation within double hashes, it should be your final confidence value.",
    "top-k": "Provide your K best guesses to the question and you confidence score (0%% - 100%%) that each guessed answer is correct.\nYour final answer MUST be in the format of:\n\
    [Final Answer 1]: **answer1**, ##confidence1##;\n\
    [Final Answer 2]: **answer2**, ##confidence2##;\n\
    ...\n\
    [Final Answer K]: **answerK**, ##confidenceK##.",
}

QUESTIONT_TYPE_TEMPLATE = {
    "bi": "'yes' or 'no'",
    "mc": "one of the letters representing the options",
    "oe": "an accurate and concise answer in a few words"
}

PROMPT_TEMPLATE = {
    "ckpt_test": CKPT_TEST_WITHOUT_RULES,
    "base_without_rules": BASE_WITHOUT_RULES,
    "base_pure": BASE_PURE,
    "base_sample": BASE_SAMPLE,
    "rag_test": {"vanilla": VANILLA, "cot": COT, "multi-step": MULTI_STEP}
}

