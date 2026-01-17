import json
import os
from prompts import *

def get_facts(fact_list):
    '''
    input: fact_list: the list of facts to be used in the prompt
    output: return a string of facts
    '''
    facts = ""
    for fact in fact_list:
        fact_index = fact_list.index(fact) + 1
        indexed_fact = f"{fact_index}. {fact}"
        facts += indexed_fact + "\n"
    return facts[:-1]



def load_single_qa_prompt(args, question, fact_list=None, answer=None, question_type=None, ground_truth_classifications=None, passage_types=None, exp_sentences=None):
    
    if args.task == "base_without_rules":
        question_type = QUESTIONT_TYPE_TEMPLATE[question_type] if question_type else QUESTIONT_TYPE_TEMPLATE[args.question_type]
        prompt = PROMPT_TEMPLATE[args.task].format(
            question=question,
            facts=get_facts(fact_list) if fact_list else "",
            question_type=question_type
        )

    elif args.task == "base_pure":
        question_type = QUESTIONT_TYPE_TEMPLATE[question_type] if question_type else QUESTIONT_TYPE_TEMPLATE[args.question_type]
        prompt = PROMPT_TEMPLATE[args.task].format(
            question=question,
            facts=get_facts(fact_list) if fact_list else "",
            question_type=question_type
        )

    elif args.task == "ckpt_test":
        question_type = QUESTIONT_TYPE_TEMPLATE[question_type] if question_type else QUESTIONT_TYPE_TEMPLATE[args.question_type]
        prompt = PROMPT_TEMPLATE[args.task].format(
            question=question,
            facts=get_facts(fact_list) if fact_list else "",
            question_type=question_type
        )

    elif args.task == "base_sample":
        question_type = QUESTIONT_TYPE_TEMPLATE[question_type] if question_type else QUESTIONT_TYPE_TEMPLATE[args.question_type]
        prompt = PROMPT_TEMPLATE[args.task].format(
            question=question,
            facts=get_facts(fact_list) if fact_list else "",
            question_type=question_type
        )

    elif args.task == "rag_test":
        question_type = QUESTIONT_TYPE_TEMPLATE[question_type] if question_type else QUESTIONT_TYPE_TEMPLATE[args.question_type]
        # Select the appropriate prompt based on prompt_type
        prompt_template = PROMPT_TEMPLATE[args.task][args.prompt_type]
        prompt = prompt_template.format(
            question=question,
            facts=get_facts(fact_list) if fact_list else "",
            question_type=question_type
        )

    return prompt
        

def load_prompt(args, batch): 
    '''
    input: 
    args: the arguments containing the task and question type
    batch: the batch of data to be used for generating prompts
    
    output: 
    return the prompt template for the task and question type
    '''
    prompts = []

    if args.task == "base_without_rules":
        for item in batch:
            question_type = 'bi' if item["id"][0] == 's' else 'oe'
            fact_list = [p["content"] for p in item.get("passages", [])]
            prompt = load_single_qa_prompt(
                args,
                item["question"],
                fact_list=fact_list,
                question_type=question_type
            )
            for sample_id in range(args.sample_num):
                prompts.append(prompt)

    elif args.task == "base_pure":
        for item in batch:
            question_type = 'bi' if item["id"][0] == 's' else 'oe'
            fact_list = [p["content"] for p in item.get("passages", [])]
            prompt = load_single_qa_prompt(
                args,
                item["question"],
                fact_list=fact_list,
                question_type=question_type
            )
            for sample_id in range(args.sample_num):
                prompts.append(prompt)

    elif args.task == "ckpt_test":
        for item in batch:
            question_type = 'bi' if item["id"][0] == 's' else 'oe'
            fact_list = [p["content"] for p in item.get("passages", [])]
            prompt = load_single_qa_prompt(
                args,
                item["question"],
                fact_list=fact_list,
                question_type=question_type
            )
            for sample_id in range(args.sample_num):
                prompts.append(prompt)

    elif args.task == "base_sample":
        for item in batch:
            question_type = 'bi' if item["id"][0] == 's' else 'oe'
            fact_list = [p["content"] for p in item.get("passages", [])]
            prompt = load_single_qa_prompt(
                args,
                item["question"],
                fact_list=fact_list,
                question_type=question_type
            )
            for sample_id in range(args.sample_num):
                prompts.append(prompt)

    elif args.task == "rag_test":
        for item in batch:
            question_type = 'bi' if item["id"][0] == 's' else 'oe'
            # Use fact_used to select between bm25-facts or Contriever-facts
            fact_list = item.get(args.fact_used, [])
            prompt = load_single_qa_prompt(
                args,
                item["question"],
                fact_list=fact_list,
                question_type=question_type
            )
            for sample_id in range(args.sample_num):
                prompts.append(prompt)

    else:
        print("expected input file: ", args.input_file)
        raise ValueError(f"Unknown input file format: {args.input_file}")
    
    # print("prompts:", prompts)
    return prompts