import re 
import torch 
import os 

from datasets import load_dataset 
import evaluate 

import sanitize 

os.environ["HF_ALLOW_CODE_EVAL"] = "1"


def get_ut_dataset(split="train"):
    scratch_dir = os.getenv('SCRATCH')
    hf_cache_dir = os.path.join(scratch_dir, 'huggingface_cache')

    data_dir = os.path.join(scratch_dir, 'code_data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    dataset = load_dataset("KAKA22/CodeRM-UnitTest", cache_dir = hf_cache_dir)

    return dataset[split] 

def check_constraints(txt, commands):
    includes_list = torch.zeros(len(commands), dtype=torch.bool)
    
    for i, command in enumerate(commands):
        if command in txt:
            includes_list[i] = True
        else:
            includes_list[i] = False

    # number of commands included
    num_included = includes_list.sum().item()

    print("Number of constraints met: ", num_included, "/", len(commands))

    return includes_list

def eval_tests(test_list, code_list):
    code_eval = evaluate.load("code_eval")
    test_cases = [sanitize.sanitize(test) for test in test_list]

    code_list = [sanitize.sanitize(code.split('```python\n', 1)[-1]) for code in code_list] 

    candidates = [[sanitize.sanitize(code) for code in code_list]]

    num_pass = 0


    for test in test_cases:
        print("\ntesting:")
        print("Test: ", test)
        print("Code: ", code_list[0])
    
        pass_at_k, results = code_eval.compute(references=[test], predictions=candidates, k=range(1, len(code_list)+1))
        
        num_pass += pass_at_k['pass@1']

    frac_pass = num_pass / len(test_cases)

    print("number passed: ", num_pass, " out of ", len(test_cases))

    return frac_pass 