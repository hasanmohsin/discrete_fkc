# load human eval dataset and look at 1 sample
import json
from datasets import load_dataset
import os 
import sys 

import ast 

from code_utils import * 
import evaluate 
import sanitize 

os.environ["HF_ALLOW_CODE_EVAL"] = "1"


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from samplers import DiffusionSampler
from prod_prompt_samplers import ProductPromptSampler, GeoAvgPromptSampler

from llada_denoiser import LLaDADenoiser

def mbpp_load():
    scratch_dir = os.getenv('SCRATCH')
    hf_cache_dir = os.path.join(scratch_dir, 'huggingface_cache')

    data_dir = os.path.join(scratch_dir, 'code_data')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    dataset = load_dataset("mbpp", cache_dir=hf_cache_dir)

    return dataset

def parse_unit_tests(unit_test_str, entry_point):
    # unit_test_list is a list of strings, each string is a unit test
    # solution_code is a string, the code to be tested
    # replace candidate with entry point in unit_test_str
    unit_test_str = unit_test_str.replace("candidate", entry_point)
    # Split the body of `def check(...)` into individual assert statements (as strings).
    lines = unit_test_str.splitlines()
    asserts = []
    in_check = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Detect start of the check function
        if not in_check and stripped.startswith("def ") and "check" in stripped and stripped.endswith(":"):
            in_check = True
            continue
        if in_check:
            # Stop if we reach another top-level definition or non-indented block
            if line and not line.startswith((" ", "\t")) and not stripped.startswith("assert") and stripped != "":
                break
            if stripped.startswith("assert"):
                asserts.append(stripped)

    return asserts

def evaluate_code_unit_tests(unit_tests, solution_code):
    code_eval = evaluate.load("code_eval")

    print("unit tests: ", unit_tests)
    preds = []

    for ut in unit_tests:
        preds.append([solution_code])

    pass_at_k, results = code_eval.compute(references=unit_tests, predictions=preds, k=[1])

    print(pass_at_k)
    print(results)

    return results 

def sample_test():
    dataset = mbpp_load()
    sample = dataset['test'][12]

    print("mbpp dataset keys: ", sample.keys())
    
    print("Prompt:\n", sample['text'])
    
    print("\nReference Solution:\n", sample['code'])

    print("\nTest Cases:\n", sample['test_list'])

    # get entry point from sample code
    code_lines = sanitize.sanitize(sample['code'])

    # get def .... ( ... ): line
    for line in code_lines.splitlines():
        line = line.strip()
        if line.startswith("def "):
            entry_point = line.split("(")[0].replace("def ", "").strip()
            break

    sample['entry_point'] = entry_point

    print("\nEntry Point:\n", entry_point)


    results = evaluate_code_unit_tests(sample['test_list'], sanitize.sanitize(sample['code'], entrypoint=sample['entry_point']))


    #print("\nEvaluation Results:\n", results)

def main():

    #denoiser = LLaDADenoiser(device='cuda')
    sample_test()

    #sampler = DiffusionSampler(denoiser=denoiser, steps=128, temperature=1.0)
    #prod_sampler = ProductPromptSampler(denoiser=denoiser, resample=True, 
    #                                    steps=128, temperature=1.0)

if __name__ == "__main__":
    main()