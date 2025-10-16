
import torch
import torch.nn as nn 
import numpy as np
import random 


import matplotlib.pyplot as plt

import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily
import time
from transformers import AutoTokenizer, EsmForProteinFolding, EsmForMaskedLM



def esm_reward(sequences, esm_tokenizer, esm_model, **kwargs):
    fitnesses = []
    for aas in sequences:
        print(f"aas:::{aas}")
        inputs = esm_tokenizer([aas], return_tensors="pt", add_special_tokens=False).to(esm_model.device)  # A tiny random peptide
        with torch.no_grad():
            outputs = esm_model(**inputs)
        f = outputs.plddt[0,...,1].mean().item()
        fitnesses.append(f)
    print("fitnesses", fitnesses)

    return fitnesses

def esm_models_unit_test():
    DEVICE='cuda'
    test_seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

    esmfold_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", torch_dtype=torch.bfloat16).eval()
    esmfold_model = esmfold_model.to(DEVICE)
    esmfold_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    print("loaded ESMFold")

    start_time = time.time()
    plddts = esm_reward([test_seq], esmfold_tokenizer, esmfold_model)
    print("--- %s seconds for ESMFold forward pass ---" % (time.time() - start_time))


if __name__ == "__main__":
    esm_models_unit_test()
