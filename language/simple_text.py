import os 
import sys 
import torch 

from lang_utils import * 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from samplers import DiffusionSampler
from prod_smc_samplers import ProductPromptSampler
from llada_denoiser import LLaDADenoiser
from utils import *


def test_simple():
    llada_denoiser = LLaDADenoiser(device='cuda')

    sampler = DiffusionSampler(llada_denoiser, steps=128, temperature=1.0)
    init_seq = llada_denoiser.prepare_init_seq(["What is the capital of France?", "What is 5*5 + 10 - 22?"], gen_length = 128)

    print("Decoded input: ", llada_denoiser.tokenizer.batch_decode(init_seq, skip_special_tokens=True))
    print("Input: ", init_seq)

    out = sampler.sample(init_seq=init_seq, batch_size=2, log_wandb=True)

    out_decoded = llada_denoiser.tokenizer.batch_decode(out, skip_special_tokens=True)
    
    print("Output: ", out)
    print("Decoded output: ", out_decoded)

    return 


def main():
    test_simple()
    return


if __name__ == "__main__":
    main()

    exit() 