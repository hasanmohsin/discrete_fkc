import os 
import sys 
import torch 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from samplers import DiffusionSampler
from prod_smc_samplers import ProductPromptSampler
from dream_denoiser import DreamDenoiser
from llada_denoiser import LLaDADenoiser
from utils import *


def test_simple():
    dream_denoiser = DreamDenoiser(device='cuda')

    sampler = DiffusionSampler(dream_denoiser, steps=128, temperature=1.0)
    init_seq = dream_denoiser.prepare_init_seq(["Implement bubble sort."], gen_length = 128)

    print("Decoded input: ", dream_denoiser.tokenizer.batch_decode(init_seq, skip_special_tokens=True))
    print("Input: ", init_seq)

    out = sampler.sample(init_seq=init_seq, batch_size=1, log_wandb=True)

    out_decoded = dream_denoiser.tokenizer.batch_decode(out, skip_special_tokens=True)
    
    print("Output: ", out)
    print("Decoded output: ", out_decoded)

    return 


def main():
    test_simple()
    return


if __name__ == "__main__":
    main()

    exit() 