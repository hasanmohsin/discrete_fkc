import os 
import sys 
import torch 

from lang_utils import * 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from samplers import DiffusionSampler
from prod_smc_samplers import ProductPromptSampler
from llada_denoiser import LLaDADenoiser
from utils import *
from eval import * 

def story_gen():

    set_all_seeds(42)

    llada_denoiser = LLaDADenoiser(device='cuda')

    base_prompt = "Write a short story."
    cond1 = " The story should include a hungry cat."
    cond2 = " The story should include a brave dog."

    joint_prompt = base_prompt + cond1 + cond2

    prompt_list = [base_prompt + cond1, base_prompt + cond2]

    sampler = DiffusionSampler(llada_denoiser, steps=128, temperature=1.0)
    prod_sampler = ProductPromptSampler(llada_denoiser, resample=True, 
                                        steps=128, temperature=1.0)

    input_ids = llada_denoiser.get_input_ids_template(prompt_list)

    joint_template = llada_denoiser.apply_prompt_template([joint_prompt])

    jp_init_seq = llada_denoiser.prepare_init_seq([joint_prompt], gen_length = 128)

    print("Decoded input: ", llada_denoiser.tokenizer.batch_decode(input_ids, skip_special_tokens=True))
    print("Input: ", input_ids)

    # Joint Prompt
    out = sampler.sample(init_seq=jp_init_seq, batch_size=2, log_wandb=True)
    wandb.finish() 

    out_decoded = llada_denoiser.tokenizer.batch_decode(out, skip_special_tokens=True)
    
    # Product prompt 
    _, prod_out, _ = prod_sampler.sample(prompt_list=input_ids, steps=128, gen_length=128, block_length=128, num_samples=1)
    prod_out_decoded = llada_denoiser.tokenizer.batch_decode(prod_out, skip_special_tokens=True)
    #print("Joint Prompt Output: ", out)
    print("Joint prompt output: ", out_decoded)
    print("Product prompt output: ", prod_out_decoded)

    # add template before prod out
    for i in range(len(prod_out_decoded)):
        prod_out_decoded[i] = joint_template[0] + prod_out_decoded[i]

    #Perplexities under Qwen
    ppl = QwenPPL(task='text')
    jp_ppl = ppl(out_decoded)
    prod_ppl = ppl(prod_out_decoded)

    print("Joint Prompt PPL: ", jp_ppl)
    print("Product Prompt PPL: ", prod_ppl)

    wandb.log({"Joint Prompt": joint_prompt, "Product Prompt": prompt_list}, step=128)
    wandb.log({"Joint Prompt PPL": jp_ppl, "Product Prompt PPL": prod_ppl}, step=128)

    return 


def main():
    story_gen()
    return


if __name__ == "__main__":
    main()

    exit() 