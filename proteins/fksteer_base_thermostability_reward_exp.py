import torch
import torch.nn as nn
import numpy as np
import os
import sys
from transformers import EsmForMaskedLM, AutoTokenizer
import warnings
import argparse 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fk_steering import FKSteeringSampler
from samplers import DiffusionSampler
from dplm_denoiser import DPLMDenoiser
from utils import set_all_seeds

from protein_reward import Thermostability

# Add the parent directory to Python path to access dplm
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from dplm.generate_dplm import initialize_generation

def main(args):
    device = 'cuda'
    denoiser = DPLMDenoiser(device=device)
    seed = args.seed 

    set_all_seeds(seed)

    scratch_dir = os.getenv('SCRATCH')
    hf_cache_dir = os.path.join(scratch_dir, 'huggingface_cache')

    tokenizer = denoiser.dplm.tokenizer  # Initialize your tokenizer here

    seq_length = args.seq_length  #50
    num_seqs = args.num_seqs

    reward_fn = Thermostability(device=device, 
                                tokenizer = denoiser.dplm.tokenizer, 
                                beta=args.beta, 
                                hf_cache_dir=hf_cache_dir)

    

    sampler = DiffusionSampler(denoiser=denoiser, steps=seq_length, temperature=1.0)
    r_sampler = FKSteeringSampler(denoiser=denoiser,
                              log_reward_func=reward_fn,
                              resample=True,
                              adaptive_resampling=False,
                              steps=seq_length,
                              temperature=1.0)

    input_seq = initialize_generation(
        length=seq_length,
        num_seqs=num_seqs,
        tokenizer=denoiser.dplm.tokenizer,
        device=device
    )


    x, x0, x_traj = sampler.sample(
        input_seq, batch_size = num_seqs, return_traj=True, remasking='low_conf_noisy', log_wandb=False)

    num_particles = args.num_particles  #5
    batch_num = args.batch_num 

    input_seq_2 = initialize_generation(
        length=seq_length,
        num_seqs=batch_num * num_particles,
        tokenizer=denoiser.dplm.tokenizer,
        device=device
    )

    input_seq_particles = input_seq_2.reshape(batch_num, num_particles, -1)
    
    set_all_seeds(seed)
    x_r, x0_r, x_traj_r, ess_traj, log_weights_traj = r_sampler.sample(input_seq_particles, 
                                                                       batch_size=batch_num, 
                                                                       num_particles = num_particles,
                                                                       remasking='low_conf_noisy',
                                                                       return_traj=True, 
                                                                        log_wandb=False,
                                                                        sim_mask_fill=True,
                                                                        clamp_val  = args.clamp_val,
                                                                        use_recent_r_i=args.recent_r_i)

    print("Unguided x: ", x)
    print("Guided x: ", x_r.view(-1, x_r.shape[-1]))

    # Compute the reward
    reward = reward_fn(x)
    print("Unguided Reward: ", reward.mean())
    print("Guided Reward: ", reward_fn(x_r).mean())

    # save guided and unguided as files

    tokenizer = denoiser.dplm.tokenizer

    output_tokens = x

    output_results = [
        "".join(seq.split(" "))
        for seq in tokenizer.batch_decode(
            output_tokens, skip_special_tokens=True
        )
    ]
    print("Unguided Results: ", output_results)

    output_results_guided = [
        "".join(seq.split(" "))
        for seq in tokenizer.batch_decode(
            x_r, skip_special_tokens=True
        )
    ]
    print("Guided Results: ", output_results_guided)

    saveto = args.save #"./dplm_out/reward_guided_esm2_uncond_true_mult_1_particle"

    os.makedirs(saveto, exist_ok=True)
    saveto_name = os.path.join(
        saveto, f"unguided_iter_{seq_length}_L_{seq_length}_beta_{reward_fn.beta}_num_particles_{num_particles}_seed_{seed}.fasta"
    )
    fp_save = open(saveto_name, "w")
    for idx, seq in enumerate(output_results):
        fp_save.write(f">SEQUENCE_{seq_length}_L={seq_length}\n")
        fp_save.write(f"{seq}\n")
    fp_save.close()

    # Save guided sequences
    saveto_name = os.path.join(
        saveto, f"guided_iter_{seq_length}_L_{seq_length}_beta_{reward_fn.beta}_num_particles_{num_particles}_seed_{seed}.fasta"
    )
    fp_save = open(saveto_name, "w")
    for idx, seq in enumerate(output_results_guided):
        fp_save.write(f">SEQUENCE_{seq_length}_L={seq_length}\n")
        fp_save.write(f"{seq}\n")
    fp_save.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Thermostability Reward with FK Steering")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--seq_length", type=int, default=50, help="Sequence length")
    parser.add_argument("--num_particles", type=int, default=5, help="Number of particles")
    parser.add_argument("--num_seqs", type=int, default=5, help="Number of sequences to generate")
    parser.add_argument("--beta", type=float, default=10.0, help="Reward scaling factor")
    parser.add_argument("--batch_num", type=int, default=1, help="Number of batches for reward sampling")
    parser.add_argument("--save", type=str, default="./dplm_out/reward_guided_thermo_mult_1_particle", help="Directory to save outputs")
    parser.add_argument("--recent_r_i", action='store_true', default = False, help="Use most recent r_i for weight updates")
    parser.add_argument("--clamp_val", type=float, default=-1.0, help="Clamp value for reward integration coefficient. Default: no clamping")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
