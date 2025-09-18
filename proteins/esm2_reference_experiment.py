import torch
import torch.nn as nn
import numpy as np
import os
import sys
from transformers import EsmForMaskedLM, AutoTokenizer
import warnings
import argparse 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smc_sampler import RewardSampler
from samplers import DiffusionSampler
from dplm_denoiser import DPLMDenoiser
from utils import set_all_seeds

from protein_esm2_reward import ESM2ProteinRewardReference

# Add the parent directory to Python path to access dplm
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from dplm.generate_dplm import initialize_generation

def create_masked_input_sequence(reference_seq, mask_positions, denoiser, num_seqs, device):
    """
    Create input sequence with specific positions masked and others fixed to reference
    
    Args:
        reference_seq: Reference protein sequence string
        mask_positions: List of 0-indexed positions to mask (e.g., [10, 15, 17])
        denoiser: DPLMDenoiser instance
        num_seqs: Number of sequences in batch
        device: Device to place tensors on
    """
    seq_length = len(reference_seq)
    
    # Initialize with all masked tokens
    input_seq = initialize_generation(
        length=seq_length,
        num_seqs=num_seqs,
        tokenizer=denoiser.dplm.tokenizer,
        device=device
    )
    
    # Set non-masked positions to reference amino acids
    for i, aa in enumerate(reference_seq):
        if i not in mask_positions:
            aa_id = denoiser.dplm.tokenizer.convert_tokens_to_ids(aa)
            input_seq[:, 1 + i] = aa_id  # +1 for BOS token
    
    return input_seq

def parse_mask_string(mask_string):
    """
    Parse mask string to extract positions that should be masked
    
    Args:
        mask_string: String like "SFNTVDEWLE<MASK>IKMGQYKESF<MASK>N<MASK>GFTSFDVVSQMMMEDILRVGVTL<MASK>GHQKKILNSIQVMR<MASK>QM"
    
    Returns:
        List of 0-indexed positions to mask
    """
    mask_positions = []
    position = 0
    
    i = 0
    while i < len(mask_string):
        if mask_string[i:i+6] == '<MASK>':
            mask_positions.append(position)
            i += 6  # Skip the <MASK> token
            position += 1
        else:
            # Regular amino acid
            i += 1
            position += 1
    
    return mask_positions

def main(args):
    device = 'cuda'
    denoiser = DPLMDenoiser(device=device)
    seed = args.seed

    set_all_seeds(seed)

    scratch_dir = os.getenv('SCRATCH')
    hf_cache_dir = os.path.join(scratch_dir, 'huggingface_cache')

    tokenizer = denoiser.dplm.tokenizer

    # Reference sequence and mask specification
    reference_seq = "SFNTVDEWLEAIKMGQYKESFANAGFTSFDVVSQMMMEDILRVGVTLAGHQKKILNSIQVMRAQM"
    mask_string = "SFNTVDEWLE<MASK>IKMGQYKESF<MASK>N<MASK>GFTSFDVVSQMMMEDILRVGVTL<MASK>GHQKKILNSIQVMR<MASK>QM"
    
    seq_length = len(reference_seq)
    num_seqs = args.num_seqs
    
    # Parse mask positions from the mask string
    mask_positions = parse_mask_string(mask_string)
    print(f"Reference sequence: {reference_seq}")
    print(f"Sequence length: {seq_length}")
    print(f"Mask positions: {mask_positions}")
    print(f"Number of masked positions: {len(mask_positions)}")

    # Initialize ESM2 reward function with reference sequence
    reward_fn = ESM2ProteinRewardReference(
        reference_sequence=reference_seq,
        tokenizer=tokenizer,
        beta=args.beta,
        hf_cache_dir=hf_cache_dir,
        device=device
    )

    # Create samplers
    sampler = DiffusionSampler(denoiser=denoiser, steps=seq_length, temperature=1.0)
    r_sampler = RewardSampler(denoiser=denoiser,
                              log_reward_func=reward_fn,
                              resample=True,
                              adaptive_resampling=False,
                              steps=seq_length,
                              temperature=1.0)

    # Create masked input sequence for unguided sampling
    input_seq = create_masked_input_sequence(
        reference_seq, mask_positions, denoiser, num_seqs, device
    )

    print("Input sequence tokens:", input_seq)
    print("Decoded input sequence:", tokenizer.batch_decode(input_seq, skip_special_tokens=False))

    # Unguided sampling
    x, x0, x_traj = sampler.sample(
        input_seq, batch_size=num_seqs, return_traj=True, remasking='low_conf_noisy', log_wandb=False)

    # Guided sampling setup
    num_particles = args.num_particles
    batch_num = 1

    # Create masked input for guided sampling
    input_seq_2 = create_masked_input_sequence(
        reference_seq, mask_positions, denoiser, batch_num * num_particles, device
    )

    input_seq_particles = input_seq_2.reshape(batch_num, num_particles, -1)
    
    set_all_seeds(seed)
    x_r, x0_r, x_traj_r, ess_traj, log_weights_traj = r_sampler.sample(input_seq_particles, 
                                                                       batch_size=batch_num, 
                                                                       num_particles=num_particles,
                                                                       remasking='low_conf_noisy',
                                                                       return_traj=True, 
                                                                       log_wandb=False)

    print("Unguided x: ", x)
    print("Guided x: ", x_r.view(-1, x_r.shape[-1]))

    # Compute the reward
    reward = reward_fn(x)
    print("Unguided Reward: ", reward.mean())
    print("Guided Reward: ", reward_fn(x_r).mean())

    # Decode sequences
    output_results = [
        "".join(seq.split(" "))
        for seq in tokenizer.batch_decode(
            x, skip_special_tokens=True
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

    # Save results
    saveto = "./dplm_out/reward_guided_esm2_reference"

    os.makedirs(saveto, exist_ok=True)
    
    # Save unguided sequences
    saveto_name = os.path.join(
        saveto, f"unguided_ref_L_{seq_length}_beta_{reward_fn.beta}_num_particles_{num_particles}_seed_{seed}.fasta"
    )
    fp_save = open(saveto_name, "w")
    for idx, seq in enumerate(output_results):
        fp_save.write(f">SEQUENCE_REF_L={seq_length}_UNGUIDED_{idx}\n")
        fp_save.write(f"{seq}\n")
    fp_save.close()

    # Save guided sequences
    saveto_name = os.path.join(
        saveto, f"guided_ref_L_{seq_length}_beta_{reward_fn.beta}_num_particles_{num_particles}_seed_{seed}.fasta"
    )
    fp_save = open(saveto_name, "w")
    for idx, seq in enumerate(output_results_guided):
        fp_save.write(f">SEQUENCE_REF_L={seq_length}_GUIDED_{idx}\n")
        fp_save.write(f"{seq}\n")
    fp_save.close()

    # Save reference sequence for comparison
    saveto_name = os.path.join(
        saveto, f"reference_sequence_L_{seq_length}.fasta"
    )
    fp_save = open(saveto_name, "w")
    fp_save.write(f">REFERENCE_SEQUENCE_L={seq_length}\n")
    fp_save.write(f"{reference_seq}\n")
    fp_save.close()

    print(f"Results saved to {saveto}")
    print(f"Reference sequence: {reference_seq}")
    print(f"Masked positions: {mask_positions}")


def parse_args():
    parser = argparse.ArgumentParser(description="ESM2 Reference-based Reward Experiment")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--num_particles", type=int, default=5, help="Number of particles for guided sampling")
    parser.add_argument("--num_seqs", type=int, default=5, help="Number of sequences to generate for unguided sampling")
    parser.add_argument("--beta", type=float, default=200.0, help="Reward scaling factor")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
