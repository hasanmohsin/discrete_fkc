import torch
import torch.nn as nn 
import numpy as np
import os 
import sys 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from dplm_denoiser import DPLMDenoiser
from samplers import DiffusionSampler
from smc_sampler import RewardSampler

from protein_esm2_reward import ESM2ProteinReward, ESM2ProteinRewardReference

# Add the parent directory to Python path to access dplm
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

#from byprot.models.dplm import DiffusionProteinLanguageModel as DPLM
from dplm.generate_dplm import initialize_generation



device = 'cuda'

ref = "MKLLVLSLVLVAPMAAQAAEITLVPSVKLQIGDRDNRGYYW"

seq_length = len(ref)
num_seqs = 1

denoiser = DPLMDenoiser(device=device)

# masked tokens [num_seqs, seq_length + 2] (2 exta are for bos and eos)
input_seq = initialize_generation(
        length=seq_length,
        num_seqs=num_seqs,
        tokenizer=denoiser.dplm.tokenizer,
        device=device
    )

# mask these positions
mask_pos = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

for i, aa in enumerate(ref):
    if i not in mask_pos:
        aa_id = denoiser.dplm.tokenizer.convert_tokens_to_ids(aa)
        input_seq[:, 1 + i] = aa_id


# decode and check
print("Input seq: ", input_seq)
print("Decoded input seq: ", denoiser.tokenizer.batch_decode(input_seq, skip_special_tokens=False))

# try inference in dplm 
sampler = DiffusionSampler(denoiser=denoiser, steps=seq_length, temperature=1.0)

x, x0, x_traj = sampler.sample(
        input_seq, return_traj=True, remasking='low_conf_noisy')

print("Output seq: ", x)
print("Decoded output seq: ", denoiser.tokenizer.batch_decode(x, skip_special_tokens=False))

