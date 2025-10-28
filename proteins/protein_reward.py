from huggingface_hub import hf_hub_download
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

from transformers import EsmForMaskedLM, EsmTokenizer

from replearning_dplm._dplm.dplm_regression_model import DPLMRegressionModel

# Add the parent directory to Python path to access dplm
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

#from byprot.models.dplm import DiffusionProteinLanguageModel as DPLM
from dplm.generate_dplm import initialize_generation

class Thermostability():
    def __init__(self, device, tokenizer, beta = 1.0, hf_cache_dir=None):
        self.tokenizer = tokenizer  # DPLM/denoiser tokenizer
        self.beta = beta
        self.device = device 
        
        # pick one file from the repo file list
        repo_id = "airkingbd/dplm_representation_learning"
        filename = "Thermostability_dplm_650m.ckpt"  # or any other .ckpt from the list

        self.model_name = "Thermostability_dplm_650m"

        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=hf_cache_dir)
        ckpt = torch.load(ckpt_path, map_location=device)

        print(f"Loaded checkpoint from {ckpt_path}")

        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        self.model = DPLMRegressionModel(test_result_path="./dplm_reg_out/", net_name = "airkingbd/dplm_650m",
            load_pretrained = True, 
            load_prev_scheduler= True,
            save_top_k= 1,
            freeze_backbone= 0,
            dropout= 0.0,
            use_lora= 0)

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(device)
                
        self.model.eval()
        
        print(f"Loaded DPLM Regression model: {self.model_name}")

    def __call__(self, input_seq):
        """
        Make the class callable for use with RewardSampler.
        """
        batch_size = input_seq.shape[0]
        #log_rewards = []
       
        with torch.no_grad():
            outputs = self.model({'input_ids': input_seq.to(self.device)})

        print("Thermostability outputs: ", outputs)

        return self.beta * torch.log(outputs)
    
class SubstringReward():
    def __init__(self, target_string, tokenizer, device, beta = 1.0):
        
        self.name = "SubstringReward_Targ_" + target_string
        
        self.target_string = target_string
        self.tokenizer = tokenizer

        # find token ids for target string
        self.target_ids = tokenizer.encode(target_string, return_tensors="pt")[0]

        # eliminate first and last token, since its bos and eos tokens
        self.target_ids = self.target_ids[1:-1]
        
        if len(self.target_ids) == 1:
            self.target_ids = self.target_ids[0]
    
        self.beta = beta
        self.device = device

    # outputs log reward of target
    # same device as input seq 
    def __call__(self, input_seq):

        # batch of input_seq [B, L]

        # count number of times the target id seq appears in each sample 
        logr_val = (input_seq == self.target_ids).sum(dim=-1)

        # tempering: 
        logr_val = self.beta * logr_val
        logr_val = logr_val.to(self.device)

        return logr_val 

class BetaSheetReward():
    def __init__(self, device, beta = 1.0):
        self.beta = beta
        self.device = device 

    def __call__(self, input_seq):
        pass

def main():
    device = 'cuda'
    denoiser = DPLMDenoiser(device=device)

    tokenizer = denoiser.dplm.tokenizer  # Initialize your tokenizer here
    
    scratch_dir = os.getenv('SCRATCH')
    hf_cache_dir = os.path.join(scratch_dir, 'huggingface_cache')



    
    #target_string = "N"
    #reward_fn = SubstringReward(target_string, tokenizer, device = device)
    #print("Target IDs: ", reward_fn.target_ids)

    reference_sequence = "SFNTVDEWLEAIKMGQYKESFANAGFTSFDVVSQMMMEDILRVGVTLAGHQKKILNSIQVMRAQM"
    reward_fn = ESM2ProteinRewardReference(device=device, reference_sequence= reference_sequence, 
                                  tokenizer = denoiser.dplm.tokenizer, beta=10.0, hf_cache_dir=hf_cache_dir)

    seq_length = len(reference_sequence)

    print(f"Sequence length: {seq_length}")

    input_seq = initialize_generation(
    length=seq_length,
    num_seqs=2,
    tokenizer=denoiser.dplm.tokenizer,
    device=device
    )

    sampler = DiffusionSampler(denoiser=denoiser, steps = 5, temperature=1.0)
    r_sampler = RewardSampler(denoiser = denoiser,
                              log_reward_func=reward_fn, 
                              resample=True, 
                              adaptive_resampling=False,
                              steps = seq_length,
                              temperature=1.0)
    x, x0, x_traj = sampler.sample(input_seq, return_traj=True, remasking='low_conf_noisy', log_wandb=True)

    num_particles = 2
    batch_num = 1

    input_seq_particles = input_seq.reshape(batch_num, num_particles, -1)

    x_r, x0_r, x_traj_r, ess_traj, log_weights_traj = r_sampler.sample(input_seq_particles, return_traj=True, remasking='low_conf_noisy',
                     batch_size = batch_num, num_particles = num_particles,
                     log_wandb=True)

    print("Unguided x: ", x)
    print("Guided x: ", x)

    # Compute the reward
    reward = reward_fn(x)
    print("Unguided Reward: ", reward.mean())
    print("Guided Reward: ", reward_fn(x_r).mean())


    # save guided and unguided as files
    
    tokenizer = denoiser.dplm.tokenizer

    output_tokens= x

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

    saveto = "./dplm_out/reward_guided"

    os.makedirs(saveto, exist_ok=True)
    saveto_name = os.path.join(
        saveto, f"unguided_iter_{seq_length}_L_{seq_length}.fasta"
    )
    fp_save = open(saveto_name, "w")
    for idx, seq in enumerate(output_results):
        fp_save.write(f">SEQUENCE_{seq_length}_L={seq_length}\n")
        fp_save.write(f"{seq}\n")
    fp_save.close()

    # Save guided sequences
    saveto_name = os.path.join(
        saveto, f"guided_iter_{seq_length}_L={seq_length}.fasta"
    )
    fp_save = open(saveto_name, "w")
    for idx, seq in enumerate(output_results_guided):
        fp_save.write(f">SEQUENCE_{seq_length}_L={seq_length}\n")
        fp_save.write(f"{seq}\n")
    fp_save.close()

if __name__ == "__main__":
    main()