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
    

