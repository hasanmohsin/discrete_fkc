from abc import ABC, abstractmethod

import torch 
import numpy as np
import random 

from tqdm import tqdm  
import torch.nn.functional as F

import wandb 
from datetime import datetime  
import utils

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

class DiffusionSampler():

    def __init__(self, denoiser, steps=10, temperature=1.0):
        self.denoiser = denoiser
        self.device = denoiser.device
        
        if hasattr(denoiser, 'tokenizer'):
            self.tokenizer = denoiser.tokenizer
        else:
            self.tokenizer = None

        if hasattr(denoiser, 'log_prob_table'):
            self.log_prob_target = denoiser.log_prob_table
        else:
            self.log_prob_target = None

        self.steps = steps
        self.temperature = temperature

        # if denoiser has a length, set it, otherwise, set to None (and must be initialized with input_seq)
        if hasattr(self.denoiser, 'length'):
            self.length = self.denoiser.length
        else:
            self.length = None
        self.mask_token = self.denoiser.mask_token

    # sampling done with linear noise schedule alpha_t for now (default with LLADA)
    
    

    def get_num_transfer_tokens(self, mask_index, steps):
        '''
        In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
        Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
        the expected number of tokens transitioned at each step should be consistent.

        This function is designed to precompute the number of tokens that need to be transitioned at each step.
        '''
        mask_num = mask_index.sum(dim=1, keepdim=True)

        base = mask_num // steps
        remainder = mask_num % steps

        num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, :remainder[i]] += 1

        return num_transfer_tokens

    @ torch.no_grad()
    def sample(self, init_seq = None, batch_size = 10, cfg_scale = 0., remasking='low_confidence', return_traj = False, sampling_strat= "default", log_wandb = True):
        '''
            init_seq: A tensor of shape (1, L).
            remasking: Remasking strategy. 'low_confidence' or 'random'.
        '''
        #x = torch.full((batch_size, self.length), self.mask_token, dtype=torch.long).to(self.denoiser.device)
        #x[:, :init_seq.shape[1]] = init_seq.clone()

        if log_wandb:
            utils.setup_wandb_run(project = "discrete_fkc", 
                                  config = {"sampler": "default", 
                                   "denoiser_name": self.denoiser.name,
                                   "start_time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                   "steps": self.steps, 
                                   "temperature": self.temperature, 
                                   "cfg_scale": cfg_scale, 
                                   "remasking": remasking, 
                                   "sampling_strat": sampling_strat,
                                   "batch_size": batch_size})

        if init_seq is not None:
            x = init_seq.clone().to(self.denoiser.device)
            self.length = init_seq.shape[-1]
        else:
            if self.length is None:
                raise ValueError("self.length is None and no init_seq provided. Either provide init_seq or initialize denoiser with a length attribute.")
            x = torch.full((batch_size, self.length), self.mask_token, dtype=torch.long).to(self.denoiser.device)

        prompt_index = (x != self.mask_token)

        mask_index = (x == self.mask_token)
        num_transfer_tokens = self.get_num_transfer_tokens(mask_index, self.steps)

        #print("num_transfer tokens: ", num_transfer_tokens)

        if return_traj:
            x0_traj = []
            x_traj = []

        for i in tqdm(range(self.steps), 'Sampling'):
            
            mask_index = (x == self.mask_token)

            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = self.mask_token
                x_ = torch.cat([x, un_x], dim=0) # concat along batch dim
                logits = self.denoiser(x_)
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = self.denoiser(x)

            logits_with_noise = add_gumbel_noise(logits, temperature = self.temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
         
            if return_traj:
                x0_traj.append(x0.clone())

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'low_conf_noisy':
                p = F.log_softmax(logits_with_noise.log().to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

           
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            
        
            if log_wandb:
                
                utils.wandb_log_xt(i, logits, x, x0, self.tokenizer, self.log_prob_target, mask_token = self.mask_token)
                #wandb.log({"num_masked_tokens": int((x == self.mask_token).sum().item())})
               
            x[transfer_index] = x0[transfer_index]


            if return_traj:
                x_traj.append(x.clone())
    
        # log final sample
        if log_wandb:
            #wandb.log({"step": self.steps})
            utils.wandb_log_xt(self.steps, logits, x, x0, self.tokenizer, self.log_prob_target, mask_token = self.mask_token)
            #wandb.log({"num_masked_tokens": int((x == self.mask_token).sum().item())})


        if return_traj:
            return x, x0_traj, x_traj
        
        return x
            