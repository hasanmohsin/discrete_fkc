from abc import ABC, abstractmethod

import torch 
import numpy as np
from tqdm import tqdm
import random 
from datetime import datetime 
import wandb 

import scipy 

import torch.nn.functional as F

import utils 
from samplers import DiffusionSampler, add_gumbel_noise
from smc_sampler import SMCSampler
from resample import systematic_resample

# FK steering with base distribution 
class FKSteeringSampler(SMCSampler):
    def __init__(self, denoiser, log_reward_func, resample = True, adaptive_resampling = False, steps=10, temperature=1.0, partial_cont = False):
        super().__init__(denoiser, resample, adaptive_resampling, steps, temperature)
        self.log_reward_func = log_reward_func
        self.anneal_schedule = lambda i: (i) / self.steps

        # when starting with partially masked sequence, start count at len - number of unmasked_tokens
        self.partial_cont = partial_cont

        self.sampling_strat = "fk_steer_rewardSMC"

    # proposal logits is [B*M, L, V]
    def get_log_weight_update(self, base_logits, r_i_new, r_i, step):
        ###########################################
        # log weights for resampling 
        
        log_mu = base_logits
        
        coeff_beta_new = self.anneal_schedule(step+1)
        coeff_beta = self.anneal_schedule(step)

        g = coeff_beta_new * r_i_new - coeff_beta * r_i # [B, ]

        return g

    # the log_diff term can be integrated with euler scheme or with explicit integral (if dt is too big)
    def log_diff_term(self, step, r_j_val, r_i_val):

        if self.steps < 5:
            log_r_diff_const = r_j_val - r_i_val

            diff_val = log_r_diff_const.exp()  
            return self.anneal_schedule(step) * (r_j_val - r_i_val)
        else:
            diff_val = self.anneal_schedule(step) * (r_j_val - r_i_val)
   
        return diff_val 

    # evaluate ratio exp( beta * reward(j))/exp( beta * reward(i)), with i = x, j = all Hamming 1 neighbours of x
    # logits are of shape [B, L, V]
    # select_idx is the list of indices that will be unmasked this step (based on base_logits)
    def eval_logr_diffs(self, x, logits, step, base_transfer_idx):
        logits = logits.log_softmax(dim=-1)
        
        # shape [B, ]
        x0_i = self.mask_fill_strat(x)  # [B, L]
        r_i = self.log_reward_func(x0_i)

        # there are M*V neighbours, where M = number of masked tokens, V = vocab size
        masked_idx = (x == self.mask_token)
        num_masked = masked_idx.sum(dim=-1) #[B,] - number of masked for each sample in batch

        r_tilted_logits = torch.clone(logits)

        B, L, V = logits.shape
        r_j = r_i.reshape((B, 1, 1)).repeat(1, L, V)  # [B, L, V]

        print("B: ", B)
        print("L: ", L)
        print("V: ", V)
        

        row_idx, selected_idx = torch.where(base_transfer_idx)      
        


        assert(row_idx.shape[0] == B)

        for b in range(B): 
            # if masked, change to v for all in batch
            # masked = (x[b, l] == self.mask_token)
            l = selected_idx[b]

            for v in range(V):
                
                
                #if masked: 
                x_neighbour = x[b, ...].clone() #[L,]

                #print("x neighbour size: ", x_neighbour.shape)

            
                x_neighbour[l] = v

                if self.sim_mask_fill:
                    x0_n = self.mask_fill_strat(x_neighbour.unsqueeze(0), b=b)  # [1, L, V]
                else:
                    x0_n = self.mask_fill_strat(x_neighbour.unsqueeze(0))  # [5, L, V], pick one at random, injects noise

                r_j[b, l, v] = self.log_reward_func(x0_n)[0] # should be equal to r_i at ~masked_in_batch positions
                #else:
                #    r_j[b, l, v] = r_i[b]  # if not masked, reward is same as r_i

                r_tilted_logits[b, l, v] = logits[b, l, v] + self.anneal_schedule(step) * (r_j[b, l, v] - r_i[b])

                


        print("r_tilted_logits shape: ", r_tilted_logits.shape)
        print("r_i shape: ", r_i.shape)
        print("r_j shape: ", r_j.shape)
        #r_tilted_logits = logits 
        #r_i = torch.tensor(0.) .to(self.denoiser.device)
        #r_j = torch.zeros((B,L,V)).to(self.denoiser.device)

        return r_tilted_logits, r_j, r_i

    # takes partially masked token and fills in masked tokens with some strategy
    # for reward eval
    def mask_fill_strat(self, x, b=-1):
        
        # theoretically correct way, requires call to denoiser
        # instead of argmax, sample from denoiser
        
        #logits = self.denoiser(x)
        #B, L, V = logits.shape

        #x_fill = torch.multinomial(logits.exp().reshape(B*L, V), num_samples=1).squeeze(-1).reshape(B,L)
        #x_fill shape is [B,L]
        #x_fill = torch.argmax(self.denoiser(x), dim=-1) 
        #x_fill = torch.where(x == self.mask_token, x_fill, x)  # fill in masked tokens        

        # other way

        if b >= 0:
            x_fill = torch.where(x == self.mask_token, self.x0_base, x)[b,:].unsqueeze(0)
        else:
            x_fill = torch.where(x == self.mask_token, self.x0_base, x)
        return x_fill

    def get_base_transfer_idx(self, x, i, base_logits, mask_index, num_transfer_tokens, remasking='low_confidence'):
        logits_with_noise = add_gumbel_noise(base_logits, temperature = self.temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1) 
        
        if remasking == 'low_confidence':
            p = F.softmax(base_logits.to(torch.float64), dim=-1)
            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
        elif remasking == 'low_conf_noisy':
            p = F.softmax(logits_with_noise.log().to(torch.float64), dim=-1)
            x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
        elif remasking == 'random':
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        else:
            raise NotImplementedError(remasking)

        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)

        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        
        #print("logits shape: ", base_logits.shape)
        #print("transfer index shape: ", transfer_index.shape)

        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
            transfer_index[j, select_index] = True

        return transfer_index, x0

    # batch size is the number of particles
    @torch.no_grad()
    def sample(self, init_seq = None, batch_size = 2, num_particles = 5, cfg_scale = 0., remasking='low_confidence', return_traj = False, 
               log_wandb = False, eos_bos = False, sim_mask_fill = False, clamp_val=-1, use_recent_r_i=False):
        
        self.sim_mask_fill = sim_mask_fill 
        self.clamp_val = clamp_val
        self.use_recent_r_i = use_recent_r_i

        if init_seq is not None:
            x = init_seq.clone().to(self.denoiser.device)
            self.length = x.shape[-1]
            
            if eos_bos:
                self.actual_len = x.shape[-1] -2
            else:
                self.actual_len = x.shape[-1]

            if self.partial_cont:
                # assuming same number of unmasked tokens in each sequence in the batch
                num_masked = (x == self.mask_token).sum(dim=-1)[0, 0].item()
                self.step_start_val = (self.actual_len) - num_masked
                self.eff_steps = self.actual_len 
                print("Eff steps: ", self.eff_steps)
                print("Starting reward annealing schedule at step: ", self.step_start_val)
                self.anneal_schedule = lambda i: (i + self.step_start_val) / self.eff_steps
            else:
                self.step_start_val = 0
                self.eff_steps = self.steps

        else:
            self.eff_steps = self.steps 
            self.step_start_val = 0
            self.actual_len = self.length

            if self.length is None:
                raise ValueError("self.length is None and no init_seq provided. Either provide init_seq or initialize denoiser with a length attribute.")
            x = torch.full((batch_size, num_particles, self.length), self.mask_token, dtype=torch.long).to(self.denoiser.device)

        if log_wandb:
            if hasattr(self.log_reward_func, 'beta'):
                self.reward_beta = self.log_reward_func.beta
            else:
                self.reward_beta = 1.0

            utils.setup_wandb_run(project = "discrete_fkc", 
                                  config = {"sampler": "RewardSMC", 
                                    "denoiser_name": self.denoiser.name,
                                    "reward": self.log_reward_func.name,
                                    "reward_beta": self.reward_beta,
                                   "start_time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                   "steps": self.steps, 
                                   "temperature": self.temperature, 
                                   "cfg_scale": cfg_scale, 
                                   "remasking": remasking, 
                                   "sampling_strat": self.sampling_strat,
                                   "batch_size": batch_size,
                                   "num_particles": num_particles,
                                   "step_start_val": self.step_start_val,
                                   "eff_steps": self.eff_steps,
                                   "eos_bos": eos_bos,
                                   "sim_mask_fill": sim_mask_fill,
                                   "partial_cont": self.partial_cont})

        # for resampling 
        log_weights = torch.zeros((batch_size, num_particles, 1), dtype=torch.float32).to(self.denoiser.device)

        # reshape 
        x = x.view(batch_size * num_particles, self.length)
        log_weights = log_weights.view(batch_size * num_particles, 1)

        prompt_index = (x != self.mask_token)

        mask_index = (x == self.mask_token)
        num_transfer_tokens = self.get_num_transfer_tokens(mask_index, self.steps)
        #print("num_transfer tokens: ", num_transfer_tokens)


        if return_traj:
            x0_traj = []
            x_traj = []
            log_weights_traj = []
            ess_traj = []

        for i in tqdm(range(self.steps), "Sampling"):
            
            mask_index = (x == self.mask_token)

            base_logits = self.denoiser(x)
            base_logits = base_logits.log_softmax(dim=-1)

            base_transfer_idx, x0_base = self.get_base_transfer_idx(x, i, base_logits, mask_index, num_transfer_tokens, remasking=remasking)

            # to optionally use in mask_fill strat
            self.x0_base = x0_base

            # proposal with modified inv temp beta
            x0_i = self.mask_fill_strat(x) 
            r_i = self.log_reward_func(x0_i) # [B,]

            logits_with_noise = add_gumbel_noise(base_logits, temperature = self.temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if return_traj:
                x0_traj.append(x0.clone())

            
            x0 = torch.where(mask_index, x0, x)
            #confidence = torch.where(mask_index, x0_p, -np.inf)

            #transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            #for j in range(confidence.shape[0]):
            #    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
            #    transfer_index[j, select_index] = True
            #
            transfer_index = base_transfer_idx

            x_pre_unmask = x.clone().view(batch_size, num_particles, self.length)

            x[transfer_index] = x0[transfer_index]
            
            if return_traj:
                x_traj.append(x.clone())


            x_pre_resample = x.view(batch_size, num_particles, self.length).clone()

            # update weights
            # update using r_i on most recent tokens (amounts to integrating log weights in different way)
            r_i_new = self.log_reward_func(x0) # [B,] - calculated on most recent x0 rather than estimated one


            g = self.get_log_weight_update(base_logits, r_i_new, r_i, step = i)
           
            log_weights = log_weights + g.unsqueeze(-1) #* num_transfer_tokens[:, i].unsqueeze(-1) / self.steps # num_samples, 1

            # for numerical stability
            #log_weights = log_weights - log_weights.max(dim=0, keepdim=True).values
            log_weights_r = log_weights.view(batch_size, num_particles, 1)
            log_weights_norm = log_weights_r - log_weights_r.logsumexp(dim=1, keepdim=True)
            
            #print("\nstep ", i)
            #print("log weights: ", log_weights_norm[0, :])

            x_r = x.view(batch_size, num_particles, self.length)

            # resample (weights may be set to 0 after resampling)
            ess_batch = []
            for b in range(batch_size):
                #print("x_r shape: ", x_r.shape)
                #print("log_weights_norm shape: ", log_weights_norm.shape)
                #print("num_particles: ", num_particles)

                x_r[b], log_weights_r[b], ess_b = self.resample_op(log_weights_norm[b], x_r[b], num_particles = num_particles, i = i)
                ess_batch.append(ess_b.item())

            x = x_r.view(batch_size * num_particles, self.length)
            log_weights = log_weights_r.view(batch_size * num_particles, 1)

            if return_traj:
                ess_traj.append(ess_batch)
                log_weights_traj.append(log_weights_norm)


            if log_wandb:
                
                log_info = utils.wandb_log_xt_smc(
                                       step = i, 
                                       logits_prop = base_logits, 
                                       x_pre_unmask = x_pre_unmask, 
                                       x_pre_resample = x_pre_resample,
                                       x_r = x_r, 
                                       x0 = x0,
                                       tokenizer = self.tokenizer,
                                       log_weights_r = log_weights_norm, 
                                       ess_batch = ess_batch,
                                       additional_metrics = {"reward": r_i},
                                       log_prob_target = self.log_prob_target,
                                       mask_token = self.mask_token,
                                       show_logits = False)
                #log_info["samples"].add_column("Reward", r_i.cpu().numpy())
                wandb.log(log_info, step=i)
                wandb.log({"Mean Reward": r_i.mean().item(), "Reward STD": r_i.std().item()}, step=i)
                

        if log_wandb:
            # final log for wandb, to show all particles unmasked
            rewards_final = self.log_reward_func(x_r.view(batch_size * num_particles, self.length))
            log_info = utils.wandb_log_xt_smc(
                                   step = i+1, 
                                   logits_prop = logits,
                                   x_pre_unmask = x_pre_unmask,
                                   x_pre_resample = x_pre_resample,
                                   x_r = x_r,
                                   x0 = x0,
                                   tokenizer = self.tokenizer,
                                   log_weights_r = log_weights_r, # will be 0's, since after each resampling step, the log weights are set to 0
                                   ess_batch = ess_batch,
                                   additional_metrics = {"reward": rewards_final},
                                   log_prob_target = self.log_prob_target,
                                   mask_token = self.mask_token,
                                   show_logits = False)

            #log_info["samples"].add_column("Reward", rewards_final.cpu().numpy())
            wandb.log(log_info, step=i+1)
            wandb.log({"Mean Reward": rewards_final.mean().item(), "Reward STD": rewards_final.std().item()}, step=i+1)
        
        wandb.finish()
        
        if return_traj:
            return x, x0_traj, x_traj, ess_traj, log_weights_traj
        
        return x
    
