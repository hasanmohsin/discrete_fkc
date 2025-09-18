from abc import ABC, abstractmethod

import torch 
import numpy as np
from tqdm import tqdm
import random 
from datetime import datetime 
import wandb 

import torch.nn.functional as F

import utils 
from samplers import DiffusionSampler, add_gumbel_noise
from resample import systematic_resample

class SMCSampler(DiffusionSampler):
    def __init__(self, denoiser, resample = True, adaptive_resampling = False, steps=10, temperature=1.0):
        super().__init__(denoiser, steps, temperature)
        
        self.resample = resample 
        self.adaptive_resampling = adaptive_resampling


    # log_weights shape [M, 1], x shape: [M, L]
    def resample_op(self, log_weights, x, num_particles, i):
        weights = log_weights.exp()

        ess = 1. / (weights ** 2).sum()
        

        if num_particles > 1 and self.resample:
            if self.adaptive_resampling:
                if ess < num_particles / 2 or i == self.steps - 1:
                    # resample if ESS is low or at the last step
                    print("\n\nResampling at step {} with ESS: {}".format(i, ess.item()))
                    resample_inds = systematic_resample(weights)
                    x = x[resample_inds, :]
                    log_weights = torch.zeros((num_particles, 1), dtype=torch.float32).to(self.denoiser.device)
                    
            else:
                # resample every step   
                resample_inds = systematic_resample(weights)
                x = x[resample_inds, :]
                log_weights = torch.zeros((num_particles, 1), dtype=torch.float32).to(self.denoiser.device)
        else:
            # no resampling 
            log_weights = torch.zeros((num_particles, 1), dtype=torch.float32).to(self.denoiser.device)

        return x, log_weights, ess
    

class AnnealSampler(SMCSampler):
    def __init__(self, denoiser, beta, resample = True, adaptive_resampling = False, steps=10, temperature=1.0):
        super().__init__(denoiser, resample, adaptive_resampling, steps, temperature)
        self.beta = beta
        self.sampling_strat = 'annealSMC'

    # proposal logits is [B*M, L, V]
    def get_log_weight_update(self, base_logits, i):
        ###########################################
        # log weights for resampling 
        
        log_mu = base_logits.log_softmax(dim=-1)

        # t=0 corresponds to fully unmasked, and t=1 to fully masked
        t = 1 - torch.tensor(i/self.steps)
        
        alpha_t = self.get_alpha_t(i)
        neg_over_t_ratio = self.get_elbo_weight(i) #-1/t
        offset = self.beta * neg_over_t_ratio #- B / t
    
        coeff = self.beta * (i)**(self.beta - 1) / (self.steps - i)**self.beta 
        log_denoiser_anneal = (log_mu)*self.beta 
        score_anneal_sum = (coeff * log_denoiser_anneal.exp()).sum(dim=-1)  # b, l

        # get a number per token 
        g_all_tok = score_anneal_sum + offset  # b, l 
        
        return g_all_tok

    
    # batch size is the number of particles
    @torch.no_grad()
    def sample(self, init_seq = None, batch_size = 2, num_particles = 5, cfg_scale = 0., remasking='low_confidence', return_traj = False, log_wandb = True):
        if log_wandb:
            utils.setup_wandb_run(project = "discrete_fkc", 
                                  config = {"sampler": "AnnealSMC", 
                                    "denoiser_name": self.denoiser.name,
                                    "anneal_beta": self.beta,
                                   "start_time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                   "steps": self.steps, 
                                   "temperature": self.temperature, 
                                   "cfg_scale": cfg_scale, 
                                   "remasking": remasking, 
                                   "sampling_strat": self.sampling_strat,
                                   "batch_size": batch_size,
                                   "num_particles": num_particles})

        if init_seq is not None:
            x = init_seq.clone().to(self.denoiser.device)
        else:
            x = torch.full((batch_size, num_particles, self.length), self.mask_token, dtype=torch.long).to(self.denoiser.device)

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

        for i in tqdm(range(self.steps)):
            
            mask_index = (x == self.mask_token)

            base_logits = self.denoiser(x)

            # proposal with modified inv temp beta
            logits = self.beta  * base_logits

            logits_with_noise = add_gumbel_noise(logits, temperature = self.temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if return_traj:
                x0_traj.append(x0.clone())

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
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

            x_pre_unmask = x.clone().view(batch_size, num_particles, self.length)

            x[transfer_index] = x0[transfer_index]

            if return_traj:
                x_traj.append(x.clone())


            # update weights
            g_all_tok = self.get_log_weight_update(base_logits, i)
            g = g_all_tok[transfer_index]
            log_weights = log_weights + g.unsqueeze(-1) * num_transfer_tokens[:, i].unsqueeze(-1) / self.steps # num_samples, 1

            # for numerical stability
            #log_weights = log_weights - log_weights.max(dim=0, keepdim=True).values
            log_weights_r = log_weights.view(batch_size, num_particles, 1)
            log_weights_norm = log_weights_r - log_weights_r.logsumexp(dim=1, keepdim=True)
            
            x_r = x.view(batch_size, num_particles, self.length)

            x_pre_resample = x_r.clone()

            # resample (weights may be set to 0 after resampling)
            ess_batch = []
            for b in range(batch_size):

                x_r[b], log_weights_r[b], ess_b = self.resample_op(log_weights_norm[b], x_r[b], num_particles = num_particles, i = i)
                ess_batch.append(ess_b.item())

            x = x_r.view(batch_size * num_particles, self.length)
            log_weights = log_weights_r.view(batch_size * num_particles, 1)

            if log_wandb:
                log_info = utils.wandb_log_xt_smc(i, 
                                       logits, 
                                       x_pre_unmask, 
                                       x_pre_resample,
                                       x_r, 
                                       x0,
                                       self.tokenizer,
                                       log_weights_norm, 
                                       ess_batch,
                                       self.log_prob_target,
                                       self.mask_token,
                                       show_logits = False)
                wandb.log(log_info, step=i)

            if return_traj:
                ess_traj.append(ess_batch)
                log_weights_traj.append(log_weights_norm)

        # final log for wandb, to show all particles unmasked
        if log_wandb:
            log_info = utils.wandb_log_xt_smc(i+1, 
                                   logits,
                                   x_r,
                                   x_r,
                                   x_r,
                                   x0,
                                   self.tokenizer,
                                   log_weights_r, # will be 0's, since after each resampling step, the log weights are set to 0
                                   ess_batch,
                                   self.log_prob_target,
                                   self.mask_token,
                                   show_logits = False)
            wandb.log(log_info, step=i+1)

        wandb.finish() 

        if return_traj:
            return x, x0_traj, x_traj, ess_traj, log_weights_traj
        
        return x
    
    
class RewardSampler(SMCSampler):
    def __init__(self, denoiser, log_reward_func, resample = True, adaptive_resampling = False, steps=10, temperature=1.0):
        super().__init__(denoiser, resample, adaptive_resampling, steps, temperature)
        self.log_reward_func = log_reward_func
        self.anneal_schedule = lambda i: (i) / self.steps

        self.sampling_strat = "rewardSMC"

    # proposal logits is [B*M, L, V]
    def get_log_weight_update(self, base_logits, r_logits, r_i, step):
        ###########################################
        # log weights for resampling 
        
        log_mu = base_logits
        
        log_mu_r = r_logits 
        
        summand =  torch.sum(log_mu.exp() - log_mu_r.exp(), dim=-1)  # b, l

        # coeff = a_t' / (1 - a_t) 
        
        
        t = 1 - torch.tensor((step)/self.steps)
        t_less = 1 - torch.tensor((step+1)/self.steps)
        
        if t_less <= 0.:
            t_less = 1/(5*self.steps)
        integ_coeff = torch.log(t/(t_less))
    

        over_t_ratio = self.steps / (self.steps - step)  
        coeff = - over_t_ratio #/ t

        # r_i has shape [B,]

        # get a number per token 
        delta_b = self.anneal_schedule(step+1) - self.anneal_schedule(step)

        g_all_tok = integ_coeff * summand  + delta_b * r_i.unsqueeze(-1) # b, l 
        
        print("integ coeffs: ", integ_coeff)

        return g_all_tok

    # evaluate ratio exp( beta * reward(j))/exp( beta * reward(i)), with i = x, j = all Hamming 1 neighbours of x
    # logits are of shape [B, L, V]
    def eval_logr_diffs_unif(self, x, logits, step):
        N = 20
        
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

        #print("B: ", B)
        #print("L: ", L)
        #print("V: ", V)
        
        # sample N vocab tokens, without replacement
        v_list = torch.randperm(V, device=logits.device)[:N]

        for l in range(L):
            # if masked, change to v for all in batch
            masked = (x[:, l] == self.mask_token)

            for v in range(V):
                
                if v in v_list:
                    #if masked: 
                    x_neighbour = x.clone()

                    #print("x neighbour size: ", x_neighbour.shape)

                
                    x_neighbour[masked, l] = v

                    x0_n = self.mask_fill_strat(x_neighbour)  # [1, L, V]

                    r_j[:, l, v] = self.log_reward_func(x0_n) # should be equal to r_i at ~masked_in_batch positions
                    #else:
                    #    r_j[b, l, v] = r_i[b]  # if not masked, reward is same as r_i
                else:
                    r_j[:, l, v] = r_i

                r_tilted_logits[:, l, v] = logits[:, l, v] + self.anneal_schedule(step) * (r_j[:, l, v] - r_i)

                


        print("r_tilted_logits shape: ", r_tilted_logits.shape)
        print("r_i shape: ", r_i.shape)
        print("r_j shape: ", r_j.shape)
        #r_tilted_logits = logits 
        #r_i = torch.tensor(0.) .to(self.denoiser.device)
        #r_j = torch.zeros((B,L,V)).to(self.denoiser.device)

        return r_tilted_logits, r_j, r_i

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

                x0_n = self.mask_fill_strat(x_neighbour.unsqueeze(0))  # [1, L, V]

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
    def mask_fill_strat(self, x):
        
        # theoretically correct way, requires call to denoiser
        # instead of argmax, sample from denoiser
        """
        logits = self.denoiser(x)
        B, L, V = logits.shape

        x_fill = torch.multinomial(logits.exp().reshape(B*L, V), num_samples=1).squeeze(-1).reshape(B,L)
        # x_fill shape is [B,L]
        #x_fill = torch.argmax(self.denoiser(x), dim=-1) 
        """

        # other way
        x_fill = torch.where(x == self.mask_token, self.x0_base, x)
        return x_fill

    def get_base_transfer_idx(self, i, base_logits, mask_index, num_transfer_tokens, remasking='low_confidence'):
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
    def sample(self, init_seq = None, batch_size = 2, num_particles = 5, cfg_scale = 0., remasking='low_confidence', return_traj = False, log_wandb = False):
        if init_seq is not None:
            x = init_seq.clone().to(self.denoiser.device)
            self.length = x.shape[-1]
        else:
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
                                   "num_particles": num_particles})

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

            base_transfer_idx, x0_base = self.get_base_transfer_idx(i, base_logits, mask_index, num_transfer_tokens, remasking=remasking)

            # to optionally use in mask_fill strat
            self.x0_base = x0_base

            # proposal with modified inv temp beta
            r_logits, r_j, r_i = self.eval_logr_diffs(x, base_logits, step = i, base_transfer_idx=base_transfer_idx)

            logits_with_noise = add_gumbel_noise(r_logits, temperature = self.temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if return_traj:
                x0_traj.append(x0.clone())

            """
            if remasking == 'low_confidence':
                p = F.softmax(r_logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            """
           
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
            g_all_tok = self.get_log_weight_update(base_logits, r_logits, r_i, step = i)
            g = g_all_tok[transfer_index]
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
                                       logits_prop = r_logits, 
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
                                   logits_prop = r_logits,
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
    
