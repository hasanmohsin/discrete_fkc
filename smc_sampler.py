from abc import ABC, abstractmethod

import torch 
import numpy as np
import random 

import torch.nn.functional as F

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

    # proposal logits is [B*M, L, V]
    def get_log_weight_update(self, base_logits, i):
        ###########################################
        # log weights for resampling 
        
        log_mu = base_logits.log_softmax(dim=-1)

        t = 1 - torch.tensor(i/self.steps)
        over_t_ratio = self.steps / (self.steps - i)  
        offset = - self.beta * over_t_ratio #/ t
    
        coeff = self.beta * (i)**(self.beta - 1) / (self.steps - i)**self.beta 
        log_denoiser_anneal = (log_mu)*self.beta 
        score_anneal_sum = (coeff * log_denoiser_anneal.exp()).sum(dim=-1)  # b, l

        # get a number per token 
        g_all_tok = score_anneal_sum + offset  # b, l 
        
        return g_all_tok

    
    # batch size is the number of particles
    @torch.no_grad()
    def sample(self, init_seq = None, batch_size = 2, num_particles = 5, cfg_scale = 0., remasking='low_confidence', return_traj = False):
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

        for i in range(self.steps):
            
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

            # resample (weights may be set to 0 after resampling)
            ess_batch = []
            for b in range(batch_size):
                print("x_r shape: ", x_r.shape)
                print("log_weights_norm shape: ", log_weights_norm.shape)
                print("num_particles: ", num_particles)

                x_r[b], log_weights_r[b], ess_b = self.resample_op(log_weights_norm[b], x_r[b], num_particles = num_particles, i = i)
                ess_batch.append(ess_b.item())

            x = x_r.view(batch_size * num_particles, self.length)
            log_weights = log_weights_r.view(batch_size * num_particles, 1)

            if return_traj:
                ess_traj.append(ess_batch)
                log_weights_traj.append(log_weights_norm)

        if return_traj:
            return x, x0_traj, x_traj, ess_traj, log_weights_traj
        
        return x
    
    
class RewardSampler(SMCSampler):
    def __init__(self, denoiser, reward, resample = True, adaptive_resampling = False, steps=10, temperature=1.0):
        super().__init__(denoiser, resample, adaptive_resampling, steps, temperature)
        self.reward = reward

    # proposal logits is [B*M, L, V]
    def get_log_weight_update(self, base_logits, i):
        ###########################################
        # log weights for resampling 
        
        log_mu = base_logits.log_softmax(dim=-1)

        t = 1 - torch.tensor(i/self.steps)
        over_t_ratio = self.steps / (self.steps - i)  
        offset = - self.beta * over_t_ratio #/ t
    
        coeff = self.beta * (i)**(self.beta - 1) / (self.steps - i)**self.beta 
        log_denoiser_anneal = (log_mu)*self.beta 
        score_anneal_sum = (coeff * log_denoiser_anneal.exp()).sum(dim=-1)  # b, l

        # get a number per token 
        g_all_tok = score_anneal_sum + offset  # b, l 
        
        return g_all_tok

    
    # batch size is the number of particles
    @torch.no_grad()
    def sample(self, init_seq = None, batch_size = 2, num_particles = 5, cfg_scale = 0., remasking='low_confidence', return_traj = False):
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

        for i in range(self.steps):
            
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

            # resample (weights may be set to 0 after resampling)
            ess_batch = []
            for b in range(batch_size):
                print("x_r shape: ", x_r.shape)
                print("log_weights_norm shape: ", log_weights_norm.shape)
                print("num_particles: ", num_particles)

                x_r[b], log_weights_r[b], ess_b = self.resample_op(log_weights_norm[b], x_r[b], num_particles = num_particles, i = i)
                ess_batch.append(ess_b.item())

            x = x_r.view(batch_size * num_particles, self.length)
            log_weights = log_weights_r.view(batch_size * num_particles, 1)

            if return_traj:
                ess_traj.append(ess_batch)
                log_weights_traj.append(log_weights_norm)

        if return_traj:
            return x, x0_traj, x_traj, ess_traj, log_weights_traj
        
        return x
    