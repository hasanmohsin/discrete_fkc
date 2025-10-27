import torch
import numpy as np

import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm
from datetime import datetime

import wandb

import utils 
from samplers import DiffusionSampler, add_gumbel_noise
from resample import systematic_resample
from smc_sampler import SMCSampler

# product sampler, which batches over number of prompts
class ProductPromptSampler(SMCSampler):
    def __init__(self, denoiser, resample = True, adaptive_resampling = False, 
                 steps=10, temperature=1.0, mask_token = 126336):
        super().__init__(denoiser, resample=resample, adaptive_resampling=adaptive_resampling, steps=steps, temperature=temperature)

        self.mask_token = mask_token
        self.sampling_strat = "prompt_product"

    def get_transfer_indx(self, remasking, logit, x0, x_ind, mask_index, num_transfer_tokens, i):
        if remasking == 'low_confidence':
            p = F.softmax(logit.to(torch.float64), dim=-1)
            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
        elif remasking == 'random':
            x0_p = torch.rand((x0.shape[0]), device=x0.device)
        else:
            raise NotImplementedError(remasking)

        x0 = torch.where(mask_index, x0, x_ind)
        confidence = torch.where(mask_index, x0_p, -np.inf)


        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        
        _, select_index = torch.topk(confidence, k=num_transfer_tokens[0, i])
        

        transfer_index[select_index] = True

        return transfer_index # same shape as x0

    # prod_logits is of shape: num_particles, length, vocab_size
    # mask_index is of shape: num_particles, length
    def get_log_weight_update(self, prod_logits, i, mask_index, N_prompts):

        sums = (prod_logits.exp() * mask_index.unsqueeze(-1)).sum(dim=-1)  # num_particles, l

        # at i = 0, alpha_t = 0, i = steps - 1, alpha_t = 1/self.steps
        alpha_t = self.get_alpha_t(i)
        neg_over_t_ratio = self.get_elbo_weight(i) #-1/t = alpha'_t / (1 - alpha_t)

        coeff = (alpha_t / (1 - alpha_t))**(N_prompts-1)
        g_all_tok = N_prompts * neg_over_t_ratio * (1 - coeff * sums)

        return g_all_tok # num_particles, length

    # prompt_seqs is of shape: num_prompts, length + prompt_length
    # x is of shape: num_particles, length
    # output are logits for prompt + x of shape: num_particles, : 
    def get_prompt_logits(self, x, prompt_seqs, prompt_len):

        # return logits for each smc sample 
        # shape : num_particles, length, vocab_size
        x0s = []
        prod_logits = []

        for j in range(x.shape[0]):
            # copy over particles to the prompt seqs (prompt part is fixed, smc sample is appended)
            prompt_seqs[:, prompt_len:] = x[j, :].unsqueeze(0).repeat(prompt_seqs.shape[0], 1)

            # [num_prompts, length + prompt_length, vocab_size]
            logits_j = self.denoiser(prompt_seqs)

            # normalize logits, this is important for correct weight updates
            logits_j = F.log_softmax(logits_j, dim=-1)

            prod_logits_j = logits_j.sum(dim = 0) # length, vocab_size

            # sample from logits to get x0 for this smc sample
            prod_logits_j_with_noise = add_gumbel_noise(prod_logits_j, temperature = self.temperature)
            x0s_j = torch.argmax(prod_logits_j_with_noise, dim=-1) # length

            prod_logits.append(prod_logits_j)
            x0s.append(x0s_j)
    
        x0s = torch.stack(x0s, dim=0) # num_particles, length
        x0s = x0s[:, prompt_len:] # only return the part corresponding to smc sample

        prod_logits = torch.stack(prod_logits, dim=0) # num_particles, length, vocab_size
        prod_logits = prod_logits[:, prompt_len:, :] # only return the part corresponding to smc sample

        return x0s, prod_logits
    
    # assuming 1 batch at a time
    # produces samples of shape: num_particles, prompt_length
    # prompt_seq: num_prompts, length
    @torch.no_grad()
    def sample(self, init_seq=None, prompt_list = None, 
               gen_length = None,
               batch_size = 1, 
               num_particles = 5, cfg_scale = 0., remasking='low_confidence', return_traj = False, 
               log_wandb = False,
               cut_off_resample = None):
    
        self.cut_off_resample = cut_off_resample
        if log_wandb:
            utils.setup_wandb_run(project = "discrete_fkc", 
                                  config = {"sampler": "NewPromptProdSMC", 
                                    "denoiser_name": self.denoiser.name,
                                   "start_time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                   "steps": self.steps, 
                                   "temperature": self.temperature, 
                                   "cfg_scale": cfg_scale, 
                                   "remasking": remasking, 
                                   "sampling_strat": self.sampling_strat,
                                   "batch_size": batch_size,
                                   "num_particles": num_particles,
                                   "prompt list": prompt_list.tolist(), 
                                   "Num prompts": prompt_list.shape[0],
                                   "cut_off_resample": cut_off_resample}) 
        
        assert(batch_size == 1) # for now only support batch size 1, since different prompt lengths
        B = prompt_list.shape[0]

        N_prompts = torch.tensor(prompt_list.shape[0]).to(self.device)

        if init_seq is not None:
            x = init_seq.clone().to(self.denoiser.device)

        else:
            # the generation length is self.length
            # fill 
            if gen_length is None:
                self.length = gen_length 
            else:
                gen_length = self.steps
                self.length = gen_length

            prompt_seqs = torch.full((B, prompt_list.shape[1] + self.length), self.mask_token, dtype=torch.long).to(self.device)
            prompt_seqs[:, :prompt_list.shape[1]] = prompt_list.clone()

            # the set of smc particles, without the prompts
            x = torch.full((num_particles, self.length), self.mask_token, dtype=torch.long).to(self.denoiser.device)

        # for resampling 
        log_weights = torch.zeros((num_particles, 1), dtype=torch.float32).to(self.denoiser.device)

        prompt_index = (prompt_seqs != self.mask_token)

        # mask index in the set of smc particles (without prompts)
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

            # prod_logits [num_particles, length, vocab_size]
            x0s, prod_logits = self.get_prompt_logits(x, prompt_seqs, prompt_list.shape[1])

            if return_traj:
                x0_traj.append(x0s.clone())


            # remask and get transfer index
            # works in parallel over all smc particles
            if remasking == 'low_confidence':
                p = F.softmax(prod_logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0s, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0s.shape[0], x0s.shape[1]), device=x0s.device)
            else:
                raise NotImplementedError(remasking)

           
            x0s = torch.where(mask_index, x0s, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0s, dtype=torch.bool, device=x0s.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True

            x_pre_unmask = x.unsqueeze(0).clone()

            x[transfer_index] = x0s[transfer_index]

            if return_traj:
                x_traj.append(x.clone())

            # resampling step
            # update weights
            g_all_tok = self.get_log_weight_update(prod_logits, i, mask_index, N_prompts)
            g = g_all_tok[transfer_index] # num_samples
            log_weights = log_weights + g.unsqueeze(-1) * num_transfer_tokens[:, i].unsqueeze(-1) / self.steps # num_samples, 1

            # for numerical stability
            log_weights_r = log_weights.clone().unsqueeze(0)

            log_weights_norm = log_weights_r - log_weights_r.logsumexp(dim=0, keepdim=True)
            
            x_r = x.view(1, num_particles, self.length)

            x_pre_resample = x_r.clone()

            # resample (weights may be set to 0 after resampling)
            ess_batch = []
            for b in range(batch_size):

                x_r[b], log_weights_r[b], ess_b = self.resample_op(log_weights_norm[b], x_r[b], num_particles = num_particles, i = i)
                ess_batch.append(ess_b.item())

            x = x_r.view(batch_size * num_particles, self.length)
            log_weights = log_weights_r.view(batch_size * num_particles, 1)

            if log_wandb:
                log_info = utils.wandb_log_xt_smc(
                                       step = i, 
                                       logits_prop = prod_logits.reshape(1, num_particles, self.length, -1), 
                                       x_pre_unmask = x_pre_unmask,
                                       x_pre_resample = x_pre_resample,
                                       x_r = x_r,
                                       x0 = x0s,
                                       tokenizer = self.tokenizer,
                                       log_weights_r = log_weights_norm,
                                       ess_batch = ess_batch,
                                       log_prob_target = self.log_prob_target,
                                       mask_token = self.mask_token,
                                       show_logits = False)
                wandb.log(log_info, step=i)


            if return_traj:
                ess_traj.append(ess_batch)
                log_weights_traj.append(log_weights_norm)

        # final log for wandb, to show all particles unmasked
        if log_wandb:
            log_info = utils.wandb_log_xt_smc(
                                step = i+1, 
                                logits_prop = prod_logits,
                                x_pre_unmask = x.reshape(1, num_particles, self.length),
                                x_pre_resample =x_r,
                                x_r = x_r,
                                x0 = x0s,
                                tokenizer = self.tokenizer,
                                log_weights_r = log_weights_r, # will be 0's, since after each resampling step, the log weights are set to 0
                                ess_batch = ess_batch,
                                log_prob_target = self.log_prob_target,
                                mask_token = self.mask_token,
                                show_logits = False)
            wandb.log(log_info, step=i+1)

        wandb.finish()

        if return_traj:
            return x, x0_traj, x_traj, ess_traj, log_weights_traj
        
        return x


# product sampler, which batches over number of prompts
class GeoAvgPromptSampler(SMCSampler):
    def __init__(self, denoiser, resample = True, adaptive_resampling = False, steps=10, temperature=1.0, mask_token = 126336):
        super().__init__(denoiser, resample=resample, adaptive_resampling=adaptive_resampling, steps=steps, temperature=temperature)

        self.mask_token = mask_token
        self.sampling_strat = "prompt_geo_average"

    def get_transfer_indx(self, remasking, logit, x0, x_ind, mask_index, num_transfer_tokens, i):
        if remasking == 'low_confidence':
            p = F.softmax(logit.to(torch.float64), dim=-1)
            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
        elif remasking == 'random':
            x0_p = torch.rand((x0.shape[0]), device=x0.device)
        else:
            raise NotImplementedError(remasking)

        x0 = torch.where(mask_index, x0, x_ind)
        confidence = torch.where(mask_index, x0_p, -np.inf)


        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        
        _, select_index = torch.topk(confidence, k=num_transfer_tokens[0, i])
        

        transfer_index[select_index] = True

        return transfer_index # same shape as x0

    # prod_logits is of shape: num_particles, length, vocab_size
    # mask_index is of shape: num_particles, length
    def get_log_weight_update(self, prod_logits, i, mask_index, N_prompts):

        sums = (prod_logits.exp() * mask_index.unsqueeze(-1)).sum(dim=-1)  # num_particles, l

        # at i = 0, alpha_t = 0, i = steps - 1, alpha_t = 1/self.steps
        alpha_t = self.get_alpha_t(i)
        neg_over_t_ratio = self.get_elbo_weight(i) #-1/t = alpha'_t / (1 - alpha_t)

        
        g_all_tok = N_prompts * neg_over_t_ratio * (1 - sums)

        return g_all_tok # num_particles, length

    # prompt_seqs is of shape: num_prompts, length + prompt_length
    # x is of shape: num_particles, length
    # output are logits for prompt + x of shape: num_particles, : 
    def get_geo_avg_prompt_logits(self, x, prompt_seqs, prompt_len, prompt_weights):

        # return logits for each smc sample 
        # shape : num_particles, length, vocab_size
        x0s = []
        prod_logits = []

        for j in range(x.shape[0]):
            # copy over particles to the prompt seqs (prompt part is fixed, smc sample is appended)
            prompt_seqs[:, prompt_len:] = x[j, :].unsqueeze(0).repeat(prompt_seqs.shape[0], 1)

            # [num_prompts, length + prompt_length, vocab_size]
            logits_j = self.denoiser(prompt_seqs)

            # normalize logits, this is important for correct weight updates
            logits_j = F.log_softmax(logits_j, dim=-1)

            coeff_logits = prompt_weights.view(-1, 1, 1)

            prod_logits_j = (coeff_logits * logits_j).sum(dim = 0)  # length, vocab_size

            # sample from logits to get x0 for this smc sample
            prod_logits_j_with_noise = add_gumbel_noise(prod_logits_j, temperature = self.temperature)
            x0s_j = torch.argmax(prod_logits_j_with_noise, dim=-1) # length

            prod_logits.append(prod_logits_j)
            x0s.append(x0s_j)
    
        x0s = torch.stack(x0s, dim=0) # num_particles, length
        x0s = x0s[:, prompt_len:] # only return the part corresponding to smc sample

        prod_logits = torch.stack(prod_logits, dim=0) # num_particles, length, vocab_size
        prod_logits = prod_logits[:, prompt_len:, :] # only return the part corresponding to smc sample

        return x0s, prod_logits
    
    # assuming 1 batch at a time
    # produces samples of shape: num_particles, prompt_length
    # prompt_seq: num_prompts, length
    # prompt_weights is of shape, torch tensor: num_prompts
    @torch.no_grad()
    def sample(self, init_seq=None, prompt_list = None, 
               prompt_weights = None,
               gen_length = None,
               batch_size = 1, 
               num_particles = 5, cfg_scale = 0., remasking='low_confidence', return_traj = False, 
               log_wandb = False,
               cut_off_resample = None):
    
        self.cut_off_resample = cut_off_resample

        # weight assigned in geometric average to each prompt        
        if prompt_weights is None:
            prompt_weights = torch.ones((prompt_list.shape[0],), device=self.device) / prompt_list.shape[0]
        else:
            prompt_weights = prompt_weights / prompt_weights.sum()

        if log_wandb:
            utils.setup_wandb_run(project = "discrete_fkc", 
                                  config = {"sampler": "PromptGeoSMC", 
                                    "denoiser_name": self.denoiser.name,
                                   "start_time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                   "steps": self.steps, 
                                   "temperature": self.temperature, 
                                   "cfg_scale": cfg_scale, 
                                   "remasking": remasking, 
                                   "sampling_strat": self.sampling_strat,
                                   "batch_size": batch_size,
                                   "num_particles": num_particles,
                                   "prompt list": prompt_list.tolist(), 
                                   "Num prompts": prompt_list.shape[0],
                                   "prompt weights": prompt_weights.tolist(),
                                   "cut_off_resample": cut_off_resample}) 
        
        assert(batch_size == 1) # for now only support batch size 1, since different prompt lengths
        B = prompt_list.shape[0]

        N_prompts = torch.tensor(prompt_list.shape[0]).to(self.device)

        if init_seq is not None:
            x = init_seq.clone().to(self.denoiser.device)

        else:
            # the generation length is self.length
            # fill 
            if gen_length is None:
                self.length = gen_length 
            else:
                gen_length = self.steps
                self.length = gen_length

            prompt_seqs = torch.full((B, prompt_list.shape[1] + self.length), self.mask_token, dtype=torch.long).to(self.device)
            prompt_seqs[:, :prompt_list.shape[1]] = prompt_list.clone()

            # the set of smc particles, without the prompts
            x = torch.full((num_particles, self.length), self.mask_token, dtype=torch.long).to(self.denoiser.device)

        # for resampling 
        log_weights = torch.zeros((num_particles, 1), dtype=torch.float32).to(self.denoiser.device)

        prompt_index = (prompt_seqs != self.mask_token)

        # mask index in the set of smc particles (without prompts)
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

            # prod_logits [num_particles, length, vocab_size]
            x0s, prod_logits = self.get_geo_avg_prompt_logits(x, prompt_seqs, prompt_list.shape[1], prompt_weights)

            if return_traj:
                x0_traj.append(x0s.clone())


            # remask and get transfer index
            # works in parallel over all smc particles
            if remasking == 'low_confidence':
                p = F.softmax(prod_logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0s, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0s.shape[0], x0s.shape[1]), device=x0s.device)
            else:
                raise NotImplementedError(remasking)

           
            x0s = torch.where(mask_index, x0s, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0s, dtype=torch.bool, device=x0s.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True

            x_pre_unmask = x.unsqueeze(0).clone()

            x[transfer_index] = x0s[transfer_index]

            if return_traj:
                x_traj.append(x.clone())

            # resampling step
            # update weights
            g_all_tok = self.get_log_weight_update(prod_logits, i, mask_index, N_prompts)
            g = g_all_tok[transfer_index] # num_samples
            log_weights = log_weights + g.unsqueeze(-1) * num_transfer_tokens[:, i].unsqueeze(-1) / self.steps # num_samples, 1

            # for numerical stability
            log_weights_r = log_weights.clone().unsqueeze(0)

            log_weights_norm = log_weights_r - log_weights_r.logsumexp(dim=0, keepdim=True)
            
            x_r = x.view(1, num_particles, self.length)

            x_pre_resample = x_r.clone()

            # resample (weights may be set to 0 after resampling)
            ess_batch = []
            for b in range(batch_size):

                x_r[b], log_weights_r[b], ess_b = self.resample_op(log_weights_norm[b], x_r[b], num_particles = num_particles, i = i)
                ess_batch.append(ess_b.item())

            x = x_r.view(batch_size * num_particles, self.length)
            log_weights = log_weights_r.view(batch_size * num_particles, 1)

            if log_wandb:
                log_info = utils.wandb_log_xt_smc(
                                       step = i, 
                                       logits_prop = prod_logits.reshape(1, num_particles, self.length, -1), 
                                       x_pre_unmask = x_pre_unmask,
                                       x_pre_resample = x_pre_resample,
                                       x_r = x_r,
                                       x0 = x0s,
                                       tokenizer = self.tokenizer,
                                       log_weights_r = log_weights_norm,
                                       ess_batch = ess_batch,
                                       log_prob_target = self.log_prob_target,
                                       mask_token = self.mask_token,
                                       show_logits = False)
                wandb.log(log_info, step=i)


            if return_traj:
                ess_traj.append(ess_batch)
                log_weights_traj.append(log_weights_norm)

        # final log for wandb, to show all particles unmasked
        if log_wandb:
            log_info = utils.wandb_log_xt_smc(
                                step = i+1, 
                                logits_prop = prod_logits,
                                x_pre_unmask = x.reshape(1, num_particles, self.length),
                                x_pre_resample =x_r,
                                x_r = x_r,
                                x0 = x0s,
                                tokenizer = self.tokenizer,
                                log_weights_r = log_weights_r, # will be 0's, since after each resampling step, the log weights are set to 0
                                ess_batch = ess_batch,
                                log_prob_target = self.log_prob_target,
                                mask_token = self.mask_token,
                                show_logits = False)
            wandb.log(log_info, step=i+1)

        wandb.finish()

        if return_traj:
            return x, x0_traj, x_traj, ess_traj, log_weights_traj
        
        return x
    