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

class ProductSampler(SMCSampler):
    def __init__(self, denoiser1, denoiser2, resample = True, adaptive_resampling = False, steps=10, temperature=1.0, mask_token = 126336):
        super().__init__(denoiser1, resample=resample, adaptive_resampling=adaptive_resampling, steps=steps, temperature=temperature)

        self.denoiser2 = denoiser2

        self.sampling_strat = "product"

        # to avoid integrating to 0 in integrated_coeff
        # is a hyperparameter that can be tuned - determines strength of final resampling 
        self.eps_constant = 3 

    def integrated_coeff(self, i):
        # integrate from t_i to t_{i+1} 
        # eg. for i = 0, integrate from 0 to 1/steps
        t_higher = 1 - torch.tensor((i)/self.steps) # since we integrate backwards, this is the startpoint 
        t_lower = 1 - torch.tensor((i+1)/self.steps) # endpoint

        # clamp become becoming 0
        t_lower = torch.clamp(t_lower, min = 1/(self.eps_constant*self.steps))

        def integrand(t):
            return 2 * (1/t) * (1 - t)/t
 
        n_points = 100
        t_points = torch.linspace(t_lower, t_higher, n_points)

        # integrate using trapezoidal rule
        coeff = torch.trapz(integrand(t_points), t_points)

        #coeff = coeff.clamp(max=10000.)

        return coeff


    def get_log_weight_update(self, base_logits_1, base_logits_2, i, integrate = False):
        '''
        For product of experts, the weight update is given by the difference of the logits of the two models.
        '''
        log_mu_1 = base_logits_1.log_softmax(dim=-1)
        log_mu_2 = base_logits_2.log_softmax(dim=-1)
        
        p_prod = log_mu_1.exp() * log_mu_2.exp()

        t = 1 - torch.tensor(i/self.steps)
        over_t_ratio = self.steps / (self.steps - i)  
        offset = - over_t_ratio #-1/(t) = a_t' / (1 - a_t), 
        
        alpha_t = self.get_alpha_t(i)
        neg_over_t_ratio = self.get_elbo_weight(i) #-1/t
        offset = neg_over_t_ratio #- 1 / t

        coeff = -2 * neg_over_t_ratio * alpha_t / (1 - alpha_t)

        if integrate:
            print("\n\nWould be coeff step {}: ".format(i), coeff)
            coeff = self.integrated_coeff(i)
            print("Large coeff at final step to avoid numerical issues for small number of steps: ", coeff)

        g_all_tok = 2 * neg_over_t_ratio  + coeff * p_prod.sum(dim = -1)

        return g_all_tok

    @torch.no_grad()
    def sample(self, init_seq = None, batch_size = 2, num_particles = 5, cfg_scale = 0., remasking='low_confidence', return_traj = False, log_wandb = True):
        if log_wandb:
            utils.setup_wandb_run(project = "discrete_fkc", 
                                  config = {"sampler": "DualProdSMC", 
                                    "denoiser_name": self.denoiser.name,
                                    "denoiser_name_2": self.denoiser2.name,
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

            base_logits_1 = self.denoiser(x)
            base_logits_2 = self.denoiser2(x)

            # proposal for product of experts is sum of logits
            logits = base_logits_1 + base_logits_2

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
            integrate = self.steps < 10 
            g_all_tok = self.get_log_weight_update(base_logits_1, base_logits_2, i, integrate=integrate)
            g = g_all_tok[transfer_index]
            
            if not integrate:
                dt = num_transfer_tokens[:, i].unsqueeze(-1) / self.steps
                log_weights = log_weights + g.unsqueeze(-1) * dt # num_samples, 1
            else:
                log_weights = log_weights + g.unsqueeze(-1) # dt already included in integrated coeff

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
                log_info = utils.wandb_log_xt_smc(
                                       step = i, 
                                       logits_prop = logits, 
                                       x_pre_unmask = x_pre_unmask,
                                       x_pre_resample = x_pre_resample,
                                       x_r = x_r,
                                       x0 = x0,
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
                                logits_prop = logits,
                                x_pre_unmask = x_pre_unmask,
                                x_pre_resample = x_pre_resample,
                                x_r = x_r,
                                x0 = x0,
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
    
    
class GeoProductSampler(SMCSampler):
    def __init__(self, denoiser1, denoiser2, resample = True, adaptive_resampling = False, steps=10, temperature=1.0, mask_token = 126336):
        super().__init__(denoiser1, resample=resample, adaptive_resampling=adaptive_resampling, steps=steps, temperature=temperature)

        self.denoiser2 = denoiser2

        self.sampling_strat = "geo_product"

    def integrated_coeff(self, i):
        # integrate from t_i to t_{i+1} 
        # eg. for i = 0, integrate from 0 to 1/steps
        t_higher = 1 - torch.tensor((i)/self.steps) # since we integrate backwards, this is the startpoint 
        t_lower = 1 - torch.tensor((i+1)/self.steps) # endpoint

        # clamp become becoming 0
        t_lower = torch.clamp(t_lower, min = 1/(10*self.steps))

        def integrand(t):
            return -1/t
 
        n_points = 100
        t_points = torch.linspace(t_lower, t_higher, n_points)

        # integrate using trapezoidal rule
        coeff = torch.trapz(integrand(t_points), t_points)

        #coeff = coeff.clamp(max=10000.)

        return coeff


    def get_log_weight_update(self, base_logits_1, base_logits_2, i, integrate = False):
        '''
        For product of experts, the weight update is given by the difference of the logits of the two models.
        '''
        log_mu_1 = base_logits_1.log_softmax(dim=-1)
        log_mu_2 = base_logits_2.log_softmax(dim=-1)
        

        log_geom = self.prod_weights[0] * log_mu_1 + self.prod_weights[1] * log_mu_2

        p_geom = log_geom.exp()

        t = 1 - torch.tensor(i/self.steps)
        over_t_ratio = self.steps / (self.steps - i)  
        offset = - over_t_ratio #-1/(t) = a_t' / (1 - a_t), 
        
        alpha_t = self.get_alpha_t(i)
        neg_over_t_ratio = self.get_elbo_weight(i) #-1/t
        offset = neg_over_t_ratio #- 1 / t

        coeff = neg_over_t_ratio 

        if integrate:
            print("\n\nWould be coeff step {}: ".format(i), coeff)
            coeff = self.integrated_coeff(i)
            print("Large coeff at final step to avoid numerical issues for small number of steps: ", coeff)

        g_all_tok = neg_over_t_ratio  - coeff * p_geom.sum(dim = -1)

        return g_all_tok

    @torch.no_grad()
    def sample(self, init_seq = None, batch_size = 2, num_particles = 5, cfg_scale = 0., remasking='low_confidence', return_traj = False, 
               prod_weights = torch.Tensor([0.5,0.5]), log_wandb = True):
        if log_wandb:
            utils.setup_wandb_run(project = "discrete_fkc", 
                                  config = {"sampler": "DualGeoProdSMC", 
                                    "denoiser_name": self.denoiser.name,
                                    "denoiser_name_2": self.denoiser2.name,
                                   "start_time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                   "steps": self.steps, 
                                   "temperature": self.temperature, 
                                   "cfg_scale": cfg_scale, 
                                   "remasking": remasking, 
                                   "sampling_strat": self.sampling_strat,
                                   "batch_size": batch_size,
                                   "num_particles": num_particles,
                                   "prod_weights": prod_weights.tolist()})

        self.prod_weights = prod_weights.to(self.denoiser.device)

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

            base_logits_1 = self.denoiser(x)
            base_logits_2 = self.denoiser2(x)

            
            # proposal for product of experts is sum of logits
            logits = self.prod_weights[0] * base_logits_1 + self.prod_weights[1] * base_logits_2

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
            integrate = self.steps < 10 
            g_all_tok = self.get_log_weight_update(base_logits_1, base_logits_2, i, integrate=integrate)
            g = g_all_tok[transfer_index]
            
            if not integrate:
                dt = num_transfer_tokens[:, i].unsqueeze(-1) / self.steps
                log_weights = log_weights + g.unsqueeze(-1) * dt # num_samples, 1
            else:
                log_weights = log_weights + g.unsqueeze(-1) # dt already included in integrated coeff

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
                log_info = utils.wandb_log_xt_smc(
                                       step = i, 
                                       logits_prop = logits, 
                                       x_pre_unmask = x_pre_unmask,
                                       x_pre_resample = x_pre_resample,
                                       x_r = x_r,
                                       x0 = x0,
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
                                logits_prop = logits,
                                x_pre_unmask = x_pre_unmask,
                                x_pre_resample = x_pre_resample,
                                x_r = x_r,
                                x0 = x0,
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
class ProductPromptSampler(SMCSampler):
    def __init__(self, denoiser, resample = True, adaptive_resampling = False, steps=10, temperature=1.0, mask_token = 126336):
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

    @torch.no_grad()
    def sample(self, prompt_list, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='random', num_samples = 5, tokenizer=None, adaptive_resampling=False, log_wandb = False):
        '''
        Args:
            model: Mask predictor.
            prompt_list: Batch of prompts, of shape (B, L). Will take output over product of prompts in batch 
            steps: Sampling steps, less than or equal to gen_length.
            gen_length: Generated answer length.
            block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
            temperature: Categorical distribution sampling temperature.
            cfg_scale: Unsupervised classifier-free guidance scale.
            remasking: Remasking strategy. 'low_confidence' or 'random'.
            mask_id: The toke id of [MASK] is 126336.
        '''

        B = prompt_list.shape[0]
        N_prompts = torch.tensor(prompt_list.shape[0]).to(self.device)

        x = torch.full((B, prompt_list.shape[1] + gen_length), self.mask_token, dtype=torch.long).to(self.device)
        x[:, :prompt_list.shape[1]] = prompt_list.clone()

        prompt_index = (x != self.mask_token)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks
        
        # batch of samples
        smc_samples = torch.full((num_samples, gen_length), self.mask_token, dtype=torch.long).to(self.device)
        
        # for resampling 
        log_weights = torch.zeros((num_samples, 1), dtype=torch.float32).to(self.device)

        assert num_blocks == 1 # for now, single block 
        

        if log_wandb:
            utils.setup_wandb_run(project = "discrete_fkc", 
                                  config = {"sampler": "ProductPromptSMC", 
                                   "denoiser_name": self.denoiser.name,
                                   "start_time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                   "steps": steps, 
                                   "temperature": temperature, 
                                   "cfg_scale": cfg_scale, 
                                   "remasking": remasking, 
                                   "sampling_strat": self.sampling_strat,
                                   "batch_size": B,
                                   "num_particles": num_samples,
                                   "gen_length": gen_length,
                                   "block_length": block_length})

        mask_index_samples = []
        logits_samples = []
        ess_list = []  # to track effective sample size

        for num_block in range(num_blocks):
            block_mask_index = (x[:, prompt_list.shape[1] + num_block * block_length: prompt_list.shape[1] + (num_block + 1) * block_length:] == self.mask_token)
            num_transfer_tokens = self.get_num_transfer_tokens(block_mask_index, steps)

            for i in tqdm(range(steps)):
                t = 1 - torch.tensor(i/steps)

                x0 = []
                
                x_pre_unmask = smc_samples.clone()

                if i == 0:  
                    mask_index_ans = (smc_samples == self.mask_token)

                    # logits will be of length prompt_list.shape[1] + gen_length
                    # batched along number of different prompts 
                    logits = self.denoiser(x)
                    prompt_logits = logits[:, prompt_list.shape[1]:] # b, l, vocab_size, only for the generated part

                    prod_logits = logits.sum(dim=0, keepdim=True)  # not geometric avg # 1, l, vocab_size
                    prod_logits = prod_logits[:, prompt_list.shape[1]:] # 1, l, vocab_size, only for the generated part

                    for j in range(num_samples):
                        prod_logits_j = add_gumbel_noise(prod_logits, temperature=temperature)
                        x0s_j = torch.argmax(prod_logits_j, dim=-1)
                        

                        x0.append(x0s_j[0]) # b, l

                    
                    x0s = torch.stack(x0, dim=0) # num_samples, l
                    logits_samples = prod_logits.repeat(num_samples, 1, 1) # num_samples, l, vocab_size 
                    mask_index_samples = mask_index_ans #[0,...].unsqueeze(0).repeat(num_samples, 1) # num_samples, l

                    # for reweighting
                    # in this case, all logits are equal, so different smc samples will have equal weights
                    log_gamma = prompt_logits.log_softmax(dim=-1).sum(dim=0, keepdim=True) 
                    
                    """ OLD way of computing log weights
                    sum_rate = (log_gamma.exp() * mask_index_ans.unsqueeze(-1)).sum()
                    gamma_sum =  (N_prompts/t) * sum_rate * ((1-t)/t)**(N_prompts - 1) - N_prompts/t  
                    gamma_sum = gamma_sum.unsqueeze(0).repeat(num_samples, 1) # num_samples, 1 
                    """

                    # new way
                    sum_rate = (log_gamma.exp() * mask_index_ans.unsqueeze(-1)).sum(dim=-1) # num_samples, l
                    #print("prompt logits shape: ", prompt_logits.shape)
                    #print("log_gamma shape: ", log_gamma.shape)
                    #print("sum rate shape: ", sum_rate.shape)

                    gamma_sum =  (N_prompts/t) * sum_rate * ((1-t)/t)**(N_prompts - 1) - N_prompts/t  
                    #gamma_sum = gamma_sum.unsqueeze(0).repeat(num_samples, 1) # num_samples, l  

                    #transfer_index = self.get_transfer_indx(remasking, 
                    #                                        prod_logits, 
                    #                                        x0s, 
                    #                                        smc_samples, 
                    #                                        mask_index_ans, 
                    #                                        num_transfer_tokens, 
                    #                                        i)

                    #print("log_weights shape: ", log_weights.shape)
                    #print("gamma_sum shape: ", gamma_sum.shape)
                    log_weights += gamma_sum.mean(dim=-1, keepdim=True) * num_transfer_tokens[0, i]/steps # num_samples, 1


                else:

                    x_ = x.clone()
                    for j in range(num_samples):
                        smc_samples_j = smc_samples[j, :].unsqueeze(0).repeat(B, 1)
                        
                        # copy over smc_sample to each prompt in batch
                        x_[:, prompt_list.shape[1]:] = smc_samples_j
                        
                        #print("mask index: ", mask_index_samples.shape)
                        mask_index_samples[j, :] = (smc_samples[j, :] == self.mask_token)
                        logits_j = self.denoiser(x_)

                        logits_j = logits_j[:, prompt_list.shape[1]:] # b, l, vocab_size, only for the generated part
                        #print("logits_j shape: ", logits_j.shape)
                        
                        prod_logits_j = logits_j.sum(dim=0, keepdim=True)  # not geometric average # 1, l, vocab_size
                    
                        prod_logits_j_noise = add_gumbel_noise(prod_logits_j, temperature=temperature)
                        x0s_j = torch.argmax(prod_logits_j_noise, dim=-1)
                        x0s[j, :] = x0s_j[0]
                        logits_samples[j, :, :] = prod_logits_j[0, :]

                        """
                        # OLD compute log weights
                        log_gamma_j = logits_j.log_softmax(dim=-1).sum(dim=0, keepdim=True) # 1, l, vocab_size
                        sum_rate_j = (log_gamma_j.exp() * mask_index_samples[j, :].unsqueeze(-1)).sum()
                        gamma_sum_j =  (N_prompts/t) * sum_rate_j * ((1-t)/t)**(N_prompts - 1) - N_prompts/t
                        log_weights[j, :] = log_weights[j, :] + gamma_sum_j * num_transfer_tokens[0, i]/steps # num_samples, 1
                        """

                        # new 
                        log_gamma_j = logits_j.log_softmax(dim=-1).sum(dim=0, keepdim=True) # 1, l, vocab_size
                        sum_rate_j = (log_gamma_j.exp() * mask_index_samples[j, :].unsqueeze(-1)).sum(dim=-1) # 1, l
                        gamma_sum_j =  (N_prompts/t) *sum_rate_j * ((1-t)/t)**(N_prompts - 1) - N_prompts/t

                        transfer_index = self.get_transfer_indx(remasking, 
                                                                logits_samples[j, ...].unsqueeze(0), 
                                                                x0s[j, ...], 
                                                                smc_samples[j, :], 
                                                                mask_index_samples[j, :], 
                                                                num_transfer_tokens, 
                                                                i)

                        #print("transfer index shape: ", transfer_index.shape)
                        #print("transfer index: ", transfer_index)
                        #print("gamma_sum_j shape: ", gamma_sum_j.shape)
                        #print("gamma_sum[transfer_index] shape: ", gamma_sum_j[:, transfer_index].shape)

                        log_weights[j, :] = log_weights[j, :] + gamma_sum_j[:, transfer_index] * num_transfer_tokens[0, i]/steps # num_samples, 1
                        

                        log_weights = log_weights - log_weights.max(dim=0,keepdim=True).values # for numerical stability
                     
                       
                # now should have mask_index_samples (starting from after prompt section), x0, logits_samples <- each of shape [num_samples, ...], for each SMC sample      
                
                

                # now go through each SMC sample, and remask/sample 
                for j in range(num_samples):
                    logit_j = logits_samples[j, ...].unsqueeze(0) # b, l, vocab_size
                    x0_j = x0s[j, ...]  # b, l
                    x_ind_j = smc_samples[j, :].clone() 
                    
                    mask_index_j = mask_index_samples[j, ...] # b, l


                    if remasking == 'low_confidence':
                        p = F.softmax(logit_j.to(torch.float64), dim=-1)
                        x0_p = torch.squeeze(
                            torch.gather(p, dim=-1, index=torch.unsqueeze(x0_j, -1)), -1) # b, l
                    elif remasking == 'random':
                        x0_p = torch.rand((x0_j.shape[0]), device=x0_j.device)
                    else:
                        raise NotImplementedError(remasking)
        
                    x0_j = torch.where(mask_index_j, x0_j, x_ind_j)
                    confidence = torch.where(mask_index_j, x0_p, -np.inf)


                    transfer_index = torch.zeros_like(x0_j, dtype=torch.bool, device=x0_j.device)
                    
                    _, select_index = torch.topk(confidence, k=num_transfer_tokens[0, i])
                    

                    transfer_index[select_index] = True
                    
                    x_ind_j[transfer_index] = x0_j[transfer_index]

                    smc_samples[j, :] = x_ind_j

                    

                # resampling
                # track log weights, and resample if necessary
                log_weights_norm = log_weights - log_weights.logsumexp(dim=0, keepdim=True)  # normalize log weights
                weights = log_weights_norm.exp().squeeze(1)
                #weights = weights / weights.sum()  # normalize weights   
                
                ess = 1. / (weights ** 2).sum()
                ess_list.append(ess.item())

                x_pre_resample = smc_samples.clone()

                if weights.shape[0] > 1:
                    if adaptive_resampling:
                        if ess < num_samples / 2 or i == steps - 1:
                            # resample if ESS is low or at the last step
                            print("\n\nResampling at step {} with ESS: {}".format(i, ess.item()))
                            resample_inds = systematic_resample(weights)
                            smc_samples = smc_samples[resample_inds, :]
                            log_weights = torch.zeros((num_samples, 1), dtype=torch.float32).to(self.device)
                        else:
                            # do nothing, weights also not set to 0 
                            smc_samples = smc_samples
                            
                    else:
                        # resample every step   
                        resample_inds = systematic_resample(weights)
                        smc_samples = smc_samples[resample_inds, :]
                        log_weights = torch.zeros((num_samples, 1), dtype=torch.float32).to(self.device)
                else:
                    smc_samples = smc_samples

                    log_weights = torch.zeros((num_samples, 1), dtype=torch.float32).to(self.device)

                if log_wandb and (steps <= 128 or i % 16 == 0):
                    log_info = utils.wandb_log_xt_smc(
                                           step = i, 
                                           logits_prop = logits_samples, 
                                           x_pre_unmask = x_pre_unmask.unsqueeze(0), 
                                           x_pre_resample = x_pre_resample.unsqueeze(0), 
                                           x_r = smc_samples.unsqueeze(0), 
                                           x0 = x0s, # 1*num_samples, l 
                                           tokenizer = self.tokenizer,
                                           log_weights_r = log_weights_norm, 
                                           ess_batch = [ess.item()],
                                           log_prob_target=None,
                                           mask_token=self.mask_token,
                                           show_logits=False)

                    wandb.log(log_info, step=i)

        if log_wandb:
            log_info = utils.wandb_log_xt_smc(
                                   step = steps, 
                                   logits_prop = logits_samples, 
                                   x_pre_unmask = x_pre_unmask.unsqueeze(0), 
                                   x_pre_resample = x_pre_resample.unsqueeze(0), 
                                   x_r = smc_samples.unsqueeze(0), 
                                   x0 = x0s, 
                                   tokenizer = self.tokenizer,
                                   log_weights_r = log_weights_norm, 
                                   ess_batch = [ess.item()],
                                   log_prob_target=None,
                                   mask_token=self.mask_token,
                                   show_logits=False)
            wandb.log(log_info, step=steps)

        wandb.finish()

        return x, smc_samples, ess_list 

