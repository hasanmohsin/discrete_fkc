from abc import ABC, abstractmethod

import torch
import torch.nn as nn 
import numpy as np
import random 

import matplotlib.pyplot as plt 

from target_probs import MogGridEnergyFunction
from rewards import DiagRewardFunction
from samplers import DiffusionSampler, add_gumbel_noise
from smc_sampler import AnnealSampler, RewardSampler 
from denoisers import AnalyticDenoiser

def main():
    num_particles = 10
    
    batch_size = 1000 // num_particles

    energy = MogGridEnergyFunction(side_length = 128, dimensionality=2, device='cuda')
    reward = DiagRewardFunction(side_length = 128, dimensionality=2, device='cuda')

    logprob = energy

    # make 2d plot of prob values
    ad = AnalyticDenoiser(log_prob = energy.__call__, vocab_size=128, length=2)

    reward_arr = torch.zeros((128, 128), device='cuda')
    t_reward_arr = torch.zeros((128, 128), device='cuda')

    for indices in np.ndindex(reward_arr.shape):
        log_prob_value = logprob(torch.tensor(list(indices)).to(logprob.device))
        r_val = reward(torch.tensor(list(indices)).unsqueeze(0).to(reward.device))
        
        reward_arr[indices] = r_val 
        t_reward_arr[indices] = r_val + log_prob_value 

        #print(f"Indices: {indices}, Log Probability: {log_prob_value.item()}, Reward: {r_val.item()}")

    # sample from true tilted distribution as categorical
    #tilted_dist = torch.distributions.Categorical(t_reward_arr.view(-1))
    #tilted_samples = tilted_dist.sample((1000,)) # 1000 samples, 

    ## convert back to 2d (from index 0, 128**2 to (x, y))
    #tilted_samples_2d = 

    #plt.figure()
    #plt.imshow(t_reward_arr.exp().detach().cpu().numpy())
    #plt.colorbar()
    #plt.title("True Samples")
    #plt.scatter(tilted_samples)

    plt.figure()
    plt.imshow(reward_arr.exp().detach().cpu().numpy())
    plt.colorbar()
    plt.title("Reward")
    plt.xlabel("Token")
    plt.ylabel("Token")
    plt.show()
    plt.savefig("./plots/reward_table.png")

    plt.figure()
    plt.imshow(t_reward_arr.exp().detach().cpu().numpy())
    plt.colorbar()
    plt.title("Tilted Reward")
    plt.xlabel("Token")
    plt.ylabel("Token")
    plt.show()
    plt.savefig("./plots/tilted_reward_table.png")

    plt.imshow(ad.log_prob_table.exp().detach().cpu().numpy())
    plt.colorbar()
    plt.title("Analytic Denoiser Probability Table")
    plt.xlabel("Token")
    plt.ylabel("Token")
    plt.show()
    plt.savefig("./plots/analytic_denoiser_prob_table.png")

    test_seq = torch.tensor([[128, 128], [128, 0], [64, 128], [64, 64]])
    logits = ad(test_seq)
    print("Denoiser logits for test sequences:")
    print(logits)

    # sample using gumbel 
    samples = add_gumbel_noise(logits, temperature=1.0).argmax(dim=-1)
    print("Denoiser samples for test sequences:")
    print(samples)

    sampler = RewardSampler(denoiser=ad, log_reward_func=reward, resample = True,
                            adaptive_resampling = False, 
                            steps=2, temperature=1.0)
    
    x_samples, x0_traj, x_traj, ess_traj, log_weights_traj = sampler.sample(batch_size = batch_size, num_particles = num_particles, return_traj = True)
    
    print("Sampled sequences:   ", x_samples)
    #print("Intermediate samples:", x0_traj)
    #print("x traj: ", x_traj)

    # plot samples
    plt.figure()
    plt.imshow(t_reward_arr.exp().detach().cpu().numpy())
    plt.scatter(x_samples[:, 0].cpu(), x_samples[:, 1].cpu())
    plt.title("Sampled Points")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()
    plt.savefig("./plots/Reward_Diag_Samples_20_unif_v_final_num_particle_{}.png".format(num_particles))

    # intermediate samples
    plt.figure(figsize=(10, 5))
    
    for t in range(2):
        plt.scatter(x0_traj[t][:, 0].cpu(), x0_traj[t][:, 1].cpu(), label=f'Timestep {t}')
    plt.legend()
    plt.title("Intermediate Samples")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()
    plt.savefig("./plots/Reward_Diag_Intermediate_Samples_20_unif_v_num_particles_{}.png".format(num_particles))

if __name__ == "__main__":
    main()