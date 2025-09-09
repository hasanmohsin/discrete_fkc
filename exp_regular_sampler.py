from abc import ABC, abstractmethod

import torch
import torch.nn as nn 
import numpy as np
import random 

import matplotlib.pyplot as plt 

from target_probs import MogGridEnergyFunction
from samplers import DiffusionSampler, add_gumbel_noise
from smc_sampler import AnnealSampler
from denoisers import AnalyticDenoiser

def main():
    num_particles = 1
    
    batch_size = 1000 // num_particles

    beta = 1.0 

    energy = MogGridEnergyFunction(side_length = 128, dimensionality=2, device='cuda')

    # make 2d plot of prob values
    ad = AnalyticDenoiser(log_prob = energy.__call__, vocab_size=128, length=2)

    plt.figure()
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
    
    #print("Denoiser samples for test sequences:")
    #print(samples)

    sampler = DiffusionSampler(denoiser=ad,
                            steps=2, temperature=1.0)
    
    x_samples, x0_traj, x_traj = sampler.sample(batch_size = batch_size, return_traj = True, log_wandb=True)
    
    #print("Sampled sequences:   ", x_samples)
    #print("Intermediate samples:", x0_traj)
    #print("x traj: ", x_traj)

    # plot samples
    plt.figure()
    plt.scatter(x_samples[:, 0].cpu(), x_samples[:, 1].cpu())
    plt.title("Sampled Points")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()
    plt.savefig("./plots/Default_Samples_final_num_particle_{}.png".format(num_particles))

    # intermediate samples
    plt.figure(figsize=(10, 5))
    
    for t in range(2):
        plt.scatter(x0_traj[t][:, 0].cpu(), x0_traj[t][:, 1].cpu(), label=f'Timestep {t}')
    plt.legend()
    plt.title("Intermediate Samples")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()
    plt.savefig("./plots/Default_Intermediate_Samples_num_particles_{}.png".format(num_particles))

if __name__ == "__main__":
    main()