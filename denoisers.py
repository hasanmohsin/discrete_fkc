from abc import ABC, abstractmethod

import torch
import torch.nn as nn 
import numpy as np
import random 

import matplotlib.pyplot as plt 

from target_probs import MogGridEnergyFunction
from samplers import DiffusionSampler, add_gumbel_noise
from smc_sampler import AnnealSampler

class AnalyticDenoiser():
    """
    log_prob: function from sequences of tokens (length L) to their probability
    (converted into a prob_table - torch tensor of shape (V, V ,... ,V) (L times) where target_prob[x0, x1, ..., xL] = prob at sequence (x0, x1, .., xL))
    vocab_size: each token is an index in [0, vocab_size]
    length: The length of the tokens allowed (fixed)
    """
    def __init__(self, log_prob, vocab_size = 128, length = 2, device = 'cuda'):
        self.log_prob_func = log_prob
        self.vocab_size = vocab_size
        self.length = length

        self.device = device 

        self.mask_token = vocab_size  # Mask token is the last index

        self.log_prob_table = torch.zeros(
            (vocab_size,) * length
        ).to(device)

        self.construct_log_prob_table()


    # Construct the probability table from the probability function
    def construct_log_prob_table(self):
        for indices in np.ndindex(self.log_prob_table.shape):
            self.log_prob_table[indices] = self.log_prob_func(torch.tensor(list(indices)).to(self.device))

    # input sequence is of shape (L)
    # can contain masked tokens (index = vocab_size)
    def denoiser(self, input_seq):
        L = input_seq.shape[0]

        denoise_logits = torch.zeros((L, self.vocab_size)).to(self.device)
        masked = (input_seq == self.mask_token)
        
        num_masked = masked.sum().item()

        inds = []
        masked_pos = []

        for i in range(input_seq.shape[0]):
            if masked[i]:
                inds.append(slice(None))
                masked_pos.append(i)
            else:
                inds.append(input_seq[i])
                # make all elements 0 except input_seq[i] (which is 0)
                denoise_logits[i, :] = torch.full((self.vocab_size,), float('-inf')).to(self.device)
                denoise_logits[i, input_seq[i]] = torch.tensor(0.0).to(self.device)

        masked_p = self.log_prob_table[inds]  # This is a tensor of shape (V, V, ..., V) (num masked times)

        # the remaining logits will be determined by marginalizing over the remaining mask positions
        for m in range(num_masked):
            dims_to_sum = list(range(num_masked))
            dims_to_sum.pop(m)

            if len(dims_to_sum) > 0:
                denoise_logits[masked_pos[m], :] = masked_p.logsumexp(dim=dims_to_sum).log_softmax(dim=-1)

        return denoise_logits # should be of shape [L, V]

 

    # batched call that works on input seq of shape (B,L)
    def __call__(self, input_seq):
        outs = []

        input_seq = input_seq.to(self.device)
        
        for i in range(input_seq.shape[0]):
            out = self.denoiser(input_seq[i])
            outs.append(out)

        return torch.stack(outs)

def main():
    energy = MogGridEnergyFunction(side_length = 128, dimensionality=2, device='cuda')

    # make 2d plot of prob values
    ad = AnalyticDenoiser(log_prob = energy.__call__, vocab_size=128, length=2)

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

    sampler = DiffusionSampler(denoiser=ad, steps=2, temperature=1.0)
    x_samples, x0_traj, x_traj = sampler.sample(batch_size = 100, return_traj = True)
    
    print("Sampled sequences:   ", x_samples)
    print("Intermediate samples:", x0_traj)
    print("x traj: ", x_traj)

    # plot samples
    plt.scatter(x_samples[:, 0].cpu(), x_samples[:, 1].cpu())
    plt.title("Sampled Points")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()
    plt.savefig("./plots/Samples_final.png")

    # intermediate samples
    plt.figure(figsize=(10, 5))
    
    for t in range(2):
        plt.scatter(x0_traj[t][:, 0].cpu(), x0_traj[t][:, 1].cpu(), label=f'Timestep {t}')
    plt.legend()
    plt.title("Intermediate Samples")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()
    plt.savefig("./plots/Intermediate_Samples.png")

if __name__ == "__main__":
    main()