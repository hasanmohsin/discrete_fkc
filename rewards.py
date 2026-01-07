
import torch
import torch.nn as nn 
import numpy as np
import random 


import matplotlib.pyplot as plt

import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily

from target_probs import MogGridEnergyFunction

class DiagRewardFunction():
    def __init__(
        self,
        side_length: int,
        dimensionality: int,
        device: str
    ):
        self.mask_token_idx = side_length 

        self.scale = 0.05 #0.005
        
        self.side_length = side_length
        self.dimensionality = dimensionality
        self.device = device
        self.name = "DiagRewardFunction"

        if dimensionality == 2 and side_length == 4:
            
            self.log_r_table = torch.tensor(
                [[-0.0, -5.0, -5.0, -5.0],
                 [-5.0,  -0.0, -5.0, -5.0],
                 [-5.0, -5.0,  0.0, -5.0],
                 [-5.0, -5.0, -5.0, -0.0]], device=self.device
            )

    def reward(self, samples: torch.Tensor) -> torch.Tensor:
        return self.__call__(samples).exp()

    def get_log_reward_table(self):
        reward_arr = torch.zeros((self.side_length, self.side_length), device=self.device)
        for indices in np.ndindex(reward_arr.shape):
            r_val = self(torch.tensor(list(indices)).unsqueeze(0).to(self.device))
            reward_arr[indices] = r_val 
        return reward_arr


    # returns log reward
    # assumes input of shape (batch_size, 2)
    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        # Get index of one-hot
        if (samples == self.mask_token_idx).any():
            raise ValueError(
                "Evaluating the reward of a masked state which is not possible"
            )
        if samples.ndim != 2:
            raise ValueError(
                "Expected input of shape (batch_size, 2)"
            )

      
        prop = (samples + 1) / self.side_length

        rewards = torch.full((prop.shape[0],), -1.0, device=self.device)  # Initialize rewards tensor

        cond = (prop[..., 0] + prop[..., 1] <= 1)

        # make reward high for everything in the bottom left quadrant
        rewards[cond] = torch.tensor(0.0).to(self.device)


        return rewards  
class Reward_2D_Function():
    def __init__(
        self,
        side_length: int,
        dimensionality: int,
        device: str
    ):
        self.mask_token_idx = side_length 
        
        self.side_length = side_length
        self.dimensionality = dimensionality
        self.device = device

        self.name = "Reward_2D_Function"

    def get_log_reward_table(self):
        reward_arr = torch.zeros((self.side_length, self.side_length), device=self.device)
        for indices in np.ndindex(reward_arr.shape):
            r_val = self(torch.tensor(list(indices)).unsqueeze(0).to(self.device))
            reward_arr[indices] = r_val 
        return reward_arr

    def reward(self, samples: torch.Tensor) -> torch.Tensor:
        return self.__call__(samples).exp()

    # returns log reward
    # assumes input of shape (batch_size, 2)
    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        # Get index of one-hot
        if (samples == self.mask_token_idx).any():
            raise ValueError(
                "Evaluating the reward of a masked state which is not possible"
            )
        if samples.ndim != 2:
            raise ValueError(
                "Expected input of shape (batch_size, 2)"
            )

        prop = (samples + 1) / self.side_length

        rewards = torch.full((prop.shape[0],), -10.0, device=self.device)  # Initialize rewards tensor

        cond = (prop[..., 0] < 0.5) & (prop[..., 1] < 0.5)

        # for even sums high reward, otherwise low reward
        #cond = (prop[..., 0] + prop[..., 1]) % 2 == 0

        # make reward high for everything in the bottom left quadrant
        rewards[cond] = torch.tensor(0.0).to(self.device)
        rewards[~cond] = torch.tensor(-5.0).to(self.device)

        return rewards  

def main():
    logprob = MogGridEnergyFunction(side_length = 128, dimensionality=2, device='cuda')
    reward = DiagRewardFunction(side_length=128, dimensionality=2, device='cuda')

    reward_arr = torch.zeros((128, 128), device='cuda')
    t_reward_arr = torch.zeros((128, 128), device='cuda')

    for indices in np.ndindex(reward_arr.shape):
        log_prob_value = logprob(torch.tensor(list(indices)).to(logprob.device))
        r_val = reward(torch.tensor(list(indices)).unsqueeze(0).to(reward.device))
        
        reward_arr[indices] = r_val 
        t_reward_arr[indices] = r_val + log_prob_value 


      
    # sample from reward using it as distribution
    torch.distributions.Categorical(t_reward_arr.view(-1))

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

if __name__ == "__main__":
    main()