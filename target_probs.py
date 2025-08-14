
import torch
import torch.nn as nn 
import numpy as np
import random 


import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily

class MogGridEnergyFunction():
    def __init__(
        self,
        side_length: int,
        dimensionality: int,
        device: str
    ):
        self.mask_token_idx = side_length + 1 

        self.scale = 0.05 #0.005
        self.mean_ls = [
            [-5., -5.], [-5., 0.], [-5., 5.],
            [0., -5.], [0., 0.], [0., 5.],
            [5., -5.], [5., 0.], [5., 5.],
        ]
        self.nmode = len(self.mean_ls)
        self.mean = torch.stack([torch.tensor(xy) for xy in self.mean_ls])

        # translate all modes to within [0, 1]
        self.mean = (self.mean)  / 15.0 + 0.5

        self.side_length = side_length
        self.dimensionality = dimensionality
        self.device = device

    def prob(self, samples: torch.Tensor) -> torch.Tensor:
        log_prob = self.__call__(samples)
        prob = log_prob.exp()
        return prob

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        # Get index of one-hot
        if (samples == self.mask_token_idx).any():
            raise ValueError(
                "Evaluating the energy of a masked state which is not possible"
            )

        prop = (samples + 1) / self.side_length

        device = prop.device
        comp = D.Independent(D.Normal(self.mean.to(device), torch.ones_like(self.mean).to(device) * self.scale), 1)
        mix_coeffs = torch.ones(self.nmode).to(device)
        mix_coeffs[-1] = 3.0 * mix_coeffs[-1]
        mix = D.Categorical(mix_coeffs)
        self.gmm = MixtureSameFamily(mix, comp)

        return self.gmm.log_prob(prop) 