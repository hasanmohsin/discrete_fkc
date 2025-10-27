
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
        self.name = f"MoG_{side_length}_{dimensionality}D"
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

class GaussianGridEnergyFunction():
    def __init__(
        self,
        side_length: int,
        dimensionality: int,
        device: str
    ):
        self.name = f"Gaussian_{side_length}_{dimensionality}D"
        self.mask_token_idx = side_length + 1 

        means_list = [5.] * dimensionality
        means_list[0] = 0.

        means_list2 = [0.] * dimensionality
        means_list2[0] = 5.

        self.scale = 0.5 #0.005
        self.mean_ls = [
            means_list,
            means_list2
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
    
class TrickyGaussianGridEnergyFunction():
    def __init__(
        self,
        side_length: int,
        dimensionality: int,
        device: str
    ):
        self.name = f"Gaussian_{side_length}_{dimensionality}D"
        self.mask_token_idx = side_length + 1 

        means_list = [5.] * dimensionality
        means_list[0] = 0.

        means_list2 = [-5.]*dimensionality
        means_list2[0] = 0.

        means_list2 = [0.] * dimensionality
        means_list2[0] = 5.

        self.scale = 0.5 #0.005
        self.mean_ls = [
            means_list,
            means_list2
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
        mix_coeffs[-1] = 5.0 * mix_coeffs[-1]
        mix = D.Categorical(mix_coeffs)
        self.gmm = MixtureSameFamily(mix, comp)

        return self.gmm.log_prob(prop) 

class GaussianGridEnergyFunction2():
    def __init__(
        self,
        side_length: int,
        dimensionality: int,
        device: str
    ):
        self.name = f"Gaussian_{side_length}_{dimensionality}D"
        self.mask_token_idx = side_length + 1 

        means_list = [0.] * dimensionality
        means_list[0] = 0.

        means_list2 = [2.] * dimensionality
        means_list2[0] = 2.

        self.scale = 0.5 #0.005
        self.mean_ls = [
            means_list,
            means_list2
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

class Categorical_2D_NI():
    def __init__(
        self,
        device: str
    ):
        self.name = f"Categorical_4_2D_NI"
        side_length = 4
        dimensionality = 2
        self.mask_token_idx = side_length + 1

       
        self.side_length = side_length
        self.dimensionality = dimensionality
        self.device = device

        s = 1.5
        self.prob_table = torch.tensor([
                    [s, 2, s, 2],
                    [s, s, s, s],
                    [2, s, 2, s],
                    [s, s, s, 2],
                ]).to(device)
        
        #self.prob_table[self.prob_table==1] = 1.9

        # increasing values left to right, top to bottom
        incr = torch.arange(0, 16).view(4,4).to(device)
        incr = incr.float() / incr.sum()

        self.prob_table = self.prob_table + 1000*incr

        
        self.prob_table = self.prob_table / self.prob_table.sum()

        self.log_prob_table = self.prob_table.log()

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

        if len(samples.shape) == 1:
            samples = samples.unsqueeze(0)
        if samples.shape[1] != 2:
            raise ValueError("Samples should have shape (batch_size, 2) for 2D categorical distribution")


        # answer is lookup in prob table
        log_prob_vals=  self.log_prob_table[samples[:,0], samples[:,1]]

        return log_prob_vals
    

class Categorical_2D_NI_v2():
    def __init__(
        self,
        device: str
    ):
        self.name = f"Other_Categorical_4_2D_NI"
        side_length = 4
        dimensionality = 2
        self.mask_token_idx = side_length + 1

       
        self.side_length = side_length
        self.dimensionality = dimensionality
        self.device = device

        s = 10.0
        t = 0.5 
        self.prob_table = torch.tensor([
                    [s, t, s/3, t],
                    [s-1, s+t, s, t],
                    [t, s, t, s],
                    [t/2, s*2, t, t],
                ]).to(device)
        
        self.prob_table = self.prob_table / self.prob_table.sum()
        #self.prob_table[self.prob_table==1] = 1.9

        # increasing values left to right, top to bottom
        incr = 5 * torch.flip(torch.arange(0, 16), dims=[0]).view(4,4).to(device)
        incr = incr.float() / incr.sum()

        self.prob_table = self.prob_table + 0.5 * incr 
        self.prob_table = self.prob_table / self.prob_table.sum()

        self.log_prob_table = self.prob_table.log()

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

        if len(samples.shape) == 1:
            samples = samples.unsqueeze(0)
        if samples.shape[1] != 2:
            raise ValueError("Samples should have shape (batch_size, 2) for 2D categorical distribution")


        # answer is lookup in prob table
        log_prob_vals=  self.log_prob_table[samples[:,0], samples[:,1]]

        return log_prob_vals
    
class Categorical_4D_NI():
    def __init__(
        self,
        device: str
    ):
        self.name = f"Categorical_4_4D_NI"
        side_length = 2
        dimensionality = 4
        self.mask_token_idx = side_length + 1

       
        self.side_length = side_length
        self.dimensionality = dimensionality
        self.device = device

        self.prob_table = torch.tensor([
                            [1, 2, 1, 2],
                            [1, 1, 1, 1],
                            [2, 1, 2, 1],
                            [1, 1, 1, 2],
                        ]).to(device)
        
        # reshape so it has 4 dimensions, each of size 2
        self.prob_table = self.prob_table.view(2, 2, 2, 2)

        self.prob_table = self.prob_table / self.prob_table.sum()

        self.log_prob_table = self.prob_table.log()

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

        if len(samples.shape) == 1:
            samples = samples.unsqueeze(0)
        if samples.shape[1] != self.dimensionality:
            raise ValueError(f"Samples should have shape (batch_size, {self.dimensionality}) for {self.dimensionality}D categorical distribution")


        # answer is lookup in prob table
        log_prob_vals=  self.log_prob_table[samples[:,0], samples[:,1], samples[:, 2], samples[:, 3]]

        return log_prob_vals