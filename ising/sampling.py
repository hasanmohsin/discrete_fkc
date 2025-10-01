import torch
from utils.model import ReverseDenoiseModel, RealReverseModel
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from utils.loss import KL
from utils.sampling import get_distr_from_samples, forward_corrupt, generate_result_annealing
from utils.data import get_data
import einops
from utils.unet import SpinUNet
from utils.unet_new import SpinUNetUpgraded
from utils.glauber import energy, magnetization
import os
import argparse
device = "cuda"

parser = argparse.ArgumentParser()
parser.add_argument("--beta", type=float, default=0.2)
parser.add_argument("--model_name", type=str)
args = parser.parse_args()

beta = args.beta
dim_1d = 2
num_samples = 10000
integration_steps = 5000
t_max = 4.0
Q = -dim_1d*torch.eye(dim_1d, dim_1d) + torch.ones(dim_1d, dim_1d)
Q = Q.to(device)

model = SpinUNetUpgraded(base_channels=64, time_emb_dim=256).to(device)
state_dict = torch.load(f'models/ising{args.model_name}/699.pth')
model.load_state_dict(state_dict)
model.eval()
xw = torch.zeros(10*num_samples, 256)
x = torch.zeros(10*num_samples, 256)
betas = torch.linspace(beta, 0.45, 10)
for i in range(8, 10):
    x_generated_w, u = generate_result_annealing(num_samples, num_samples, t_max, integration_steps, dim_1d, device, Q, model, beta=betas[i]/beta, reweight=True, data_dim=256)
    x_generated, u = generate_result_annealing(num_samples, num_samples, t_max, integration_steps, dim_1d, device, Q, model, beta=betas[i]/beta, reweight=False, data_dim=256)
    xw[i*num_samples:(i+1)*num_samples] = x_generated_w
    x[i*num_samples:(i+1)*num_samples] = x_generated
    torch.save(einops.rearrange(2*xw-1, 'b (k l) -> b k l', l=16), f'samples/ising{args.model_name}_10betas_weights1')
    torch.save(einops.rearrange(2*x-1, 'b (k l) -> b k l', l=16), f'samples/ising{args.model_name}_10betas1')
    