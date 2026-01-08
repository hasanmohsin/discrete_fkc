import torch
from utils.model import ReverseDenoiseModel, RealReverseModel
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from utils.loss import KL, wasserstein2_from_samples
from utils.sampling import get_distr_from_samples, forward_corrupt, generate_result_annealing
from utils.data import get_data
import einops
from utils.unet import SpinUNetUpgraded
from utils.glauber import energy, magnetization, swendsen_wang_step_open, correlation_function_ensemble, swendsen_wang_step_periodic
import os
import argparse
device = "cuda"

parser = argparse.ArgumentParser()
parser.add_argument("--beta_target", type=float, default=0.35)
parser.add_argument("--beta_train", type=float, default=0.3)
parser.add_argument("--model_path", type=str)
parser.add_argument("--save_dir", type=str)
args = parser.parse_args()

beta_target = args.beta_target
beta_train = args.beta_train
dim_1d = 2
num_samples = 1000
integration_steps = 5000
Q = -dim_1d*torch.eye(dim_1d, dim_1d) + torch.ones(dim_1d, dim_1d)
Q = Q.to(device)

states = torch.load(args.model_path, weights_only=False)
model = SpinUNetUpgraded(base_channels=states['config']['base_channels'], 
                    time_emb_dim=states['config']['time_emb_dim']).to(device)
model.load_state_dict(states['state'])
model.eval()
x_generated_w, _ = generate_result_annealing(num_samples, num_samples, states['config']['schedule'], 
                                             integration_steps, dim_1d, device, Q, model, 
                                             beta=beta_target/beta_train, reweight=True, data_dim=256)
x_generated_w = einops.rearrange(2*x_generated_w-1, 'b (k l) -> b k l', l=16)
torch.save(x_generated_w, os.path.join(args.save_dir, f"{beta_target}_weights"))
x_generated, _ = generate_result_annealing(num_samples, num_samples, states['config']['schedule'], 
                                           integration_steps, dim_1d, device, Q, model, 
                                           beta=beta_target/beta_train, reweight=False, data_dim=256)
x_generated = einops.rearrange(2*x_generated-1, 'b (k l) -> b k l', l=16)
torch.save(x_generated, os.path.join(args.save_dir, f"{beta_target}_noweights"))
