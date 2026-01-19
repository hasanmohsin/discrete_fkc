import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
from utils.loss import DSE_loss, warmup_lambda
from utils.data import get_data, Dset
from utils.model import ReverseDenoiseModel
from utils.sampling import get_distr_from_samples, forward_corrupt
from utils.unet import SpinUNetUpgraded
import einops
import random

device = 'cuda'

def get_sigma(t, schedule):
    sigma_max = 10.0
    if schedule == 'linear':
        return sigma_max*t
    if schedule == 'sine':
        return sigma_max*torch.sin(t*np.pi/2)
        
parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str)
parser.add_argument("--data_path", type=str)
parser.add_argument("--from_checkpoint", type=str, default=None)
args = parser.parse_args()
os.makedirs(os.path.join("models", args.results_dir), exist_ok=True)

dim_1d = 2
Q = -dim_1d*torch.eye(dim_1d, dim_1d) + torch.ones(dim_1d, dim_1d)
Q = Q.to(device)


params = {"batch_size": [100, 200, 400, 800, 1600],
          "base_channels": [16, 32, 64, 128],
          "time_emb_dim": [32, 64, 128, 256],
          "weight_decay": [0.0, 5e-5, 1e-4, 5e-4, 1e-3],
          "lr": [5e-5, 1e-4, 2e-4, 3e-4, 4e-4],
          "schedule": ["linear", "sine"]}

random_config = {k: random.choice(v) for k, v in params.items()}

batch_size=random_config["batch_size"]
data = torch.load(args.data_path)
data = (data.long()+1)//2
train_set = TensorDataset(data)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

model = SpinUNetUpgraded(base_channels=random_config["base_channels"], time_emb_dim=random_config["time_emb_dim"]).to(device)
if args.from_checkpoint is not None:
    state_dict = torch.load(args.from_checkpoint)
    model.load_state_dict(state_dict)

decay, no_decay = [], []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if (
        name.endswith("bias") or
        "norm" in name.lower() or
        "ln" in name.lower() or
        "bn" in name.lower() or
        "logvar" in name.lower() or
        "log_sigma" in name.lower() or
        "logsnr" in name.lower()
    ):
        no_decay.append(param)
    else:
        decay.append(param)

optimizer = torch.optim.AdamW(
    [
        {"params": decay, "weight_decay": random_config["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=random_config["lr"],
)

warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
num_epochs = 10000

losses = []
for epoch in tqdm(range(num_epochs)):
    for x in train_loader:
        model.train()
        optimizer.zero_grad()
        times = get_sigma(torch.rand((batch_size, 1), device=device), random_config["schedule"])
        x = x[0].to(device)
        x = einops.rearrange(x, 'b k l -> b (k l)')
        loss = DSE_loss(x, times, model, Q)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        warmup_scheduler.step()
        losses.append(loss.item())
    if (epoch+1)%200 == 0:
        torch.save({"state": model.state_dict(), 
                    "config": random_config, 
                    "losses": torch.tensor(losses)}, 
                   os.path.join("models", args.results_dir, f"{epoch}.pth"))