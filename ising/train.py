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
from utils.unet import SpinUNet
from utils.unet_new import SpinUNetUpgraded
import einops

device = 'cuda'
# torch.manual_seed(42)
# np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str)
parser.add_argument("--data", type=str)
parser.add_argument("--from_checkpoint", type=str, default=None)
args = parser.parse_args()
os.makedirs(args.results_dir, exist_ok=True)

dim_1d = 2
Q = -dim_1d*torch.eye(dim_1d, dim_1d) + torch.ones(dim_1d, dim_1d)
Q = Q.to(device)
t_max = 4.0

batch_size = 400
data = torch.load(args.data)
data = (data.long()+1)//2
train_set = TensorDataset(data)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

model = SpinUNetUpgraded(base_channels=64, time_emb_dim=256).to(device)
if args.from_checkpoint is not None:
    state_dict = torch.load(args.from_checkpoint)
    model.load_state_dict(state_dict)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
num_epochs = 10000

losses = []
for epoch in tqdm(range(num_epochs)):
    for x in train_loader:
        model.train()
        optimizer.zero_grad()
        times = t_max*torch.rand((batch_size, 1), device=device)
        x = x[0].to(device)
        x = einops.rearrange(x, 'b k l -> b (k l)')
        loss = DSE_loss(x, times, model, Q)
        loss.backward()
        optimizer.step()
        warmup_scheduler.step()
        losses.append(loss.item())
    if (epoch+1)%50 == 0:
        torch.save(model.state_dict(), os.path.join(args.results_dir, f"{epoch}.pth"))
        np.save(os.path.join(args.results_dir, "losses"), np.array(losses))