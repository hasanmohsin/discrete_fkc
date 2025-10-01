import torch
import torch.distributions as D
from torch.utils.data import Dataset

def get_data(density_type, dim_1d, device):
    dim_x, dim_y = dim_1d, dim_1d
    dim_data = dim_x*dim_y
    xy = torch.cartesian_prod(torch.linspace(-1,1,dim_x), torch.linspace(-1,1,dim_y)).to(device)
    xy_ind = torch.cartesian_prod(torch.arange(0,dim_x), torch.arange(0,dim_y)).to(device)
    
    if density_type == 'normal':
        normal_2d = D.MultivariateNormal(torch.zeros(2).to(device), torch.eye(2).to(device))
        probs = torch.softmax(normal_2d.log_prob(xy), dim=0)
    if density_type == 'checkers':
        probs = torch.ones(dim_data).to(device)
        for i in range(dim_data):
            if xy[i,0]*xy[i,1] < 0:
                probs[i] = 0
        probs = torch.softmax(probs, dim=0)
    if 'png' in density_type:
        from PIL import Image
        import numpy as np
        im = Image.open(density_type).resize((dim_1d, dim_1d))
        probs = torch.tensor(np.array(im, dtype=np.float32), device=device)
        probs /= probs.sum()
        probs = probs.flatten()
        
    return probs, xy_ind


class Dset(Dataset):
    def __init__(self , x):
        self.x = x.long()

    def __getitem__(self , index):
         return self.x[index]

    def __len__(self):
        return len(self.x)