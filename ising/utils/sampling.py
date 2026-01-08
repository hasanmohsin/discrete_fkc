import torch
from tqdm import tqdm
import einops
import numpy as np

def get_sigma(t, schedule):
    sigma_max = 10.0
    if schedule == 'linear':
        return sigma_max*t, sigma_max
    if schedule == 'sine':
        return sigma_max*np.sin(t*np.pi/2), np.pi*sigma_max*np.cos(t*np.pi/2)/2

def get_distr_from_samples(samples, dim_x, dim_y):
    inds, counts = torch.unique(samples, return_counts=True, dim=0)
    sampled_distr = torch.zeros(dim_x, dim_y, dtype=int, device=samples.device)
    sampled_distr[inds[:,0], inds[:,1]] = counts
    return sampled_distr


def forward_corrupt(samples_2d, Q, t, minibatch_size = 10000):
    R = torch.matrix_exp(einops.einsum(Q, t, 'n m, t -> t n m'))
    T, dim_1d, _ = R.shape
    x_corrupted = torch.zeros(T, *samples_2d.shape, dtype=int, device=samples_2d.device)
    for minibatch in range(len(samples_2d)//minibatch_size):
        start_i = minibatch*minibatch_size
        end_i = (minibatch+1)*minibatch_size
        probs_batch = R[:,einops.rearrange(samples_2d[start_i:end_i,:], 'b d -> (b d)'),:]
        sampled = torch.multinomial(einops.rearrange(probs_batch, 't n d -> (t n) d'), 1, replacement=True)
        x_corrupted[:,start_i:end_i,:] = einops.rearrange(sampled, '(t b d) 1 -> t b d', b = minibatch_size, t=T)
    return x_corrupted

def sample_cat_sys(logits):
    bs = len(logits)
    device = logits.device
    u = (torch.rand(1, device=device) + torch.arange(bs, device=device)/bs) % 1
    s = torch.softmax(logits, dim=0)
    bins = torch.cumsum(s, dim=0)
    bins[-1] = 1.0
    ids = torch.bucketize(u, bins)
    return ids

def sample_step_annealing(x_t, t, dt, dim_1d, Q, model, beta=1.0, reweight=False):
    logits = model(x_t, t)
    Q_bar = beta*Q[x_t,:]*dt*torch.exp(logits)
    B, D = x_t.shape
    b = torch.arange(B, device=x_t.device)[:, None]
    d = torch.arange(D, device=x_t.device)[None, :]
    Q_bar[b,d,x_t] = 0
    w_temp = einops.reduce(Q_bar, 'b d n -> b', 'sum')
    Q_bar = Q_bar*torch.exp((beta-1)*logits)
    temp = einops.reduce(Q_bar, 'b d n -> b', 'sum')
    w_temp = temp - w_temp
    temp = -einops.repeat(temp, 'b -> b d', d=D) + torch.ones_like(x_t, device=x_t.device)
    Q_bar[b,d,x_t] = temp/D
    sampled = torch.multinomial(einops.rearrange(Q_bar, 'b d n -> b (d n)').clip(0), 1, replacement=True)
    x_t[b, (sampled//dim_1d)] = sampled.remainder(dim_1d)
    if reweight:
        ids = sample_cat_sys(w_temp)
        return x_t[ids], ids
    else:
        return x_t, w_temp

def sample_step_product(x_t, t, dt, dim_1d, Q, model1, model2, reweight=False):
    logits1 = model1(x_t, t)
    logits2 = model2(x_t, t)
    score1 = torch.exp(logits1)
    score2 = torch.exp(logits2)
    Q_bar = Q[x_t,:]*dt
    B, D = x_t.shape
    b = torch.arange(B, device=x_t.device)[:, None]
    d = torch.arange(D, device=x_t.device)[None, :]
    Q_bar[b,d,x_t] = 0
    w_temp = einops.reduce(Q_bar*(score1+score2), 'b d n -> b', 'sum')
    Q_bar = 2*Q_bar*torch.exp(logits1+logits2)
    temp = einops.reduce(Q_bar, 'b d n -> b', 'sum')
    w_temp = temp - w_temp
    temp = -einops.repeat(temp, 'b -> b d', d=D) + torch.ones_like(x_t, device=x_t.device)
    Q_bar[b,d,x_t] = temp/D
    sampled = torch.multinomial(einops.rearrange(Q_bar, 'b d n -> b (d n)').clip(0), 1, replacement=True)
    x_t[b, (sampled//dim_1d)] = sampled.remainder(dim_1d)
    if reweight:
        ids = sample_cat_sys(w_temp)
        return x_t[ids], w_temp
    else:
        return x_t, w_temp

def generate_result_annealing(num_samples, minibatch_size, schedule, num_steps, dim_1d, device, Q, model, beta=1.0, reweight=False, data_dim=2):
    x = torch.randint(0, dim_1d, size=(num_samples, data_dim), device=device).long()
    t = 1.0
    dt = 1.0/num_steps
    unique_samples = torch.zeros(num_steps)
    with torch.no_grad():
        for i in tqdm(range(num_steps)):
            for minibatch in range(num_samples//minibatch_size):
                start_i = minibatch*minibatch_size
                end_i = (minibatch+1)*minibatch_size
                time, deriv = get_sigma(t, schedule)
                time_cur = time*torch.ones(minibatch_size,1,device=device)
                dt_cur = dt*deriv
                x[start_i:end_i,:], ww = sample_step_annealing(x[start_i:end_i,:], time_cur, dt_cur, dim_1d, Q, model, beta=beta, reweight=reweight)
            t -= dt
            unique_samples[i] = len(torch.unique(ww))
    return x, unique_samples

def generate_result_product(num_samples, minibatch_size, t_max, num_steps, dim_1d, device, Q, model1, model2, reweight=False, data_dim=2):
    x = torch.randint(0, dim_1d, size=(num_samples, data_dim), device=device).long()
    t = t_max
    dt = t_max/num_steps
    with torch.no_grad():
        for i in tqdm(range(num_steps)):
            for minibatch in range(num_samples//minibatch_size):
                start_i = minibatch*minibatch_size
                end_i = (minibatch+1)*minibatch_size
                x[start_i:end_i,:], ww = sample_step_product(x[start_i:end_i,:], t*torch.ones(minibatch_size,1,device=device), dt, dim_1d, Q, model1, model2, reweight=reweight)
            t -= dt
    return x

def exact_Ising_1d(length, Nsamples, J, beta):
    samples = torch.zeros(Nsamples, length).to(torch.bool)
    samples[:,0] = torch.randint(0,2,(Nsamples,)).to(torch.bool)
    for i in range(length-1):
        spins2flip = torch.rand(Nsamples) > (np.tanh(beta*J)+1)/2
        samples[:,i+1] = torch.logical_xor(samples[:,i], spins2flip)
    return 2*samples.to(torch.float16)-1

def get_correlation(X):
    M, N = X.shape
    C = torch.zeros(N-1)
    for r in range(1, N):
        C[r-1] = (X[:, :-r] * X[:, r:]).mean()
    return C