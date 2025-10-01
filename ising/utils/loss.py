import torch
import einops
import numpy as np

def DSE_loss(x, t, model, Q):
    bs, dimensions = x.shape
    R = torch.matrix_exp(einops.einsum(Q, einops.repeat(t, 't 1 -> (t d)', d=dimensions), 'n m, t -> t n m'))
    probs_batch = R[torch.arange(bs*dimensions),einops.rearrange(x, 'b d -> (b d)'),:]
    x_c = einops.rearrange(torch.multinomial(probs_batch, 1, replacement=True), '(b d) 1 -> b d', b=bs)
    logits = model(x_c, t)
    logits = einops.rearrange(logits, 'b d n -> (b d) n')
    x_c = einops.rearrange(x_c, 'b d -> (b d)')
    before_sum = torch.exp(logits) - einops.einsum(logits, probs_batch, 1/probs_batch[torch.arange(bs*dimensions),x_c], 'n d, n d, n -> n d')
    before_sum[torch.arange(len(before_sum)), x_c] = 0
    return before_sum.mean()

def warmup_lambda(step, warmup_steps=3200):
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0

def KL(p, q, num_samples, mode='forward'):
    K = len(p)
    a = 0.5/num_samples
    p = (p + a)/(1 + a*K)
    q = (q + a)/(1 + a*K)
    if mode == 'forward':
        return torch.sum(p*torch.log(p/q))
    if mode == 'reverse':
        return torch.sum(q*torch.log(q/p))

def wasserstein2_from_samples(data1, data2, bins=100):
    """
    Compute 2-Wasserstein distance between two 1D datasets
    using histograms with identical bins.
    
    Parameters
    ----------
    data1, data2 : array-like
        Input samples.
    bins : int or array-like
        Number of bins or explicit bin edges.
    
    Returns
    -------
    float
        W2 distance between the two histograms.
    """
    # build common bins covering both datasets
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    lo = min(data1.min(), data2.min())
    hi = max(data1.max(), data2.max())
    edges = np.linspace(lo, hi, bins + 1) if np.isscalar(bins) else np.asarray(bins)

    # histograms
    h1, _ = np.histogram(data1, bins=edges, density=False)
    h2, _ = np.histogram(data2, bins=edges, density=False)

    # normalize to probability mass
    p = h1.astype(float) / h1.sum()
    q = h2.astype(float) / h2.sum()

    # bin centers
    centers = 0.5 * (edges[:-1] + edges[1:])

    # cumulative transport algorithm
    i = j = 0
    cost = 0.0
    mass_p, mass_q = p.copy(), q.copy()
    while i < len(mass_p) and j < len(mass_q):
        m = min(mass_p[i], mass_q[j])
        dx = centers[i] - centers[j]
        cost += m * (dx * dx)
        mass_p[i] -= m
        mass_q[j] -= m
        if mass_p[i] <= 1e-15:
            i += 1
        if mass_q[j] <= 1e-15:
            j += 1

    return np.sqrt(cost)