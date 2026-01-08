import torch
import torch.nn.functional as F
from typing import Tuple
from tqdm import tqdm
import numpy as np

# ------------------------------ Utilities ------------------------------

def init_spins(batch: int, L: int, device: torch.device = None, ordered: bool = False) -> torch.Tensor:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if ordered:
        spins = torch.ones((batch, L, L), dtype=torch.int8, device=device)
    else:
        # torch.randint returns ints in {0,1} -> map to {-1,+1}
        r = torch.randint(0, 2, (batch, L, L), device=device)
        spins = (r * 2 - 1).to(torch.int8)
    return spins


def _conv_kernel(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    k = torch.tensor([[0, 1, 0],
                      [1, 0, 1],
                      [0, 1, 0]], dtype=dtype, device=device)
    # shape (1,1,3,3) for conv2d
    return k.unsqueeze(0).unsqueeze(0)


# ---------------------------- Core operations ---------------------------

def neighbor_sum(spins: torch.Tensor) -> torch.Tensor:
    # spins -> (B,1,L,L) float
    B, L, _ = spins.shape
    device = spins.device
    x = spins.to(torch.float32).unsqueeze(1)  # (B,1,L,L)
    k = _conv_kernel(device, dtype=x.dtype)
    # use circular padding to implement periodic BCs
    x_padded = F.pad(x, (1, 1, 1, 1), mode='circular')
    neigh = F.conv2d(x_padded, k, padding=0)
    return neigh.squeeze(1)  # (B,L,L)


def glauber_step_random_site(spins: torch.Tensor, beta: float, rng: torch.Generator = None) -> torch.Tensor:
    B, L, _ = spins.shape
    device = spins.device

    # compute neighbour sums for all sites (we'll index into it)
    neigh = neighbor_sum(spins)  # (B,L,L), float32

    # pick random integer coordinates for each replica
    if rng is None:
        # default generator (non-deterministic)
        rows = torch.randint(0, L, (B,), device=device)
        cols = torch.randint(0, L, (B,), device=device)
    else:
        rows = torch.randint(0, L, (B,), generator=rng, device=device)
        cols = torch.randint(0, L, (B,), generator=rng, device=device)

    # gather current spin and local field at selected sites
    batch_idx = torch.arange(B, device=device)
    sigma = spins[batch_idx, rows, cols].to(torch.float32)  # (B,)
    h = neigh[batch_idx, rows, cols]  # (B,)

    # Glauber flip probability (no external field, J=1):
    # P(flip) = 1 / (1 + exp(2 * beta * sigma * h))
    # numerically stable via sigmoid: P = sigmoid(-2 * beta * sigma * h)
    arg = -2.0 * beta * sigma * h
    p_flip = torch.sigmoid(arg)

    # sample uniform randoms and decide flips
    u = torch.rand(B, device=device)
    flips = (u < p_flip)

    if flips.any():
        # flip those spins in-place
        flips_idx = flips.nonzero(as_tuple=False).squeeze(1)
        rsel = rows[flips_idx]
        csel = cols[flips_idx]
        # multiply selected spins by -1
        spins[flips_idx, rsel, csel] = -spins[flips_idx, rsel, csel]

    return spins


def parallel_sweep(spins: torch.Tensor, beta: float, rng: torch.Generator = None) -> torch.Tensor:
    B, L, _ = spins.shape
    device = spins.device

    # generate a random permutation of site indices for the sweep
    # we'll flatten coords: idx in [0, L*L)
    idxs = torch.randperm(L * L, device=device)

    for idx in idxs:
        # decode index
        i = (idx // L).item()
        j = (idx % L).item()
        # compute neighbour sums (full) and attempt flips at (i,j) for all replicas
        neigh = neighbor_sum(spins)  # (B,L,L)
        sigma = spins[:, i, j].to(torch.float32)
        h = neigh[:, i, j]
        p_flip = torch.sigmoid(-2.0 * beta * sigma * h)
        u = torch.rand(B, device=device)
        flips = (u < p_flip)
        if flips.any():
            sel = flips.nonzero(as_tuple=False).squeeze(1)
            spins[sel, i, j] = -spins[sel, i, j]
    return spins


# --------------------------- Higher-level API --------------------------

def sweep(spins: torch.Tensor, beta: float, mode: str = 'random_site', rng: torch.Generator = None) -> torch.Tensor:
    B, L, _ = spins.shape
    if mode == 'random_site':
        # do L*L single-site updates (random locations) per replica
        for _ in range(L * L):
            spins = glauber_step_random_site(spins, beta, rng=rng)
        return spins
    elif mode == 'parallel_sweep':
        return parallel_sweep(spins, beta, rng=rng)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_glauber(L: int, beta: float, n_sweeps: int, mode: str = 'random_site', rng: torch.Generator = None, verbose: bool = False) -> torch.Tensor:
    r = torch.randint(0, 2, (1, L, L), device="cuda")
    spins = (r * 2 - 1).to(torch.int8)
    all_chain = torch.zeros(n_sweeps+1, 1, L, L)
    all_chain[0] = spins
    for s in tqdm(range(n_sweeps)):
        all_chain[s+1] = sweep(all_chain[s], beta, mode=mode, rng=rng)
    return all_chain.squeeze(1)


# --------------------------- Observables --------------------------------

def magnetization(spins: torch.Tensor) -> torch.Tensor:
    """Return magnetization per replica (mean spin)."""
    return spins.to(torch.float32).mean(dim=(1, 2))  # (B,)


def energy(spins: torch.Tensor) -> torch.Tensor:
    """Compute energy per replica for J=1, zero external field.

    E = - sum_{<ij>} s_i s_j  (each pair counted once)
    We compute using neighbor_sum and divide by 2 to avoid double counting.
    """
    neigh = neighbor_sum(spins)  # sum of neighbours for each site
    # local energy per site = -0.5 * s_i * sum_neigh (0.5 to correct double count)
    E_site = -0.5 * spins.to(torch.float32) * neigh
    # sum over lattice
    E = E_site.sum(dim=(1, 2))
    return E

def swendsen_wang_step_open(spins, beta, device="cuda"):
    """
    One Swendsen–Wang update for 2D Ising model with open boundaries.
    spins: (L,L) tensor with values ±1
    beta: inverse temperature
    """
    L = spins.shape[0]
    p = 1.0 - torch.exp(-2.0 * beta)

    # Neighbor bonds (open boundary: no wraparound)
    bond_right = torch.zeros_like(spins, dtype=torch.bool, device=device)
    bond_down  = torch.zeros_like(spins, dtype=torch.bool, device=device)

    bond_right[:, :-1] = (spins[:, :-1] == spins[:, 1:]) & \
                         (torch.rand((L, L-1), device=device) < p)

    bond_down[:-1, :] = (spins[:-1, :] == spins[1:, :]) & \
                        (torch.rand((L-1, L), device=device) < p)

    # ---- cluster labeling (union-find on CPU, fine for 16x16) ----
    bonds_cpu = (bond_right.cpu().numpy(), bond_down.cpu().numpy())
    spins_cpu = spins.cpu().numpy()

    parent = np.arange(L*L)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x,y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    # Right bonds
    for i in range(L):
        for j in range(L-1):
            if bonds_cpu[0][i,j]:
                union(i*L+j, i*L+(j+1))
    # Down bonds
    for i in range(L-1):
        for j in range(L):
            if bonds_cpu[1][i,j]:
                union(i*L+j, (i+1)*L+j)

    # Relabel clusters
    labels = np.array([find(i) for i in range(L*L)]).reshape(L, L)

    # Random flips per cluster
    unique_clusters = np.unique(labels)
    flip_dict = {c: (1 if torch.rand(1).item() < 0.5 else -1) for c in unique_clusters}
    flips = np.vectorize(flip_dict.get)(labels)

    new_spins = spins_cpu * flips
    return torch.tensor(new_spins, device=device, dtype=spins.dtype)

def swendsen_wang_step_periodic(spins, beta, device="cuda"):
    """
    One Swendsen–Wang update for 2D Ising model with periodic boundaries.
    spins: (L,L) tensor with values ±1
    beta: inverse temperature
    """
    L = spins.shape[0]
    spins = spins.to(device)
    p = 1.0 - torch.exp(-2.0 * beta)  # J=1

    # --- Neighbor bonds (periodic: wraparound via torch.roll) ---
    # Right bonds: bond between (i, j) and (i, (j+1) % L)
    bond_right = (spins == torch.roll(spins, shifts=-1, dims=1)) & \
                 (torch.rand((L, L), device=device) < p)

    # Down bonds: bond between (i, j) and ((i+1) % L, j)
    bond_down  = (spins == torch.roll(spins, shifts=-1, dims=0)) & \
                 (torch.rand((L, L), device=device) < p)

    # ---- cluster labeling (union-find on CPU, fine for 16x16) ----
    bonds_cpu = (bond_right.cpu().numpy(), bond_down.cpu().numpy())
    spins_cpu = spins.cpu().numpy()

    parent = np.arange(L * L)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    # Right bonds: (i, j) ↔ (i, (j+1) % L)
    for i in range(L):
        for j in range(L):
            if bonds_cpu[0][i, j]:
                x = i * L + j
                y = i * L + ((j + 1) % L)
                union(x, y)

    # Down bonds: (i, j) ↔ ((i+1) % L, j)
    for i in range(L):
        for j in range(L):
            if bonds_cpu[1][i, j]:
                x = i * L + j
                y = ((i + 1) % L) * L + j
                union(x, y)

    # Relabel clusters
    labels = np.array([find(i) for i in range(L * L)]).reshape(L, L)

    # Random flips per cluster
    unique_clusters = np.unique(labels)
    flip_dict = {c: (1 if torch.rand(1).item() < 0.5 else -1)
                 for c in unique_clusters}
    flips = np.vectorize(flip_dict.get)(labels)

    new_spins = spins_cpu * flips
    return torch.tensor(new_spins, device=device, dtype=spins.dtype)

def autocorr_fft(x):
    """Return normalized autocorrelation function r[k] for k=0..n-1"""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    # next power-of-two for zero-padding (speed)
    nfft = 1 << (2*n - 1).bit_length()
    f = np.fft.rfft(x, n=nfft)
    acf = np.fft.irfft(f * np.conjugate(f), n=nfft)[:n]
    acf /= acf[0]
    return acf

def integrated_time_from_acf(r):
    """
    Estimate integrated autocorrelation time tau_int using automatic windowing.
    r: autocorrelation array with r[0]=1
    Returns tau_int (float)
    """
    # iterative automatic windowing (Sokal/Madras style)
    tau = 0.5
    for _ in range(50):
        # window length = min(max_lag, int(5 * tau))
        max_lag = len(r) - 1
        window = int(min(max_lag, max(1, 5 * tau)))
        # sum r[1..window]
        s = r[1:window+1].sum() if window >= 1 else 0.0
        tau_new = 0.5 + s
        if abs(tau_new - tau) < 1e-8:
            tau = tau_new
            break
        tau = tau_new
    return tau

def estimate_tau(obs):
    """
    obs: 1D array of observable (e.g. magnetization) vs step
    returns: tau_int, acf (array)
    """
    r = autocorr_fft(obs)
    tau = integrated_time_from_acf(r)
    return tau, r


def correlation_function_single(spins, max_r=None, margin=1, direction="row"):
    """
    Compute C(r) for a single Ising configuration.
    
    spins : (L, L) ndarray of ±1
    max_r : maximum distance. If None, use L - 2*margin - 1
    margin : how many sites to drop from each boundary
    direction : "row" or "col"
    
    Returns
    -------
    r_vals : array of distances
    C : array of correlations
    """
    L = spins.shape[0]
    bulk = L - 2*margin
    if bulk <= 1:
        raise ValueError("Margin too large, no bulk left")
    if max_r is None:
        max_r = bulk - 1

    r_vals = np.arange(1, max_r+1)
    C = np.zeros_like(r_vals, dtype=float)

    if direction == "row":
        for idx, r in enumerate(r_vals):
            products = []
            for i in range(margin, L-margin):
                for j in range(margin, L-margin-r):
                    products.append(spins[i,j] * spins[i,j+r])
            C[idx] = np.mean(products)
    elif direction == "col":
        for idx, r in enumerate(r_vals):
            products = []
            for i in range(margin, L-margin-r):
                for j in range(margin, L-margin):
                    products.append(spins[i,j] * spins[i+r,j])
            C[idx] = np.mean(products)
    else:
        raise ValueError("direction must be 'row' or 'col'")
    
    return r_vals, C


def correlation_function_ensemble(spin_configs, margin=1, direction="row"):
    """
    Compute ensemble-averaged correlation function C(r) over many configs.
    
    spin_configs : array of shape (n_samples, L, L)
        Each entry is a spin configuration (±1).
    margin : int
        Boundary margin to drop
    direction : "row" or "col"
    
    Returns
    -------
    r_vals : array of distances
    C_mean : mean correlation function (length = max_r)
    C_se   : standard error at each r
    """
    spin_configs = np.asarray(spin_configs)
    n_samples, L, _ = spin_configs.shape

    # use first config to determine r_vals
    r_vals, C0 = correlation_function_single(spin_configs[0], margin=margin, direction=direction)
    C_all = np.zeros((n_samples, len(r_vals)))

    C_all[0] = C0
    for s in range(1, n_samples):
        _, C_all[s] = correlation_function_single(spin_configs[s], margin=margin, direction=direction)

    C_mean = C_all.mean(axis=0)
    C_se = C_all.std(axis=0, ddof=1) / np.sqrt(n_samples)

    return C_mean
