import numpy as np
import torch 
import matplotlib.pyplot as plt

# w_vec of shape (n_features,), [w_0, w_1, w_2, ...] with w_0 being the intercept
# X out is of shape (n_samples, n_features) [1, x, x^2, ...]
# y out is of shape (n_samples,)
def make_lin_reg_dataset(w_vec, n_samples, n_features, noise_std):
    w_vec = torch.tensor(w_vec)

    assert w_vec.shape == (n_features,)
    
    x = torch.linspace(-10, 10, n_samples)
    
    # form features 
    X = torch.zeros(n_samples, n_features)
    for i in range(n_features):
        X[:, i] = x ** i
    
    # form targets 
    y = X @ w_vec + torch.randn(n_samples) * noise_std
    
    return X, y

def conv_dataset_to_str(X, y):
    total_str = ""
    for i in range(X.shape[0]):
        x = X[i, :]
        y_i = y[i]

        if i < X.shape[0] - 1:
            total_str += "({:.4f}, {:.4f}), ".format(x[1], y_i)
        else:
            total_str += "({:.4f}, {:.4f})".format(x[1], y_i)

    return total_str


def plot_lin_reg_dataset(X, y):
    plt.scatter(X[:, 1], y)
    plt.show()

def plot_given_w_vec(w_vec):
    x = torch.linspace(-10, 10, 100)
    X = torch.zeros(100, len(w_vec))
    for i in range(len(w_vec)):
        X[:, i] = x ** i
    y = X @ w_vec
    plt.plot(x, y)
    plt.show()

def get_least_squares_soln(X, y):
    n_features = X.shape[1]
    x_np = X[:, 1].detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    w_ls = np.polyfit(x_np, y_np, deg = n_features - 1)
    w_ls = torch.tensor(w_ls)
    return w_ls

