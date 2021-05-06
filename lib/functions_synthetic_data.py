import numpy as np

def norm_squared(X, noise=0, rng=None):
    norms = (X**2).sum(axis=1)
    if noise != 0:
        return norms + noise * rng.normal(size=len(X))
    else:
        return norms

def exp_norm_squared(X, noise=0, sigma2=None, rng=None):
    norms = (X**2).sum(axis=1)
    if sigma2 is None:
        sigma2 = X.shape[-1]
    out = np.exp(-norms/(2*sigma2))
    if noise != 0:
        return out + noise * rng.normal(size=len(X))
    else:
        return out
    
def exp_norm(X, noise=0, sigma=None, rng=None):
    norms = np.sqrt((X**2).sum(axis=1))
    if sigma is None:
        sigma = X.shape[-1]
    out = np.exp(-norms/sigma)
    if noise != 0:
        return out + noise * rng.normal(size=len(X))
    else:
        return out