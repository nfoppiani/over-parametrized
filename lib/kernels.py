import numpy as np
from scipy.spatial.distance import cdist

def exponential_kernel(x, y, sigma2=None):
    '''x and y should have last dimension == d'''
    assert x.shape[-1] == y.shape[-1]
    
    dist2 = cdist(x, y, metric='sqeuclidean')
    if sigma2 is None:
        sigma2 = x.shape[-1]
    kernel_matrix = np.exp(-dist2/(2*sigma2))
    return kernel_matrix

def linear_kernel(x, y, sigma2=None):
    '''x and y should have last dimension == d'''
    assert x.shape[-1] == y.shape[-1]  
    xy = np.tensordot(x, y, axes=([-1], [-1]))
    if sigma2 is None:
        sigma2 = x.shape[-1]
    xy /= sigma2
    return xy