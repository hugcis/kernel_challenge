import numpy as np
from kernels.base_kernel import Kernel
from scipy.spatial.distance import pdist, squareform

class RBFKernel:
    def __init__(self, sigma):
        self.sigma = sigma
    
    def get_kernel_matrix(self, X):
        return (np.exp(- squareform(pdist(X, 'euclidean')**2) / 
                (2 * self.sigma**2)))
    
    def predict_function(self, alphas, data, X):
        dists = np.sum((data[:, :, None] - X.T[None, :, :])**2, axis=1)
        kernel_mat = np.exp(- dists / (2 * self.sigma**2))
        return np.sign( np.sum(alphas * kernel_mat, axis=0) )