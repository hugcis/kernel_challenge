import numpy as np
from scipy.spatial.distance import pdist, squareform

class RBFKernel:
    def __init__(self, sigma):
        self.sigma = sigma
    
    def get_kernel_matrix(self, X):
        return np.exp(-squareform(pdist(X, 'euclidean')**2) / (2 * self.sigma**2))
    
    def predict_function(self, alphas, data, X):
        dist = np.exp(-np.sum((data[:, :, None] - 
                               X.T[None, :, :])**2, axis=1) / (2 * self.sigma**2))
        return np.sign( np.sum(alphas * dist, axis=0) )