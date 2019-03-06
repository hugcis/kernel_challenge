import numpy as np
from kernels.base_kernel import Kernel
from scipy.spatial.distance import pdist, squareform

class RBFKernel:
    def __init__(self, sigma, name="RBF Kernel"):
        self.name = name
        self.sigma = sigma
    
    def get_kernel_matrix(self, X):
        return (np.exp(- squareform(pdist(X, 'euclidean')**2) / 
                (2 * self.sigma**2)))
    
    def predict_function(self, alphas, data, X):
        # Much faster if the kernel mat is symmetric
        if np.all(data == X):
            kernel_mat = (np.exp(- squareform(pdist(X, 'euclidean')**2)/ 
                          (2 * self.sigma**2)))
        else:
            kernel_mat = np.zeros((data.shape[0], X.shape[0]))
            for i in range(X.shape[0]):
                kernel_mat[:, i] = np.exp(
                    - np.sum((data[:, :] - X.T[:, i])**2, axis=1)/ 
                    (2 * self.sigma**2))
    
        return np.sign( np.sum(alphas.reshape(-1, 1) * kernel_mat, 
                        axis=0) )