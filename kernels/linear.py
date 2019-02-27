import numpy as np
from kernels.base_kernel import Kernel

class LinearKernel(Kernel):
    def get_kernel_matrix(self, X):
        return X @ X.T
    
    def predict_function(self, alphas, data, X):
        w = np.sum(alphas * data, axis = 0)
        return np.sign( X @ w )
