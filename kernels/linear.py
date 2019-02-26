import numpy as np

class LinearKernel:
    def get_kernel_matrix(self, X):
        return X @ X.T
    
    def predict_function(self, alphas, data, X):
        w = np.sum(alphas * data, axis = 0)
        return np.sign( X @ w )