import numpy as np
from kernels.rbf import RBFKernel
from kernels.linear import LinearKernel

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class KernelLogistic:
    def __init__(self, kernel, lbda=1, tol=1e-6, **kwargs):
        self.lbda = lbda 
        self.tol = tol

        if kernel == 'linear':
            self.kernel = LinearKernel()
        elif kernel == 'rbf':
            # Default sigma is 1 if not given
            self.kernel = RBFKernel(kwargs.get('sigma', 1))
        else:
            self.kernel = kernel

    def fit(self, X, Y):
        self.t_data = X.copy() # store data for predictions
        kernel_mat = self.kernel.get_kernel_matrix(X)
 
        alphas = 0.001 * np.ones(kernel_mat.shape[0])
        alphas_new = alphas.copy()
        start = True

        while np.linalg.norm(alphas_new - alphas) > self.tol or start:
            print("Difference {:.5f}".format(np.linalg.norm(alphas_new - alphas)))
            start = False
            alphas = alphas_new.copy()

            m = kernel_mat @ alphas
            P = sigmoid(m * Y.reshape(-1))
            W = np.diag(sigmoid(m) * sigmoid(-m))
            z = m + Y.reshape(-1) / sigmoid(-m * Y.reshape(-1))
            
            center_term = np.sqrt(W) @ kernel_mat @ np.sqrt(W)
            inverted_mat = np.linalg.inv(center_term + 
                                         np.eye(kernel_mat.shape[0]))
            
            alphas_new = ((np.sqrt(W) @ inverted_mat @ np.sqrt(W)) @ 
                          Y.reshape(-1))
        
        self.fitted = True
        self.alphas = alphas_new
        
    def predict(self, X_pred):
        if not self.fitted:
            raise ValueError("Predict called on a non fitted estimator")
        
        return self.kernel.predict_function(self.alphas[:, None], 
                                            self.t_data, 
                                            X_pred)
    
    def score(self, X, Y):
        return sum((Y.reshape(-1) * self.predict(X)) > 0)/Y.shape[0]
