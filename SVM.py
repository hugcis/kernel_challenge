import numpy as np
from cvxopt import matrix, solvers
from kernels.linear import LinearKernel
from kernels.rbf import RBFKernel

class SVM:
    def __init__(self, C=1, kernel='linear', **kwargs):
        self.fitted = False
        self.C = C
        
        if kernel == 'linear':
            self.kernel = LinearKernel()
        elif kernel == 'rbf':
            # Default sigma is 1 if not given
            self.kernel = RBFKernel(kwargs.get('sigma', 1))
        else:
            self.kernel = kernel
            

    def predict(self, X_pred):
        if not self.fitted:
            raise ValueError("Predict called on a non fitted estimator")
        
        return self.kernel.predict_function(np.array(self.sol['x']), self.t_data, X_pred)

    
    def fit(self, X, Y): 
        self.t_data = X.copy() # store data for predictions
        
        kernel_mat = self.kernel.get_kernel_matrix(X)
        P = matrix(kernel_mat)
        q = matrix(-Y.astype(np.double))
        G = matrix(
            np.concatenate(
                [-np.diag(Y.reshape(-1)), 
                 np.diag(Y.reshape(-1))], 
                axis=0
            ).astype(np.double))
        
        h = matrix(
            np.concatenate(
                [np.zeros(X.shape[0]), 
                 self.C * np.ones(X.shape[0])], 
                axis=0))
        self.sol = solvers.qp(P, q, G, h)
        self.fitted = True
        
        return self
    
    def score(self, X, Y):
        return sum((Y.reshape(-1) * self.predict(X)) > 0)/Y.shape[0]
        
    def _add_bias(self, X):
        return np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)