import numpy as np
from cvxopt import matrix, solvers
from scipy.spatial.distance import pdist, squareform

class SVM:
    def __init__(self, C=1, kernel='linear'):
        self.fitted = False
        self.C = C
        self.kernel = kernel

    def predict(self, X):
        if not self.fitted:
            raise ValueError("Predict called on a non fitted estimator")
        
        if self.kernel == 'linear':
            w = np.sum(np.array(self.sol['x']) * self.t_data, axis = 0)
            return np.sign( X @ w )   
        elif self.kernel == 'rbf':
            
    
    def fit(self, X, Y): 
        self.t_data = X.copy()
        kernel_mat = self._get_kernel_matrix(X)
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
    
    def _get_kernel_matrix(self, X):
        if self.kernel == 'linear':
            return X @ X.T
        elif self.kernel == 'rbf':
            return np.exp(-squareform(pdist(X, 'euclidean')**2) / (2 * self.sigma**2))