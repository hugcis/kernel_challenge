import numpy as np
from cvxopt import matrix, solvers

class SVM:
    def __init__(self):
        self.fitted = False

    def predict(self, X):
        if not self.fitted:
            raise ValueError("Predict called on a non fitted estimator")
        return np.sign( X @ self.w )
    
    def fit(self, X, Y): 
        P = matrix((Y*X) @ (Y*X).T)
        q = matrix(-np.ones((X.shape[0], 1)))
        G = matrix(
            np.concatenate(
                [-np.diag(Y.reshape(-1)), 
                 np.diag(Y.reshape(-1))], 
                axis=0
            ).astype(np.double))
        
        h = matrix(
            np.concatenate(
                [np.zeros(X.shape[0]), 
                 (1/(2*X.shape[0])) * np.zeros(X.shape[0])], 
                axis=0))

        sol = solvers.qp(P, q, G, h)
        self.fitted = True
        self.w = np.sum(np.array(sol['x']) * Y * X, axis = 0)
        return self
    
    def score(self, X, Y):
        return sum((Y.reshape(-1)*self.predict(X)) > 0)/Y.shape[0]