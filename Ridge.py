import numpy as np
import math
from kernels.linear import LinearKernel
from kernels.rbf import RBFKernel

class RidgeRegression():
    def __init__(self, x, y,kernel='linear',reg=0,**kwargs):
        # Data for the model
        self.x = x
        self.y = y
        self.n = len(x)
        self.reg=reg
        #Kernel
        if kernel == 'linear':
            self.kernel = LinearKernel()
        elif kernel == 'rbf':
            # Default sigma is 1 if not given
            self.kernel = RBFKernel(kwargs.get('sigma', 1))
        else:
            self.kernel = kernel

        # Parameters of the model
        self.w = np.zeros(self.n)


    def fit(self):
        """ Fit the model : Estimate w, which contains the biais w[0] """
        x_ = np.append(np.ones((self.n,1)), self.x, axis=1)
        kernel_mat = self.kernel.get_kernel_matrix(x_)
        #ans=x_.dot(x_.T)+ self.reg*self.n*np.ones((self.n,self.n))
        ans=kernel_mat+ self.reg*self.n*np.ones((self.n,self.n))
        ans_inv=np.linalg.pinv(ans)
        alpha=ans_inv.dot(self.y)
        w_hat = x_.T.dot(alpha)
        self.w = w_hat


    def predict(self, x):
        xi = np.append(np.ones((len(x),1)), x, axis=1)
        return self.w.T.dot(xi.T)
    
    def compute_misclassif_error(self, X, y):
        res=np.sign(self.predict(X)).T
        error=np.sum(res!= y)/len(y)
        return error

