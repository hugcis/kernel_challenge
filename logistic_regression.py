import numpy as np
import math
import matplotlib.pyplot as plt

class LogisiticRegression():

    def __init__(self, x, y):
        # Data for the model
        self.x = x
        self.y = y
        self.n = len(x)
        # Parameters of the model, computed in the fit function
        self.w = np.zeros(x.shape[1] + 1,)



    def fit(self, N=100, e=None, step_size=0.001):
        """ Fit the model: compute w with the IRLS algorithm
        args :: N : int : number of iterations
                e float : precision for the optimization algorithm, optional,
                          replaces N
                step_size : float """

        x_ = np.append(np.ones((self.n,1)), self.x, axis=1)
        i = 0

        condition = True
        while condition:
            
            diag_eta = np.array([self.eta(x_i,self.w) for x_i in x_])
            D = np.diag(diag_eta * (1 - diag_eta))
            try:
                inv_mat = np.linalg.inv(x_.T.dot(D).dot(x_))
            except:
                print("Stopped the iteration : non sigular matrix")
                break
        
            #print("intermediaire shape")
            #print(inv_mat.dot(x_.T).shape)
            y = self.y.reshape(self.y.shape[0], 1)
            diag_eta = diag_eta.reshape(self.y.shape[0], 1)
            
            diff_w = inv_mat.dot(x_.T).dot(self.y - diag_eta) * step_size
            diff_w = diff_w.reshape(diff_w.shape[0],)
           
            self.w, prev_w = self.w + diff_w, self.w
            error = np.linalg.norm(self.w - prev_w)

            i += 1
            if e is not None:
                
                condition = (error > e)
            else:
                print(error)
                condition = (i <= N)
            


    def eta(self,x,w):
        z = w.T.dot(x)
       
        if z > 0:
            return 1 / (1 + math.exp(-1 * z))
        else:
            a = math.exp(z)
            return a / (a + 1)


    def predict(self, x):
        """ Returns p(y=1|x) """
        x_ = np.append(1, x)
        return self.eta(x_, self.w)

    def score(self, x, y):
        pred = [self.predict(x_i) for x_i in x]
 

        return np.sum(pred == y.reshape(len(pred))) / len(pred)
    
    def plot_boundary(self, N=100):
        """ Plot the boundary of the model """
        xmin = min(self.x[:,0]) - 2
        xmax = max(self.x[:,0]) + 2
        ymin = min(self.x[:,1]) - 2
        ymax = max(self.x[:,1]) + 2
        x1 = np.linspace(xmin, xmax, N)
        x2 = [- 1 * (self.w[0] + x * self.w[1]) / self.w[2] for x in x1]
        plt.plot(x1,x2)
        plt.ylim(top=ymax, bottom=ymin)
