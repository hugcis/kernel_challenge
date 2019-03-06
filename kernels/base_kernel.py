import numpy as np 


class Kernel:
    """ Abstract Class. Each kernel should inherit from this class """
    def __init__(self, name, is_feature_map=False):
        """ Feature map is a boolean"""
        
        self.name = name 
        self.is_feature_map = is_feature_map
        
    def feature_map(self, X) -> np.ndarray:
        pass

    def get_kernel_matrix(self, X, Xi=None):
        X_f = self.feature_map(X)
        if Xi is None:
            Xi_f = X_f
        else:
            Xi_f = self.feature_map(Xi)
        
        return X_f.dot(Xi_f.T).astype(np.double)

    def predict_function(self, alphas, data, X_pred):
        raise NotImplementedError(
            "You should implement that function for this kernel")
     