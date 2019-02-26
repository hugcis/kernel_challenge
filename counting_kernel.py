from kernel import Kernel
import numpy as np
import pandas as pd


class CountingKernel(Kernel):
    """ Feature map counts the different patterns of length k in the sequence """
    
    def __init__(self, k=3, name="Counting Kernel"):
        super().__init__(name, is_feature_map=True)
        letters = ["A", "C","G", "T"]
        if k == 1:
            self.patterns = sorted(["".join([l1]) for l1 in letters])
        elif k == 2: 
            self.patterns = sorted(["".join([l1, l2]) for l1 in letters for l2 in letters])
        elif k == 3:
            self.patterns = sorted(["".join([l1, l2, l3]) for l1 in letters for l2 in letters for l3 in letters])
        self.k = k
       
    def count_patterns(self, sequence):
        values = {}
        for pat in self.patterns:
            c = sequence.count(pat)

            values[pat] = c

        return pd.Series(values)
    
    def feature_map(self, X, column="seq"):
        """ X is a pandas dataframe which contains "seq". Return an array""" 
        
        X[self.patterns] = X[column].apply(lambda x: self.count_patterns(x))
       
        return X[self.patterns].values
        
        
    
            
            
        
        
        