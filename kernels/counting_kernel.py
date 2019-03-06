from kernels.base_kernel import Kernel
import numpy as np
import pandas as pd


class CountingKernel(Kernel):
    """ Feature map counts the different patterns of length k in the sequence """
    
    def __init__(self, k=(3,), m=0, max_size=200, dimismatch=None, name="Counting Kernel"):
        super().__init__(name, is_feature_map=True)
        letters = ["A", "C", "G", "T"]
        self.patterns = []
        if 1 in k:
            self.patterns += ["".join([l1]) 
                                    for l1 in letters]
        if 2 in k: 
            self.patterns += ["".join([l1, l2]) 
                                    for l1 in letters 
                                    for l2 in letters]
        if 3 in k:
            self.patterns += ["".join([l1, l2, l3]) 
                                    for l1 in letters 
                                    for l2 in letters 
                                    for l3 in letters]
  
        if 4 in k:
            self.patterns += ["".join([l1, l2, l3, l4]) 
                                    for l1 in letters 
                                    for l2 in letters 
                                    for l3 in letters
                                    for l4 in letters]
        if 5 in k:
            self.patterns += ["".join([l1, l2, l3, l4, l5]) 
                                    for l1 in letters 
                                    for l2 in letters 
                                    for l3 in letters
                                    for l4 in letters
                                    for l5 in letters]
            
        if 6 in k:
            self.patterns += ["".join([l1, l2, l3, l4, l5, l6]) 
                                    for l1 in letters 
                                    for l2 in letters 
                                    for l3 in letters
                                    for l4 in letters
                                    for l5 in letters
                                    for l6 in letters]
        if 10 in k:
             self.patterns += ["".join([l1, l2, l3, l4, l5, l6, l7, l8, l9, l10]) 
                                    for l1 in letters 
                                    for l2 in letters 
                                    for l3 in letters
                                    for l4 in letters
                                    for l5 in letters
                                    for l6 in letters
                                   for l7 in letters
                                   for l8 in letters
                                   for l9 in letters
                                   for l10 in letters
                              ]
            
        self.patterns = sorted(self.patterns)
        self.vocab = None
        self.dimismatch = dimismatch
        self.max_size = max_size
        self.k = k
        self.m = m
        
    def define_vocabulary(self, X_train):
        patterns = {pat:0 for pat in self.patterns}
        for seq in X_train:
            for pat in self.patterns:
                c = seq.count(pat)
                patterns[pat] += c
        vocab = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:self.max_size]
        self.vocab = sorted([v[0] for v in vocab])
     
    
    def _count_patterns_dimismatch(self, sequence):
        values = {pat:0 for pat in self.vocab}
        k = self.k[0]
        for pat in self.vocab:
            pat_arr = np.array(list(pat))
            for i in range(len(sequence) - self.k[0]):
                gram = np.array(list(sequence[i:i+k]))
                score = np.sum(gram == pat_arr)
              
                if score <= k - self.m - 1:
                    score = 0
                values[pat] += score
        return pd.Series(values) 
                
    
       
    def _count_patterns(self, sequence):
        if self.dimismatch and self.vocab is None:
            raise Error("Call define vocabulary before")
        if self.dimismatch:
            return self._count_patterns_dimismatch(sequence)
        values = {}

        for pat in self.patterns:
          
            c = sequence.count(pat)

            values[pat] = c

        return pd.Series(values)
    
    def feature_map(self, X, column="seq"):
        map_count = map(lambda x: self._count_patterns(x), 
                        X.reshape(-1))
        return np.array(list(map_count))
            
    def predict_function(self, alphas, data, X_pred):
        matrix = self.get_kernel_matrix(data, Xi=X_pred)
        return np.sign(np.sum(alphas * matrix, axis = 0))