import numpy as np
from kernels.base_kernel import Kernel

class CountingKernel(Kernel):
    """ Feature map counts the different patterns of length k in the sequence """
    
    patterns_2 = sorted(["".join([l1, l2]) for l1 in letters for l2 in letters])
    patterns_3 = sorted(["".join([l1, l2, l3]) for l1 in letters for l2 in letters for l3 in letters])
    
    def __init__(self, k=3, name="Counting Kernel"):
        
        super().__init__()
        self.feature_map = False
     
    
    def align(self, str_0, str_1):
        """ Align 2 strings. Too long for now. Last part can probably be removed"""
        n = len(str_0) 
        m = len(str_1) 
        D = np.zeros((n + 1,m + 1))
        D[0,0] = 0
        aln_0 = ""
        aln_1 = ""
        tracks = np.zeros((n+1, m+1))
        # initializa D which keeps tracks of the scores to align substrings 
        for j in range(m):
            D[0,j+1] = D[0, j] + gap_score
        for i in range(n):
            D[i+1, 0] = D[i,0] + gap_score

        for i in range(1, n+1):
            for j in range(1, m+1):
                match = D[i-1, j-1] + matrix_sim[index_w[str_0[i-1]], index_w[str_1[j-1]]]
                gaps = D[i, j-1] + gap_score
                gapt = D[i-1,j] + gap_score
                D[i,j] = max(match, gaps, gapt)
                tracks[i,j] = np.argmax([match, gaps, gapt])
        i = n
        j = m
        # Fill in D. Action tells if one should align or add "_" on a string or another
        while i > 0 and j > 0:

            action = tracks[i, j]
            if action == 0:
                aln_0 = str_0[i-1] + aln_0
                aln_1 = str_1[j-1] + aln_1
                i = i-1
                j = j-1
            elif action == 1:
                aln_0 = "_" + aln_0
                aln_1 = str_1[j-1] + aln_1

                j = j-1
            else:

                aln_0 = str_0[i-1] + aln_0
                aln_1 = "_" + aln_1
                i = i-1

        # Backtracking to align the strings. Can be removed if we just want the score
        # and return np.max(D) - gap_score * num_occurences("_"). But shouldn't change much 
        while i > 0:
            aln_0 = str_0[i-1] + aln_0
            aln_1 = '_' + aln_1
            i = i-1

        while j > 0:
            aln_1 = str_1[j-1] + aln_1
            aln_0 = '_' + aln_0
            j = j-1
        score = 0
        for i in range(len(aln_0)):
            a = aln_0[i]
            b = aln_1[i]
            if a == "_" or b == "_":
                score += gap_score
            else:
                score += matrix_sim[index_w[a], index_w[b]]

        return score
    
    def get_matrix(self, X):
        """ X is a dataframe. Returns an array """
        K = np.zeros((len(seq), len(seq)))
        seq = X["seq"]
        for i,s1 in enumerate(seq):
            for j in range(i):

                s2 = seq[i]
                score = self.align(s1, s2)
                K[i, j] = score
                K[j, i] = score
            K[i,i] = len(s1) * matrix_sim[0]
        return K 