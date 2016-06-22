import numpy as np
import pandas as pd 
from numpy.linalg import svd




if __name__ == '__main__':
    df = pd.read_csv()


    U, s , V = svd(df, full_matrices=False)
    order = s.argsort()[::-1]
    s = s[order]
    U = U[order]
    V = V[order]
    S = np.dias(s)
    
    df2 = pd.read_csv()
    U2, s2, V2, = svd(df2, full_matrices=False)
    order2 = s2.argsort()[::-1]
    U2 = U2[order2]
    V2 = V2[order2]
    s2 = s2[order2]
    S2  = np.diag(s[order2])
