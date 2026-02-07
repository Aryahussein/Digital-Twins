from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import splu

def solve_sparse(G, I):
    return spsolve(G, I) 

def solve_LU(G):
    return splu(G)
