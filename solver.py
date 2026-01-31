from scipy.sparse.linalg import spsolve

def solve_sparse(G, I):
    return spsolve(G, I) 
