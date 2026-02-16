from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import splu
import numpy as np

def solve_sparse(G, I):
    return spsolve(G, I) 

def solve_LU(G):
    return splu(G)

def get_node_and_branch_currents(lu, sources):
    VI = lu.solve(sources)
    return VI

def solve_adjoint(lu, target, var_map, total_dim):
    """
    target can be a node number (e.g. 2) or a name (e.g. 'V1')
    """
    d = np.zeros(total_dim)
    idx = var_map[target]  # Works for both nodes and branches!
    d[idx] = 1.0
    return lu.solve(d, trans='T')
