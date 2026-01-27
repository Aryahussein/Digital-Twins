from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import splu


def solve_sparse(G, I):
    """
    G: nxn matrix
    I: nx1 vector
    """
    return spsolve(G, I)

def lu_factorize(Y):
    """
    Returns a super-LU object. Individual L and U matrices
    can be accessed as lu.L and lu.U, but is not encouraged.
    To use the transpose of the super-LU object,
    v = lu.solve(I, trans='T').
    :param Y:
    :return: lu
    """
    lu = splu(Y)
    return lu

