import numpy as np
from scipy.sparse import lil_matrix

def stamp_resistor(Y, n1, n2, R, node_index):
    """
    Stamps a resistor into the admittance matrix Y.
    Y: NxN numpy array
    n1, n2: node numbers
    R: resistance value
    node_index: dict mapping node numbers to matrix indices
    """
    g = 1 / R
    if n1 != 0:
        i = node_index[n1]
        Y[i, i] += g
    if n2 != 0:
        j = node_index[n2]
        Y[j, j] += g
    if n1 != 0 and n2 != 0:
        i = node_index[n1]
        j = node_index[n2]
        Y[i, j] -= g
        Y[j, i] -= g

def stamp_current_source(I, n1, n2, value, node_index):
    """
    Stamps a current source into the RHS vector I.
    I: Nx1 numpy array
    n1, n2: node numbers
    value: current value (A)
    node_index: dict mapping node numbers to matrix indices
    """
    if n1 != 0:
        i = node_index[n1]
        I[i] -= value
    if n2 != 0:
        j = node_index[n2]
        I[j] += value

def stamp_vccs(Y, n1, n2, n3, n4, value, node_index):
    """
    Stamps a VCCS into the admittance matrix Y.
    Y: NxN numpy array
    n1, n2: node numbers (of current source connection)
    n3, n4: node numbers (of voltage meter connection)
    value: gm value (A/V)
    node_index: dict mapping node numbers to matrix indices

    **Note: the indexing switches to rows deemed 'j' and 'k' and columns deemed 'x' and 'y' to match the indexing in the slides 
    """
    print("Processing stamp_vccs and n1=", n1)
    print("Processing stamp_vccs  n1 != 0 and n3 != 0 is ", n1 != 0 and n3 != 0)
    if n1 != 0 and n3 != 0:
        print("HEEEEEEEEEEEERE")
        j = node_index[n1]
        x = node_index[n3]
        Y[j,x] += value
    if n2 != 0 and n3 != 0:
        j = node_index[n2]
        y = node_index[n3]
        Y[j,y] -= value
    if n1 != 0 and n4 != 0:
        k = node_index[n1]
        x = node_index[n4]
        Y[k,x] -= value
    if n2 != 0 and n4 != 0:
        k = node_index[n2]
        y = node_index[n4]
        Y[k,y] += value


def generate_stamps(components, node_index):
    N = len(node_index)
    # LIL is efficient for incremental construction
    Y = lil_matrix((N, N), dtype=float)
    I = np.zeros(N)

    for name, comp in components.items():
        n1, n2, n3, n4, value = comp["n1"], comp["n2"], comp["n3"], comp["n4"], comp["value"]

        if name.startswith("R"):
            stamp_resistor(Y, n1, n2, value, node_index)

        elif name.startswith("I"):
            stamp_current_source(I, n1, n2, value, node_index)

        elif name.startswith("G"):
            stamp_vccs(Y, n1, n2, n3, n4, value, node_index)

    # Convert to CSR for solving using sparse matrix solver
    return Y.tocsr(), I


