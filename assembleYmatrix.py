import numpy as np
from scipy.sparse import lil_matrix

def stamp_resistor(Y, n1, n2, R, node_index):
    """
    Stamps a resistor into the admittance matrix Y.
    Y: NxN sparse lilmatrix (complex)
    n1, n2: node numbers
    R: resistance value
    node_index: dict mapping node numbers to matrix indices
    """
    g = 1.0 / R
    if n1 != 0:
        i = node_index[n1]
        Y[i, i] += g
        if n2 != 0:
            j = node_index[n2]
            Y[i, j] -= g
    if n2 != 0:
        j = node_index[n2]
        Y[j, j] += g
        if n1 != 0:
            i = node_index[n1]
            Y[j, i] -= g

def stamp_capacitor(Y, n1, n2, C, w, node_index):
    """
    Stamps a capacitor into the admittance matrix Y.
    Y: NxN sparse lilmatrix (complex)
    n1, n2: node numbers
    C: capacitance (Farads)
    w: angular frequency (rad/s)
    node_index: dict mapping node numbers to matrix indices
    """
    g = 1j * w * C
    
    if n1 != 0:
        i = node_index[n1]
        Y[i, i] += g
        if n2 != 0:
            j = node_index[n2]
            Y[i, j] -= g
    if n2 != 0:
        j = node_index[n2]
        Y[j, j] += g
        if n1 != 0:
            i = node_index[n1]
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

def stamp_independent_voltage(Y, I_rhs, n1, n2, value, idx, node_index):
    """
    Stamps an independent voltage source using MNA (extra row/col).
    Y: NxN sparse lilmatrix (complex)
    I_rhs: RHS vector (complex)
    n1, n2: node numbers (n1=positive, n2=negative)
    value: voltage value (Volts)
    idx: matrix index for the branch current variable
    node_index: dict mapping node numbers to matrix indices
    """
    if n1 != 0:
        i = node_index[n1]
        Y[i, idx] += 1
        Y[idx, i] += 1
    if n2 != 0:
        j = node_index[n2]
        Y[j, idx] -= 1
        Y[idx, j] -= 1
        
    I_rhs[idx] = value

def stamp_inductor(Y, n1, n2, value, w, idx, node_index):
    """
    Stamps an inductor using MNA. Handles DC (short) and AC (impedance).
    Y: NxN sparse lilmatrix (complex)
    n1, n2: node numbers
    value: inductance (Henrys)
    w: angular frequency (rad/s)
    idx: matrix index for the branch current variable
    node_index: dict mapping node numbers to matrix indices
    """
    # Equation: V_n1 - V_n2 - (jwL)*I_L = 0
    # At DC (w=0), this becomes V_n1 - V_n2 = 0 (Short Circuit)
    
    Z_L = 1j * w * value
    
    # 1. KCL Connections (Same as Voltage Source)
    if n1 != 0:
        i = node_index[n1]
        Y[i, idx] += 1
        Y[idx, i] += 1
    if n2 != 0:
        j = node_index[n2]
        Y[j, idx] -= 1
        Y[idx, j] -= 1
        
    # 2. Impedance Term (subtracted from diagonal)
    Y[idx, idx] -= Z_L

def stamp_vccs(Y, n1, n2, n3, n4, value, node_index):
    """
    Stamps a VCCS into the admittance matrix Y.
    Y: NxN sparse lilmatrix (complex)
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


def generate_stamps(components, node_index, mna_index, total_dim, w=0):
    """
    w: Angular frequency (rad/s). Set to 0 for DC.
    """
    # Matrix must be complex to handle AC, even if w=0
    Y = lil_matrix((total_dim, total_dim), dtype=complex)
    I = np.zeros(total_dim, dtype=complex)

    for name, comp in components.items():
        # Extract basic nodes (default to 0 if not present)
        n1 = comp.get("n1", 0)
        n2 = comp.get("n2", 0)
        n3 = comp.get("n3", 0)
        n4 = comp.get("n4", 0)
        value = comp["value"]

        if name.startswith("R"):
            stamp_resistor(Y, n1, n2, value, node_index)

        elif name.startswith("C"):
            stamp_capacitor(Y, n1, n2, value, w, node_index)

        elif name.startswith("I"):
            stamp_current_source(I, n1, n2, value, node_index)

        elif name.startswith("G"):
            stamp_vccs(Y, n1, n2, n3, n4, value, node_index)

        # --- MNA COMPONENTS ---
        elif name.startswith("V"):
            idx = mna_index[name]
            stamp_independent_voltage(Y, I, n1, n2, value, idx, node_index)

        elif name.startswith("L"):
            idx = mna_index[name]
            stamp_inductor(Y, n1, n2, value, w, idx, node_index)

    return Y.tocsr(), I

