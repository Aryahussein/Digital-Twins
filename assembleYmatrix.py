import numpy as np
from scipy.sparse import lil_matrix
from constants import *

def get_idx(node, node_map):
    """Returns the matrix index for a node/name, or None if it is Ground (0)."""
    if node == 0 or node is None:
        return None
    return node_map.get(node)

def stamp_resistor(Y, n1, n2, R, node_map):
    """
    Stamps a resistor into the admittance matrix Y.
    Y: NxN sparse lilmatrix (complex)
    n1, n2: node numbers
    R: resistance value
    node_map: dict mapping node numbers to matrix indices
    """
    g = 1/R
    i, j = get_idx(n1, node_map), get_idx(n2, node_map)
    
    if i is not None:
        Y[i, i] += g
        if j is not None:
            Y[i, j] -= g
            Y[j, i] -= g # Symmetric cross-term
    if j is not None:
        Y[j, j] += g

def stamp_capacitor(Y, n1, n2, C, w, node_map):
    """
    Stamps a capacitor into the admittance matrix Y.
    Y: NxN sparse lilmatrix (complex)
    n1, n2: node numbers
    C: capacitance (Farads)
    w: angular frequency (rad/s)
    node_map: dict mapping node numbers to matrix indices
    """
    g = 1j * w * C
    i, j = get_idx(n1, node_map), get_idx(n2, node_map)
    
    if i is not None:
        Y[i, i] += g
        if j is not None:
            Y[i, j] -= g
            Y[j, i] -= g
    if j is not None:
        Y[j, j] += g

def stamp_current_source(sources, n1, n2, value, node_map):
    """
    Stamps a current source into the RHS vector I.
    I: Nx1 numpy array
    n1, n2: node numbers
    value: current value (A)
    node_map: dict mapping node numbers to matrix indices
    """
    i, j = get_idx(n1, node_map), get_idx(n2, node_map)
    if i is not None: sources[i] -= value
    if j is not None: sources[j] += value

def stamp_independent_voltage(Y, sources, n1, n2, value, name, node_map):
    """
    Stamps an independent voltage source using MNA (extra row/col).
    Y: NxN sparse lilmatrix (complex)
    sources: RHS source vector (complex)
    n1, n2: node numbers (n1=positive, n2=negative)
    value: voltage value (Volts)
    name: matrix index for the branch current variable
    node_map: dict mapping node numbers to matrix indices
    """
    idx = node_map[name] 
    i, j = get_idx(n1, node_map), get_idx(n2, node_map)
    
    if i is not None:
        Y[i, idx] += 1
        Y[idx, i] += 1
    if j is not None:
        Y[j, idx] -= 1
        Y[idx, j] -= 1
    sources[idx] = value

def stamp_inductor(Y, n1, n2, value, w, name, node_map):
    """
    Stamps an inductor using MNA. Handles DC (short) and AC (impedance).
    Y: NxN sparse lilmatrix (complex)
    n1, n2: node numbers
    value: inductance (Henrys)
    w: angular frequency (rad/s)
    name: name of current variable
    node_map: dict mapping node numbers to matrix indices
    """
    # Equation: V_n1 - V_n2 - (jwL)*I_L = 0
    # At DC (w=0), this becomes V_n1 - V_n2 = 0 (Short Circuit)

    idx = node_map[name]
    i, j = get_idx(n1, node_map), get_idx(n2, node_map)
    
    Z_L = 1j * w * value
    
    # 1. KCL Connections (Same as Voltage Source)
    if i is not None:
        Y[i, idx] += 1
        Y[idx, i] += 1
    if j is not None:
        Y[j, idx] -= 1
        Y[idx, j] -= 1
        
    # 2. Impedance Term (subtracted from diagonal)
    Y[idx, idx] -= Z_L

def stamp_vccs(Y, n1, n2, n3, n4, value, node_map):
    """
    Stamps a VCCS into the admittance matrix Y.
    Y: NxN sparse lilmatrix (complex)
    n1, n2: node numbers (of current source connection)
    n3, n4: node numbers (of voltage meter connection)
    value: gm value (A/V)
    node_map: dict mapping node numbers to matrix indices

    **Note: the indexing switches to rows deemed 'j' and 'k' and columns deemed 'x' and 'y' to match the indexing in the slides 
    """
    # n1, n2: current connection; n3, n4: voltage sense
    # Rows: j, k; Columns: x, y

    j, k = get_idx(n1, node_map), get_idx(n2, node_map)
    x, y = get_idx(n3, node_map), get_idx(n4, node_map)

    if j is not None:
        if x is not None: Y[j, x] += value
        if y is not None: Y[j, y] -= value
    if k is not None:
        if x is not None: Y[k, x] -= value
        if y is not None: Y[k, y] += value


def pnjlim(v_new, v_old, critical_v):
    """
    Standard SPICE limiting algorithm for PN junctions.
    Prevents V_guess from jumping too far in one iteration.
    """
    if v_new > critical_v and abs(v_new - v_old) > (2 * Vt):
        if v_old > critical_v:
            # If we were already above critical, limit the rate of change
            v_limit = v_old + 2 * Vt * np.log(v_new / v_old)
        else:
            # If we are crossing the threshold, land exactly at critical_v
            v_limit = critical_v
        return v_limit
    return v_new

def stamp_diode(Y, sources, n1, n2, Is, p_V_guess, V_guess, node_map):
    idx1, idx2 = get_idx(n1, node_map), get_idx(n2, node_map)
    
    # Calculate current Vd from the previous iteration's guess
    v1 = V_guess[idx1] if idx1 is not None else 0
    v2 = V_guess[idx2] if idx2 is not None else 0
    vd_k = v1 - v2
    
    p_v1 = p_V_guess[idx1] if idx1 is not None else 0
    p_v2 = p_V_guess[idx2] if idx2 is not None else 0
    p_vd_k = p_v1 - p_v2

    # limit the amount v can jump at a time & prevent overflows
    n_vd_k = pnjlim(vd_k, p_vd_k, 5)

    # 1. Calculate linearization components
    exp_term = np.exp(n_vd_k / Vt)
    id_k = Is * (exp_term - 1)

    gd = (Is / Vt) * exp_term
    
    # linearized companion model
    ieq = id_k - gd * n_vd_k
    
    # 2. Stamp gd into Y (like a resistor)
    if idx1 is not None:
        Y[idx1, idx1] += gd
        if idx2 is not None:
            Y[idx1, idx2] -= gd
            Y[idx2, idx1] -= gd
    if idx2 is not None:
        Y[idx2, idx2] += gd
        
    # 3. Stamp Ieq into RHS vector
    if idx1 is not None: sources[idx1] -= ieq
    if idx2 is not None: sources[idx2] += ieq


def update_nonlinear_stamps(Y_ori, sources_ori, components, node_map, prev_V_guess, V_guess):
    """
    Maintains the base linear circuit and injects updated non-linear models.
    """
    # Create fresh copies of the linear base for this iteration
    Y = Y_ori.copy().tolil() 
    sources = sources_ori.copy()
    
    for name, comp in components.items():
        if name.startswith("D"):
            stamp_diode(Y, sources, comp['n1'], comp['n2'], comp.get('value', 1e-12), prev_V_guess, V_guess, node_map)

    return Y.tocsc(), sources

def generate_stamps(components, node_map, total_dim, w=0.0):
    """
    w: Angular frequency (rad/s). Set to 0 for DC.
    """
    # Matrix must be complex to handle AC, even if w=0
    dtype = float if w==0 else complex
    Y = lil_matrix((total_dim, total_dim), dtype=dtype)
    sources = np.zeros(total_dim, dtype=dtype)

    for name, comp in components.items():
        # Extract basic nodes (default to 0 if not present)
        n1 = comp.get("n1", 0)
        n2 = comp.get("n2", 0)
        n3 = comp.get("n3", 0)
        n4 = comp.get("n4", 0)
        val = comp["value"]

        if name.startswith("R"):
            stamp_resistor(Y, n1, n2, val, node_map)

        elif name.startswith("C"):
            stamp_capacitor(Y, n1, n2, val, w, node_map)
            
        elif name.startswith("I"):
            stamp_current_source(sources, n1, n2, val, node_map)

        elif name.startswith("G"):
            stamp_vccs(Y, n1, n2, n3, n4, val, node_map)

        elif name.startswith("V"):
            # We pass 'name' (e.g., 'V1') to look up its MNA row
            stamp_independent_voltage(Y, sources, n1, n2, val, name, node_map)

        elif name.startswith("L"):
            stamp_inductor(Y, n1, n2, val, w, name, node_map)

        # only stamp linear components here!!
        # stamp only the nonlinear ones using the update stamps
        # elif name.startswith("D"):
        #     if V_guess == None: 
        #         raise ValueError("Initial V_guess was not defined correctly!")
        #     prev_V_guess = V_guess.copy()
        #     stamp_diode(Y, sources, n1, n2, comp.get("Is"), prev_V_guess, V_guess, node_map)

    return Y.tocsc(), sources
