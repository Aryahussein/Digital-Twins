import numpy as np
try:
    from constants import Vt
except ImportError:
    Vt = 0.0258

# =============================================================================
# HELPERS
# =============================================================================
def get_idx(node, node_map):
    """Returns matrix index for a node, or None if Ground (0)."""
    if node == 0 or node is None:
        return None
    return node_map.get(node)

# =============================================================================
# LINEAR STATIC STAMPS (Resistors, Sources, VCCS)
# =============================================================================
def stamp_resistor(Y, sources, comp, node_map, **kwargs):
    n1, n2, R = comp["n1"], comp["n2"], comp["value"]
    if R == 0: R = 1e-12 # Avoid division by zero
    g = 1.0 / R
    i, j = get_idx(n1, node_map), get_idx(n2, node_map)
    
    if i is not None:
        Y[i, i] += g
        if j is not None:
            Y[i, j] -= g
            Y[j, i] -= g
    if j is not None:
        Y[j, j] += g

def stamp_vccs(Y, sources, comp, node_map, **kwargs):
    n1, n2, n3, n4, gm = comp["n1"], comp["n2"], comp["n3"], comp["n4"], comp["value"]
    j, k = get_idx(n1, node_map), get_idx(n2, node_map) # Current connection
    x, y = get_idx(n3, node_map), get_idx(n4, node_map) # Control connection
    
    if j is not None:
        if x is not None: Y[j, x] += gm
        if y is not None: Y[j, y] -= gm
    if k is not None:
        if x is not None: Y[k, x] -= gm
        if y is not None: Y[k, y] += gm

def stamp_independent_voltage(Y, sources, comp, node_map, **kwargs):
    n1, n2, val, name = comp["n1"], comp["n2"], comp["value"], kwargs["name"]
    idx = node_map[name] 
    i, j = get_idx(n1, node_map), get_idx(n2, node_map)
    
    if i is not None:
        Y[i, idx] += 1
        Y[idx, i] += 1
    if j is not None:
        Y[j, idx] -= 1
        Y[idx, j] -= 1
    sources[idx] = val

def stamp_current_source(Y, sources, comp, node_map, **kwargs):
    n1, n2, val = comp["n1"], comp["n2"], comp["value"]
    i, j = get_idx(n1, node_map), get_idx(n2, node_map)
    
    if i is not None: sources[i] -= val
    if j is not None: sources[j] += val

# =============================================================================
# TRANSIENT DYNAMIC STAMPS (Capacitors, Inductors - Backward Euler)
# =============================================================================
def stamp_capacitor_be(Y, sources, comp, node_map, **kwargs):
    """Backward Euler Companion Model for Capacitor."""
    dt, v_prev = kwargs["dt"], kwargs["v_prev"]
    n1, n2, C = comp["n1"], comp["n2"], comp["value"]
    
    g_eq = C / dt
    i, j = get_idx(n1, node_map), get_idx(n2, node_map)
    
    # 1. Stamp Conductance
    if i is not None:
        Y[i, i] += g_eq
        if j is not None:
            Y[i, j] -= g_eq
            Y[j, i] -= g_eq
    if j is not None:
        Y[j, j] += g_eq
        
    # 2. Stamp Current Source (Memory of previous voltage)
    v1_old = v_prev[i] if i is not None else 0.0
    v2_old = v_prev[j] if j is not None else 0.0
    i_eq = g_eq * (v1_old - v2_old)
    
    # Current source flows OUT of positive node
    if i is not None: sources[i] += i_eq
    if j is not None: sources[j] -= i_eq

def stamp_inductor_be(Y, sources, comp, node_map, **kwargs):
    """Backward Euler Companion Model for Inductor."""
    dt, v_prev, name = kwargs["dt"], kwargs["v_prev"], kwargs["name"]
    n1, n2, L = comp["n1"], comp["n2"], comp["value"]
    idx = node_map[name] # The index of the inductor current variable
    i, j = get_idx(n1, node_map), get_idx(n2, node_map)

    # 1. KCL Connections
    if i is not None:
        Y[i, idx] += 1
        Y[idx, i] += 1
    if j is not None:
        Y[j, idx] -= 1
        Y[idx, j] -= 1

    # 2. Impedance Term (L/dt)
    r_eq = L / dt
    Y[idx, idx] -= r_eq

    # 3. Memory Term (L/dt * i_old)
    # The current is stored in the solution vector at 'idx'
    i_old = v_prev[idx] 
    v_eq = (L / dt) * i_old
    sources[idx] -= v_eq

# =============================================================================
# NON-LINEAR STAMPS (Diodes)
# =============================================================================
def stamp_diode_linearized(Y, sources, comp, node_map, **kwargs):
    """Linearized Diode Model for Newton-Raphson."""
    v_guess = kwargs["v_guess"]
    n1, n2 = comp["n1"], comp["n2"]
    Is = comp.get("value", 1e-14) # Default saturation current
    
    idx1, idx2 = get_idx(n1, node_map), get_idx(n2, node_map)
    
    v1 = v_guess[idx1] if idx1 is not None else 0.0
    v2 = v_guess[idx2] if idx2 is not None else 0.0
    vd = v1 - v2
    
    # Simple overflow protection
    if vd > 3.0: vd = 3.0
    
    # Calculate linearized parameters
    # Id = Is * (e^(Vd/Vt) - 1)
    # Gd = Is/Vt * e^(Vd/Vt)
    # Ieq = Id - Gd * Vd
    
    exp_term = np.exp(vd / Vt)
    gd = (Is / Vt) * exp_term
    id_val = Is * (exp_term - 1)
    i_eq = id_val - (gd * vd)
    
    # Stamp G_eq (Conductance)
    if idx1 is not None:
        Y[idx1, idx1] += gd
        if idx2 is not None:
            Y[idx1, idx2] -= gd
            Y[idx2, idx1] -= gd
    if idx2 is not None:
        Y[idx2, idx2] += gd
        
    # Stamp I_eq (Current Source)
    if idx1 is not None: sources[idx1] -= i_eq
    if idx2 is not None: sources[idx2] += i_eq