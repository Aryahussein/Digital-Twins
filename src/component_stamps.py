import numpy as np
from constants import *
import models

def get_idx(node, node_map):
    """Returns the matrix index for a node/name, or None if it is Ground (0)."""
    if node == 0 or node is None:
        return None
    return node_map.get(node)

def stamp_mna_connection(Y, comp, node_map, name):
    """
    Stamps ONLY the topological 1/-1 connections for MNA branch equations.
    Used for V and L.
    """
    idx = node_map[name]
    # Handle both V (n1, n2) and M/L (n_d, n_s) terminal names
    n1 = comp.get("n1")
    n2 = comp.get("n2")
    
    i, j = get_idx(n1, node_map), get_idx(n2, node_map)

    if i is not None:
        Y[i, idx] += 1
        Y[idx, i] += 1
    if j is not None:
        Y[j, idx] -= 1
        Y[idx, j] -= 1

# =============================================================================
# LINEAR STATIC STAMPS (Resistors, Sources, VCCS)
# =============================================================================
def stamp_resistor(Y, sources, comp, node_map, name):
    """
    Stamps a resistor into the admittance matrix Y.
    Y: NxN sparse lilmatrix (complex)
    n1, n2: node numbers
    R: resistance value
    node_map: dict mapping node numbers to matrix indices
    """
    n1, n2 = comp["n1"], comp["n2"]
    R = comp["value"]
    g = 1/R
    i, j = get_idx(n1, node_map), get_idx(n2, node_map)
    
    if i is not None:
        Y[i, i] += g
        if j is not None:
            Y[i, j] -= g
            Y[j, i] -= g # Symmetric cross-term
    if j is not None:
        Y[j, j] += g


def stamp_current_source(Y, sources, comp, node_map, name):
    """
    Stamps a current source into the RHS vector I.
    I: Nx1 numpy array
    n1, n2: node numbers
    value: current value (A)
    node_map: dict mapping node numbers to matrix indices
    """
    n1, n2 = comp["n1"], comp["n2"]
    value = comp["value"]
    i, j = get_idx(n1, node_map), get_idx(n2, node_map)
    if i is not None: sources[i] -= value
    if j is not None: sources[j] += value

# def stamp_independent_voltage(Y, sources, comp, node_map, name):
#     """
#     Stamps an independent voltage source using MNA (extra row/col).
#     Y: NxN sparse lilmatrix (complex)
#     sources: RHS source vector (complex)
#     n1, n2: node numbers (n1=positive, n2=negative)
#     value: voltage value (Volts)
#     name: matrix index for the branch current variable
#     node_map: dict mapping node numbers to matrix indices
#     """
#     n1, n2 = comp["n1"], comp["n2"]
#     value = comp["value"]
#     idx = node_map[name] 
#     i, j = get_idx(n1, node_map), get_idx(n2, node_map)
#
#     # print(f"stamping {name} with value {value}")
#     
#     if i is not None:
#         Y[i, idx] += 1
#         Y[idx, i] += 1
#     if j is not None:
#         Y[j, idx] -= 1
#         Y[idx, j] -= 1
#
#     sources[idx] = value

def stamp_independent_voltage(Y, sources, comp, node_map, name):
    """
    Value-only stamper for Voltage Sources. 
    Assumes Y-matrix structure is already set by stamp_mna_connection.
    """
    idx = node_map[name]
    sources[idx] = comp["value"]

def stamp_vccs(Y, source, comp, node_map, name):
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
    n1, n2 = comp["n1"], comp["n2"]
    n3, n4 = comp["n3"], comp["n4"]
    value = comp["value"]

    j, k = get_idx(n1, node_map), get_idx(n2, node_map)
    x, y = get_idx(n3, node_map), get_idx(n4, node_map)

    if j is not None:
        if x is not None: Y[j, x] += value
        if y is not None: Y[j, y] -= value
    if k is not None:
        if x is not None: Y[k, x] -= value
        if y is not None: Y[k, y] += value

# =============================================================================
# DYNAMIC STAMPS (capacitors, inductors)
# =============================================================================
def stamp_capacitor(Y, sources, comp, node_map, name, w=0.0):
    """
    Stamps a capacitor into the admittance matrix Y.
    Y: NxN sparse lilmatrix (complex)
    n1, n2: node numbers
    C: capacitance (Farads)
    w: angular frequency (rad/s)
    node_map: dict mapping node numbers to matrix indices
    """
    n1, n2 = comp["n1"], comp["n2"]
    C = comp["value"]
    g = 1j * w * C
    i, j = get_idx(n1, node_map), get_idx(n2, node_map)
    
    if i is not None:
        Y[i, i] += g
        if j is not None:
            Y[i, j] -= g
            Y[j, i] -= g
    if j is not None:
        Y[j, j] += g

def stamp_inductor(Y, sources, comp, node_map, name, w=0.0):
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

    L = comp["value"]
    idx = node_map[name]
    Z_L = 1j * w * L
        
    # 2. Impedance Term (subtracted from diagonal)
    Y[idx, idx] -= Z_L



# =============================================================================
# TRANSIENT DYNAMIC STAMPS (Capacitors, Inductors - Backward Euler)
# =============================================================================

def stamp_capacitor_be(Y, sources, comp, node_map, name, dt, v_prev):
    """
    Stamp a capacitor into the MNA matrix using Backward Euler.

    Parameters:
        Y       : Admittance matrix (numpy array)
        source  : RHS vector (numpy array)
        comp    : Component dictionary entry for capacitor
        dt      : Time step
        v_prev  : Previous timestep node voltage vector
    """

    n1 = comp["n1"]
    n2 = comp["n2"]
    C = comp["value"]

    # Equivalent conductance
    Y_eq = C / dt

    i1 = get_idx(n1, node_map)
    i2 = get_idx(n2, node_map)

    # Voltage difference from previous timestep
    v1_prev = v_prev[i1] if i1 is not None else 0.0
    v2_prev = v_prev[i2] if i2 is not None else 0.0

    I_eq = Y_eq * (v1_prev - v2_prev)

    # Stamp conductance matrix
    if i1 is not None:
        Y[i1, i1] += Y_eq
    if i2 is not None:
        Y[i2, i2] += Y_eq
    if i1 is not None and i2 is not None:
        Y[i1, i2] -= Y_eq
        Y[i2, i1] -= Y_eq

    # Stamp RHS vector
    if i1 is not None:
        sources[i1] += I_eq
    if i2 is not None:
        sources[i2] -= I_eq


def stamp_inductor_be(Y, sources, comp, node_map, name, dt, v_prev):
    """
    Stamp an inductor into the MNA matrix using Backward Euler.

    Parameters:
        Y       : Admittance matrix (numpy array)
        sources : RHS vector (numpy array)
        comp    : Component dictionary entry for inductor
        dt      : Time step
        v_prev  : Previous timestep node voltage vector
    """
    L = comp["value"]

    idx = node_map[name]

    # 2. Impedance Term (subtracted from diagonal)
    R_eq = L / dt
    Y[idx, idx] -= R_eq

    # Thevenin equivalent (L/dt)*i(t)
    V_eq = L * v_prev[idx] / dt
    sources[idx] -= V_eq

#==============================================================================
# ADJOINT RHS STAMPS
#==============================================================================

def get_branch_v(v_vector, comp, node_map):
    """Helper to get voltage across a component."""
    i, j = get_idx(comp["n1"], node_map), get_idx(comp["n2"], node_map)
    v1 = v_vector[i] if i is not None else 0.0
    v2 = v_vector[j] if j is not None else 0.0
    return v1 - v2, i, j

# ------------------------------------------------------------------------------
# BACKWARD EULER (BE) ADJOINT STAMPS
# ------------------------------------------------------------------------------
def stamp_adjoint_rhs_capacitor_be(J_hist, comp, node_map, name, dt, v_hat_next, adjoint_state):
    C = comp["value"]
    v_diff_next, i, j = get_branch_v(v_hat_next, comp, node_map)
    
    I_eq = (C / dt) * v_diff_next
    if i is not None: J_hist[i] += I_eq
    if j is not None: J_hist[j] -= I_eq

def update_adjoint_state_be(comp, node_map, name, dt, v_hat_next, v_hat, adjoint_state):
    # Backward Euler has no persistent state variable, so this does nothing!
    pass

# ------------------------------------------------------------------------------
# TRAPEZOIDAL (TR) ADJOINT STAMPS
# ------------------------------------------------------------------------------
def stamp_adjoint_rhs_capacitor_tr(J_hist, comp, node_map, name, dt, v_hat_next, adjoint_state):
    C = comp["value"]
    v_diff_next, i, j = get_branch_v(v_hat_next, comp, node_map)
    
    # Fetch the state from the "future" (which we already calculated in the backward loop)
    # Default to 0.0 if it's the very first backward step
    i_state = adjoint_state.get(name, 0.0) 
    
    I_eq = (2.0 * C / dt) * v_diff_next + i_state
    
    if i is not None: J_hist[i] += I_eq
    if j is not None: J_hist[j] -= I_eq

def update_adjoint_state_capacitor_tr(comp, node_map, name, dt, v_hat_next, v_hat, adjoint_state):
    C = comp["value"]
    v_diff_next, _, _ = get_branch_v(v_hat_next, comp, node_map)
    v_diff_curr, _, _ = get_branch_v(v_hat, comp, node_map)
    
    i_state_next = adjoint_state.get(name, 0.0)
    
    # Update the state for the *next* backward step (t_{n-1})
    adjoint_state[name] = (2.0 * C / dt) * (v_diff_next - v_diff_curr) + i_state_next
   

# =============================================================================
# NON-LINEAR STAMPS (Diodes)
# =============================================================================

def pnjlim(v_new, v_old, critical_v):
    """
    Standard SPICE limiting algorithm for PN junctions.
    Prevents V_guess from jumping too far in one iteration.
    """
    if v_new > critical_v and abs(v_new - v_old) > (2 * Vt):
        if v_old > critical_v and v_old > 0:
            # If we were already above critical, limit the rate of change
            v_limit = v_old + 2 * Vt * np.log(max(v_new / v_old, 1e-30))
        else:
            # If we are crossing the threshold, land exactly at critical_v
            v_limit = critical_v
        return v_limit
    return v_new

def stamp_diode(Y, sources, comp, node_map, p_V_guess, V_guess):
    n1, n2 = comp["n1"], comp["n2"]
    if "value" in comp:
        Is = comp.get("value") # Default saturation current
    elif "model" in comp:
        Is = comp["model_params"]["IS"]
    else:
        raise ValueError("No value or model specified for diode!")

    idx1, idx2 = get_idx(n1, node_map), get_idx(n2, node_map)

    # Calculate current Vd from the previous iteration's guess
    v1 = V_guess[idx1] if idx1 is not None else 0
    v2 = V_guess[idx2] if idx2 is not None else 0
    vd_k = v1 - v2

    p_v1 = p_V_guess[idx1] if idx1 is not None else 0
    p_v2 = p_V_guess[idx2] if idx2 is not None else 0
    p_vd_k = p_v1 - p_v2

    # limit the amount v can jump at a time & prevent overflows
    n_vd_k = pnjlim(vd_k, p_vd_k, 1)

    # ==========================================
    # 1. CALL THE MODEL
    # ==========================================
    # (Assuming Vt is imported from your constants file)
    diode_data = models.evaluate_diode(n_vd_k, Is, Vt)
    id_k = diode_data["I_D"]
    gd = diode_data["gd"]

    # linearized companion model (Equivalent current source)
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

def nmos_lim(vds_new, vds_old, critical_vds):
    """
    Standard SPICE limiting algorithm for nMOS junctions.
    Prevents Vds from jumping from one region to the other without stopping at the boundary.
        (currently just triode <-> saturation)
    """
    # Add a tiny epsilon to prevent division by zero
    vds_old_safe = max(abs(vds_old), 1e-6) 
    vds_new_safe = max(abs(vds_new), 1e-6)
    
    if vds_new > critical_vds and abs(vds_new - vds_old) > (2 * Vt):
        if abs(vds_old) > 1e-3: # Only use log-limiting if we have a stable old value
            v_limit = vds_old + 2 * Vt * np.log(vds_new_safe / vds_old_safe)
        else:
            # If starting from near-zero, just take a linear step
            v_limit = vds_old + 2 * Vt 
        return v_limit
    return vds_new


def nmos_region(vgs, vds, vth):
    """
    Finds mosfet region of operation: 0 cut-off, 1 triode, 2 sat, 3 subth, 4 breakdown
    """
    
    if vgs < vth:                           # off
        return 0
    elif vgs >= vth and vds < (vgs-vth):    # triode
        return 1
    elif vgs >= vth and vds >= (vgs-vth):   # sat
        return 2

def nmos_vgs_lim(vgs_new, vgs_old, vth):
    """Limits the change in Vgs to prevent numerical overflow."""
    step_limit = 0.5 # Maximum allowed jump in volts per iteration
    
    delta = vgs_new - vgs_old
    
    # If the jump is too large, dampen it
    if abs(delta) > step_limit:
        vgs_new = vgs_old + np.sign(delta) * step_limit
        
    return vgs_new



def stamp_mosfet(Y, sources, comp, node_map, p_V_guess, V_guess):
    n_d, n_g, n_s, n_b = comp["n_d"], comp["n_g"], comp["n_s"], comp["n_b"]
    
    params = comp["model_params"]
    inst_params = comp["inst_params"]
    m_type = comp["model_type"] 
    
    VTO = params["VTO"]       
    W = inst_params["W"]             
    L = inst_params["L"]             

    if "KP" in params:
        KP = params["KP"]           
    elif "MU" in params:
        KP = params["MU"] * params["C_OX"]
    else:
        raise ValueError("No KP or (MU and C_ox) specified for NMOS!")

    Bn =  W/L * KP 
    
    if (m_type == "NMOS"):
        idx_d = get_idx(n_d, node_map)
        idx_g = get_idx(n_g, node_map)
        idx_s = get_idx(n_s, node_map)
        idx_b = get_idx(n_b, node_map)
    else:
        raise RuntimeError(f"Unrecognized value {m_type}")

    # Grab raw voltages
    v_d = V_guess[idx_d] if idx_d is not None else 0
    v_g = V_guess[idx_g] if idx_g is not None else 0
    v_s = V_guess[idx_s] if idx_s is not None else 0
    
    p_v_g = p_V_guess[idx_g] if idx_g is not None else 0
    p_v_s = p_V_guess[idx_s] if idx_s is not None else 0
    p_v_d = p_V_guess[idx_d] if idx_d is not None else 0

    # APPLY LIMITING
    vgs_k = nmos_vgs_lim(v_g - v_s, p_v_g - p_v_s, VTO)
    vds_k = nmos_lim(v_d - v_s, p_v_d - p_v_s, vgs_k - VTO)

    nmos_data = models.evaluate_nmos(vgs_k, vds_k, VTO, Bn)
    Id = nmos_data["I_D"]
    gm = nmos_data["gm"]
    gds = nmos_data["gds"]

    # STAMPING
    # Transconductance (gm)
    if idx_d is not None:
        if idx_g is not None: Y[idx_d, idx_g] += gm
        if idx_s is not None: Y[idx_d, idx_s] -= gm
    if idx_s is not None:
        if idx_g is not None: Y[idx_s, idx_g] -= gm
        if idx_s is not None: Y[idx_s, idx_s] += gm

    # Output Conductance (gds)
    if idx_d is not None:
        Y[idx_d, idx_d] += gds
        if idx_s is not None:
            Y[idx_d, idx_s] -= gds
            Y[idx_s, idx_d] -= gds
            Y[idx_s, idx_s] += gds

    # Equivalent Current (Ieq)
    Ieq = Id - (gm * vgs_k) - (gds * vds_k)
    
    if idx_d is not None: sources[idx_d] -= Ieq
    if idx_s is not None: sources[idx_s] += Ieq


# def stamp_mosfet(Y, sources, comp, node_map, p_V_guess, V_guess):
#     # unCox = 100e-6          # Hard-coded value for a typical mosfet
#     # W_over_L = 1            # Hard-coded placeholder
#     # Bn = unCox*W_over_L     # Hard-coded placeholder
#     # Vth = 0.4               # Hard-coded placeholder
#
#     n_d, n_g, n_s, n_b = comp["n_d"], comp["n_g"], comp["n_s"], comp["n_b"]
#     
#     # Is = comp["model_params"]["IS"]
#     params = comp["model_params"]
#     inst_params = comp["inst_params"]
#     # print(params)
#     m_type = comp["model_type"] # Default to NMOS if not specified
#     # m_type = params["model"]
#     
#     # Default Level 1 parameters
#     VTO = params["VTO"]       # Zero-bias threshold voltage
#     W = inst_params["W"]             # width
#     L = inst_params["L"]             # Length
#
#
#     if "KP" in params:
#         KP = params["KP"]           # Gain Parameter = mu * C_ox
#     elif "MU" in params:
#         KP = params["MU"] * params["C_OX"]
#     else:
#         raise ValueError("No KP or (MU and C_ox) specified for NMOS!")
#
#     # LAMBDA = params["LAMBDA"] # Channel length modulation
#     # GAMMA = params.get("GAMMA", 0.0)   # Body effect parameter
#     # PHI = params.get("PHI", 0.6)       # Surface potential
#     # IS = params.get("IS", 1e-14)       # Body diode saturation current
#
#     Bn =  W/L * KP 
#     
#     if (m_type == "NMOS"):
#         idx_d, idx_g, idx_s, idx_b = get_idx(n_d, node_map), get_idx(n_g, node_map), get_idx(n_s, node_map), get_idx(n_b, node_map)
#     else:
#         raise RuntimeError(f"Unrecognized value {m_type}")
#
#     # Find voltages of Vgs and Vds for the new guess
#     v1 = V_guess[idx_d] if idx_d is not None else 0
#     v2 = V_guess[idx_g] if idx_g is not None else 0
#     v3 = V_guess[idx_s] if idx_s is not None else 0
#     vgs_k = v2 - v3
#     vds_k = v1 - v3
#
#     # Find voltages of Vgs and Vds for the previous guess
#     p_v1 = p_V_guess[idx_d] if idx_d is not None else 0
#     p_v2 = p_V_guess[idx_g] if idx_g is not None else 0
#     p_v3 = p_V_guess[idx_s] if idx_s is not None else 0
#     p_vgs_k = p_v2 - p_v3
#     p_vds_k = p_v1 - p_v3
#
#     # Limit the change of Vds possible in one guess
#     vds_k = nmos_lim(vds_k, p_vds_k, 0.2)
#         ### it will probably be necessary to limit the change of Vgs in one guess
#
#     # Find the operating region for current and previous 
#     region = nmos_region(vgs_k, vds_k, VTO)
#     # print(f"    Current nMOS region={region} for vgs={vgs_k}, vds={vds_k}, vth={VTO}")
#     p_region = nmos_region(p_vgs_k, p_vds_k, VTO)
#     # print(f"    Previous nMOS region={p_region} for vgs={p_vgs_k}, vds={p_vds_k}, vth={VTO}")
#
#     # Region has not changed
#     # if (region == p_region):
#     #     print(f"    region unchanged from {region}")
#     #     if (region == 0):
#     #         Id = 0                                                  # Id for Off state
#     #     elif (region == 1):
#     #         Id = Bn*((vgs_k-VTO)*vds_k-(vds_k**2)/2) + 1e-6         # Id for Triode state
#     #     elif (region == 2):
#     #         Id = Bn*((vgs_k-VTO)**2) + 1e-6                         # Id for Saturation state
#     #     else:
#     #         raise ValueError(f"    Error: nMOS {region} region operation not supported")
#     #
#     # # Region changed; set Id to the edge of the old/new regions
#     # else:
#     #     print(f"    region changed from {p_region} to {region}")
#     #     if (p_region == 0):                                                 # Transitioning from Off->Triode
#     #         Id = Bn*((VTO-VTO)*vds_k-(vds_k**2)/2)                          # Find Id with Vgs=VTO and current_Vds
#     #     
#     #     elif (p_region == 1):                                               # Transitioning from Triode to Off/Saturation
#     #
#     #         if (region == 0):                                               # Transitioning from Triode->Off
#     #             Id = Bn*((VTO-VTO)*vds_k-(vds_k**2)/2)                      # Find Id with Vgs=VTO and current_Vds
#     #         
#     #         elif (region == 2):                                             # Transitioning from Triode->Saturation
#     #             Id = Bn*((VTO-VTO)*(vgs_k-VTO)-((vgs_k-VTO)**2)/2)          # Find Id with Vds=Vgs-VTO and Vgs=VTO      ***Should Vgs=VTO be used here***
#     #
#     #     elif (p_region == 2):
#     #         Id = Bn*((vgs_k-VTO)*(vgs_k-VTO)-((vgs_k-VTO)**2)/2)            # Find Id with Vds=Vgs-VTO and Vgs=VTO      ***Should Vgs=VTO be used here***
#     #     else:
#     #         raise ValueError(f"    Error: pMOS {region} region operation not supported")
#     #
#     # print(f"    Id = {Id}")
#
#     # Initialize linearized parameters
#     gm = 0.0
#     gds = 1e-12  # Your "Physical Bridge" to prevent singular matrices
#     Id = 0.0
#
#     if (region == p_region):
#         if (region == 0): # OFF
#             Id = 0
#             gm = 0
#             gds = 1e-12 
#         elif (region == 1): # TRIODE (Linear)
#             Id = Bn * ((vgs_k - VTO) * vds_k - (vds_k**2) / 2)
#             gm = Bn * vds_k                        # dId/dVgs
#             gds = Bn * (vgs_k - VTO - vds_k) + 1e-12 # dId/dVds
#         elif (region == 2): # SATURATION
#             Id = 0.5 * Bn * ((vgs_k - VTO)**2)
#             gm = Bn * (vgs_k - VTO)                # dId/dVgs
#             gds = 1e-12
#
#     # if idx_d is not None: sources[idx_d] -= Id
#     # if idx_s is not None: sources[idx_s] += Id
#
#     # 1. Stamp Transconductance (Current at Drain/Source controlled by Gate)
#     if idx_d is not None:
#         if idx_g is not None: Y[idx_d, idx_g] += gm
#         if idx_s is not None: Y[idx_d, idx_s] -= gm
#     if idx_s is not None:
#         if idx_g is not None: Y[idx_s, idx_g] -= gm
#         if idx_s is not None: Y[idx_s, idx_s] += gm
#
#     # 2. Stamp Output Conductance (The "physical" path between Drain and Source)
#     if idx_d is not None:
#         Y[idx_d, idx_d] += gds
#         if idx_s is not None:
#             Y[idx_d, idx_s] -= gds
#             Y[idx_s, idx_d] -= gds
#             Y[idx_s, idx_s] += gds
#
#     # 3. Stamp the Equivalent Current Source
#     Ieq = Id - (gm * vgs_k) - (gds * vds_k)
#     if idx_d is not None: sources[idx_d] -= Ieq
#     if idx_s is not None: sources[idx_s] += Ieq

