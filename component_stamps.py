import numpy as np
from constants import *

def get_idx(node, node_map):
    """Returns the matrix index for a node/name, or None if it is Ground (0)."""
    if node == 0 or node is None:
        return None
    return node_map.get(node)

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
    g = 1 / R
    i, j = get_idx(n1, node_map), get_idx(n2, node_map)

    if i is not None:
        Y[i, i] += g
        if j is not None:
            Y[i, j] -= g
            Y[j, i] -= g  # Symmetric cross-term
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
    if i is not None:
        sources[i] -= value
    if j is not None:
        sources[j] += value


def stamp_independent_voltage(Y, sources, comp, node_map, name):
    """
    Stamps an independent voltage source using MNA (extra row/col).
    Y: NxN sparse lilmatrix (complex)
    sources: RHS source vector (complex)
    n1, n2: node numbers (n1=positive, n2=negative)
    value: voltage value (Volts)
    name: matrix index for the branch current variable
    node_map: dict mapping node numbers to matrix indices
    """
    n1, n2 = comp["n1"], comp["n2"]
    value = comp["value"]
    idx = node_map[name]
    i, j = get_idx(n1, node_map), get_idx(n2, node_map)

    if i is not None:
        Y[i, idx] += 1
        Y[idx, i] += 1
    if j is not None:
        Y[j, idx] -= 1
        Y[idx, j] -= 1

    sources[idx] = value


def stamp_vccs(Y, source, comp, node_map, name):
    """
    Stamps a VCCS into the admittance matrix Y.
    Y: NxN sparse lilmatrix (complex)
    n1, n2: node numbers (of current source connection)
    n3, n4: node numbers (of voltage meter connection)
    value: gm value (A/V)
    node_map: dict mapping node numbers to matrix indices

    **Note: the indexing switches to rows deemed 'j' and 'k' and columns deemed 'x' and 'y'
    to match the indexing in the slides
    """
    n1, n2 = comp["n1"], comp["n2"]
    n3, n4 = comp["n3"], comp["n4"]
    value = comp["value"]

    j, k = get_idx(n1, node_map), get_idx(n2, node_map)
    x, y = get_idx(n3, node_map), get_idx(n4, node_map)

    if j is not None:
        if x is not None:
            Y[j, x] += value
        if y is not None:
            Y[j, y] -= value
    if k is not None:
        if x is not None:
            Y[k, x] -= value
        if y is not None:
            Y[k, y] += value


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
    n1, n2 = comp["n1"], comp["n2"]
    L = comp["value"]

    idx = node_map[name]
    i, j = get_idx(n1, node_map), get_idx(n2, node_map)

    Z_L = 1j * w * L

    # 1. KCL Connections (Same as Voltage Source)
    if i is not None:
        Y[i, idx] += 1
        Y[idx, i] += 1
    if j is not None:
        Y[j, idx] -= 1
        Y[idx, j] -= 1

    # 2. Impedance Term (subtracted from diagonal)
    Y[idx, idx] -= Z_L


# =============================================================================
# NON-LINEAR STAMPS (Diodes, OpAmp-tanh)
# =============================================================================
def pnjlim(v_new, v_old, critical_v):
    """
    Standard SPICE limiting algorithm for PN junctions.
    Prevents V_guess from jumping too far in one iteration.
    """
    if v_new > critical_v and abs(v_new - v_old) > (2 * Vt):
        if v_old > critical_v:
            v_limit = v_old + 2 * Vt * np.log(v_new / v_old)
        else:
            v_limit = critical_v
        return v_limit
    return v_new


def stamp_diode(Y, sources, comp, node_map, p_V_guess, V_guess):
    n1, n2 = comp["n1"], comp["n2"]
    if "value" in comp:
        Is = comp.get("value")  # Default saturation current
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

    # 1. Calculate linearization components
    exp_term = np.exp(n_vd_k / Vt)
    id_k = Is * (exp_term - 1)

    # gd = dI/dV = linear conductance
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
    if idx1 is not None:
        sources[idx1] -= ieq
    if idx2 is not None:
        sources[idx2] += ieq


def stamp_opamp_tanh(Y, sources, comp, node_map, p_V_guess, V_guess):
    """
    Nonlinear op-amp element in MNA form.

    Netlist line format (parsed into comp):
        A1 out vp vm Vsat=5 k=1e6

    Model:
        V(out) = Vsat * tanh(k * (V(vp) - V(vm)))

    Newton linearization at current guess:
        f(dV) = Vsat * tanh(k dV)
        g = df/ddV = Vsat * k * (1 - tanh^2(k dV))

        f(dV) ≈ f0 + g (dV - dV0) = (f0 - g dV0) + g dV

        Constraint becomes:
            V(out) - g*(V(vp) - V(vm)) = (f0 - g*dV0)

    MNA stamping uses an extra unknown (like a voltage source current), indexed by node_map[NAME].
    """
    # We must know this element's name to get its extra MNA index.
    # Best practice: store comp["name"] in the parser.
    name = comp.get("name", None)
    if name is None:
        raise ValueError(
            "Op-amp element missing comp['name']. "
            "Fix: store the element name in txt2dictionary.py (comp['name']=name) "
            "or modify assembleYmatrix.stamp_nonlinear_components to pass 'name' into stamp_opamp_tanh."
        )

    out = comp["out"]
    vp = comp["vp"]
    vm = comp["vm"]
    Vsat = float(comp.get("Vsat", 1.0))
    k_val = float(comp.get("k", 1e3))

    # Indices for nodes
    i_out = get_idx(out, node_map)
    i_vp = get_idx(vp, node_map)
    i_vm = get_idx(vm, node_map)

    # Extra MNA row/col index
    a = node_map[name]

    # Current guess voltages
    v_out = V_guess[i_out] if i_out is not None else 0.0
    v_p = V_guess[i_vp] if i_vp is not None else 0.0
    v_m = V_guess[i_vm] if i_vm is not None else 0.0
    dV0 = v_p - v_m

    # Evaluate tanh safely
    x0 = k_val * dV0
    t0 = np.tanh(x0)
    f0 = Vsat * t0

    # Slope (Jacobian contribution)
    # sech^2(x) = 1 - tanh^2(x)
    sech2 = 1.0 - t0 * t0
    g = Vsat * k_val * sech2

    # RHS constant term from linearization
    b = f0 - g * dV0

    # --- MNA coupling like a voltage source on node out to "equation row a"
    # KCL at node out includes +/- I_a:
    if i_out is not None:
        Y[i_out, a] += 1
        Y[a, i_out] += 1
    else:
        # If out is ground (unlikely), this model becomes weird; allow but it won't be useful.
        pass

    # --- Linearized constraint row:
    # V(out) - g*(V(vp) - V(vm)) = b
    if i_vp is not None:
        Y[a, i_vp] += -g
    if i_vm is not None:
        Y[a, i_vm] += +g

    sources[a] += b


# =============================================================================
# TRANSIENT DYNAMIC STAMPS (Capacitors, Inductors - Backward Euler)
# =============================================================================
def stamp_capacitor_be(Y, sources, comp, node_map, name, dt, v_prev):
    """
    Stamp a capacitor into the MNA matrix using Backward Euler.
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
    """
    n1 = comp["n1"]
    n2 = comp["n2"]
    L = comp["value"]

    idx = node_map[name]
    i, j = get_idx(n1, node_map), get_idx(n2, node_map)

    # 1. KCL Connections (Same as Voltage Source)
    if i is not None:
        Y[i, idx] += 1
        Y[idx, i] += 1
    if j is not None:
        Y[j, idx] -= 1
        Y[idx, j] -= 1

    # 2. Impedance Term (subtracted from diagonal)
    R_eq = L / dt
    Y[idx, idx] -= R_eq

    # Thevenin equivalent (L/dt)*i(t)
    V_eq = L * v_prev[idx] / dt
    sources[idx] -= V_eq