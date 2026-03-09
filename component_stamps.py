import numpy as np
from constants import Vt


def get_idx(node, node_map):
    """Returns the matrix index for a node/name, or None if it is Ground (0)."""
    if node == 0 or node is None:
        return None
    return node_map.get(node)


# =============================================================================
# MNA TOPOLOGY STAMPS
# =============================================================================
def stamp_mna_connection(Y, comp, node_map, name):
    """
    Stamps ONLY the topological 1/-1 connections for MNA branch equations.
    Used for V, L, and O (opamp).
    """
    idx = node_map[name]
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
    """Stamps a resistor into the admittance matrix Y."""
    n1, n2 = comp["n1"], comp["n2"]
    R = comp["value"]
    g = 1.0 / R
    i, j = get_idx(n1, node_map), get_idx(n2, node_map)

    if i is not None:
        Y[i, i] += g
        if j is not None:
            Y[i, j] -= g
            Y[j, i] -= g
    if j is not None:
        Y[j, j] += g


def stamp_current_source(Y, sources, comp, node_map, name):
    """Stamps a current source into the RHS vector."""
    n1, n2 = comp["n1"], comp["n2"]
    value = comp["value"]
    i, j = get_idx(n1, node_map), get_idx(n2, node_map)
    if i is not None:
        sources[i] -= value
    if j is not None:
        sources[j] += value


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
    n1, n2: current connection nodes; n3, n4: voltage sense nodes.
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
# DYNAMIC STAMPS (capacitors, inductors) — AC frequency domain
# =============================================================================
def stamp_capacitor(Y, sources, comp, node_map, name, w=0.0):
    """Stamps a capacitor into the admittance matrix at angular frequency w."""
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
    Stamps an inductor using MNA.
    At DC (w=0): V_n1 - V_n2 = 0 (short circuit).
    At AC: V_n1 - V_n2 - jwL * I_L = 0.
    """
    L = comp["value"]
    idx = node_map[name]
    Z_L = 1j * w * L

    Y[idx, idx] -= Z_L


# =============================================================================
# TRANSIENT DYNAMIC STAMPS (Backward Euler companion models)
# =============================================================================
def stamp_capacitor_be(Y, sources, comp, node_map, name, dt, v_prev):
    """Stamp a capacitor using Backward Euler companion model."""
    n1 = comp["n1"]
    n2 = comp["n2"]
    C = comp["value"]

    Y_eq = C / dt

    i1 = get_idx(n1, node_map)
    i2 = get_idx(n2, node_map)

    v1_prev = v_prev[i1] if i1 is not None else 0.0
    v2_prev = v_prev[i2] if i2 is not None else 0.0
    I_eq = Y_eq * (v1_prev - v2_prev)

    # Stamp conductance
    if i1 is not None:
        Y[i1, i1] += Y_eq
    if i2 is not None:
        Y[i2, i2] += Y_eq
    if i1 is not None and i2 is not None:
        Y[i1, i2] -= Y_eq
        Y[i2, i1] -= Y_eq

    # Stamp RHS (history current)
    if i1 is not None:
        sources[i1] += I_eq
    if i2 is not None:
        sources[i2] -= I_eq


def stamp_inductor_be(Y, sources, comp, node_map, name, dt, v_prev):
    """
    Stamp an inductor using Backward Euler companion model.

    Note: v_prev[idx] is actually the previous branch current i_L(t-dt),
    since idx is the branch current index in the MNA solution vector.
    """
    L = comp["value"]
    idx = node_map[name]

    R_eq = L / dt
    Y[idx, idx] -= R_eq

    # History voltage: (L/dt) * i_L(t-dt)
    i_L_prev = v_prev[idx]
    V_eq = L * i_L_prev / dt
    sources[idx] -= V_eq


# =============================================================================
# NON-LINEAR STAMPS (Diodes, MOSFETs, Opamps)
# =============================================================================
def pnjlim(v_new, v_old, critical_v):
    """
    Standard SPICE limiting algorithm for PN junctions.
    Prevents V_guess from jumping too far in one iteration.
    """
    if v_new > critical_v and abs(v_new - v_old) > (2 * Vt):
        if v_old > critical_v and v_old > 0:
            v_limit = v_old + 2 * Vt * np.log(max(v_new / v_old, 1e-30))
        else:
            v_limit = critical_v
        return v_limit
    return v_new


def stamp_diode(Y, sources, comp, node_map, name, p_V_guess, V_guess):
    """Stamp a diode using a Newton companion model.

    Supported model parameters (SPICE-ish):
      - IS: saturation current
      - N: emission coefficient
      - RS: series resistance
      - BV: breakdown voltage (soft exponential breakdown)
      - IBV: current magnitude at V = -BV
      - NBV: breakdown emission coefficient
    """
    n1, n2 = comp["n1"], comp["n2"]
    idx1, idx2 = get_idx(n1, node_map), get_idx(n2, node_map)

    params = comp.get("model_params", {})

    if "value" in comp:
        Is = float(comp.get("value"))
    else:
        Is = float(params.get("IS", 1e-14))

    N = float(params.get("N", 1.0))
    RS = float(params.get("RS", 0.0))
    BV = float(params.get("BV", 0.0))
    IBV = float(params.get("IBV", 1e-3))
    NBV = float(params.get("NBV", 1.0))

    Vte = max(N * Vt, 1e-12)
    Vtb = max(NBV * Vt, 1e-12)

    # Node voltages & limiting
    v1 = V_guess[idx1] if idx1 is not None else 0.0
    v2 = V_guess[idx2] if idx2 is not None else 0.0
    vd_k = v1 - v2

    p_v1 = p_V_guess[idx1] if idx1 is not None else 0.0
    p_v2 = p_V_guess[idx2] if idx2 is not None else 0.0
    p_vd_k = p_v1 - p_v2

    vd_lim = pnjlim(vd_k, p_vd_k, 1.0)

    def I_and_g(vj):
        arg_f = np.clip(vj / Vte, -50.0, 50.0)
        ef = np.exp(arg_f)
        Ifwd = Is * (ef - 1.0)
        gfwd = (Is / Vte) * ef

        Ibr, gbr = 0.0, 0.0
        if BV > 0.0 and vj < -BV:
            x = -(vj + BV)
            arg_b = np.clip(x / Vtb, 0.0, 50.0)
            eb = np.exp(arg_b)
            Ibr = -IBV * eb
            gbr = (IBV / Vtb) * eb

        return (Ifwd + Ibr), (gfwd + gbr)

    # Solve implicit RS equation (small fixed-point loop)
    vj = vd_lim
    if RS > 0.0:
        for _ in range(8):
            Ij, _ = I_and_g(vj)
            vj_new = vd_lim - RS * Ij
            vj = 0.6 * vj + 0.4 * vj_new

    Id, g0 = I_and_g(vj)

    gd = g0 / (1.0 + g0 * RS) if RS > 0.0 else g0
    ieq = Id - gd * vd_lim

    # Stamp conductance
    if idx1 is not None:
        Y[idx1, idx1] += gd
        if idx2 is not None:
            Y[idx1, idx2] -= gd
            Y[idx2, idx1] -= gd
    if idx2 is not None:
        Y[idx2, idx2] += gd

    # Stamp Ieq into RHS
    if idx1 is not None:
        sources[idx1] -= ieq
    if idx2 is not None:
        sources[idx2] += ieq


# =============================================================================
# MOSFET helpers
# =============================================================================
def nmos_lim(vds_new, vds_old, critical_vds):
    """SPICE limiting for Vds to prevent region-jumping."""
    vds_old_safe = max(abs(vds_old), 1e-6)
    vds_new_safe = max(abs(vds_new), 1e-6)

    if vds_new > critical_vds and abs(vds_new - vds_old) > (2 * Vt):
        if abs(vds_old) > 1e-3:
            v_limit = vds_old + 2 * Vt * np.log(vds_new_safe / vds_old_safe)
        else:
            v_limit = vds_old + 2 * Vt
        return v_limit
    return vds_new


def nmos_region(vgs, vds, vth):
    """Finds MOSFET region: 0=cutoff, 1=triode, 2=saturation."""
    if vgs < vth:
        return 0
    elif vds < (vgs - vth):
        return 1
    else:
        return 2


def nmos_vgs_lim(vgs_new, vgs_old, vth):
    """Limits the change in Vgs to prevent numerical overflow."""
    step_limit = 0.5
    delta = vgs_new - vgs_old
    if abs(delta) > step_limit:
        vgs_new = vgs_old + np.sign(delta) * step_limit
    return vgs_new


def _extract_mosfet_params(comp):
    """Extract and validate MOSFET model parameters from a component dict."""
    params = comp["model_params"]
    inst_params = comp.get("inst_params", {})
    m_type = comp["model_type"]

    # Threshold (accept common synonyms)
    VTO = params.get("VTO", params.get("VT0", params.get("VTH", params.get("VTH0"))))
    if VTO is None:
        raise ValueError(
            "No threshold specified for MOSFET model (expected VTO/VT0/VTH/VTH0)."
        )
    VTO = float(VTO)

    # Geometry
    W = float(inst_params.get("W", 1.0))
    L = float(inst_params.get("L", 1.0))
    L = max(L, 1e-12)

    # Gain parameter
    if "KP" in params:
        KP = float(params["KP"])
    else:
        mu = params.get("MU", params.get("UO", params.get("U0")))
        cox = params.get("C_OX", params.get("COX"))
        if mu is None or cox is None:
            raise ValueError(
                "No KP or (MU/UO and C_OX/COX) specified for MOSFET model!"
            )
        KP = float(mu) * float(cox)

    Bn = (W / L) * KP
    return VTO, Bn, m_type


def _mos_level1_physics(vov, vds_k, Bn):
    """Compute Level-1 MOSFET Id, gm, gds from overdrive and Vds."""
    if vov <= 0.0:
        return 0.0, 0.0, 1e-12
    elif vds_k < vov:
        # Triode
        Id = Bn * (vov * vds_k - 0.5 * vds_k**2)
        gm = Bn * vds_k
        gds = Bn * (vov - vds_k) + 1e-12
        return Id, gm, gds
    else:
        # Saturation
        Id = 0.5 * Bn * (vov**2)
        gm = Bn * vov
        return Id, gm, 1e-12


def stamp_mosfet(Y, sources, comp, node_map, name, p_V_guess, V_guess):
    """Stamp a Level-1 MOSFET (NMOS or PMOS) using Newton linearization.

    NMOS: current Id flows from drain to source (conventional).
          Controlled by Vgs = Vg - Vs and Vds = Vd - Vs.

    PMOS: current flows from source to drain through channel.
          Computed using complementary voltages Vsg = Vs - Vg, Vsd = Vs - Vd,
          with |VTO| as threshold. Stamps are derived from the Vsg/Vsd
          dependence so signs differ from NMOS.
    """
    n_d, n_g, n_s, n_b = comp["n_d"], comp["n_g"], comp["n_s"], comp["n_b"]
    VTO, Bn, m_type = _extract_mosfet_params(comp)

    idx_d = get_idx(n_d, node_map)
    idx_g = get_idx(n_g, node_map)
    idx_s = get_idx(n_s, node_map)
    idx_b = get_idx(n_b, node_map)

    # Raw voltages at current and previous guesses
    v_d = V_guess[idx_d] if idx_d is not None else 0.0
    v_g = V_guess[idx_g] if idx_g is not None else 0.0
    v_s = V_guess[idx_s] if idx_s is not None else 0.0

    p_v_d = p_V_guess[idx_d] if idx_d is not None else 0.0
    p_v_g = p_V_guess[idx_g] if idx_g is not None else 0.0
    p_v_s = p_V_guess[idx_s] if idx_s is not None else 0.0

    if m_type == "NMOS":
        # --- NMOS: Id from D to S, controlled by Vgs, Vds ---
        vgs_k = nmos_vgs_lim(v_g - v_s, p_v_g - p_v_s, VTO)
        vds_k = nmos_lim(v_d - v_s, p_v_d - p_v_s, vgs_k - VTO)

        vov = vgs_k - VTO
        Id, gm, gds = _mos_level1_physics(vov, vds_k, Bn)

        # Stamp gm terms (Vgs-based: current into drain depends on Vg-Vs)
        if idx_d is not None:
            if idx_g is not None:
                Y[idx_d, idx_g] += gm
            if idx_s is not None:
                Y[idx_d, idx_s] -= gm
        if idx_s is not None:
            if idx_g is not None:
                Y[idx_s, idx_g] -= gm
            if idx_s is not None:
                Y[idx_s, idx_s] += gm

        # Stamp gds terms (Vds-based)
        if idx_d is not None:
            Y[idx_d, idx_d] += gds
            if idx_s is not None:
                Y[idx_d, idx_s] -= gds
                Y[idx_s, idx_d] -= gds
                Y[idx_s, idx_s] += gds

        # Equivalent current source
        Ieq = Id - (gm * vgs_k) - (gds * vds_k)
        if idx_d is not None:
            sources[idx_d] -= Ieq
        if idx_s is not None:
            sources[idx_s] += Ieq

    elif m_type == "PMOS":
        # --- PMOS: current Id flows S→D inside the device ---
        # Controlled by Vsg = Vs-Vg and Vsd = Vs-Vd, with |VTO| as threshold.
        VTO_abs = abs(VTO)

        vsg = v_s - v_g
        vsd = v_s - v_d
        p_vsg = p_v_s - p_v_g
        p_vsd = p_v_s - p_v_d

        vsg_k = nmos_vgs_lim(vsg, p_vsg, VTO_abs)
        vsd_k = nmos_lim(vsd, p_vsd, vsg_k - VTO_abs)

        vov = vsg_k - VTO_abs
        Id, gm, gds = _mos_level1_physics(vov, vsd_k, Bn)

        # MNA derivation (Y*V = b convention: Y*V = current leaving node):
        #
        # Current LEAVING drain through PMOS = -Id (current enters drain)
        #   = -(gm*(vs-vg) + gds*(vs-vd) + Ieq)
        #   = gm*vg - (gm+gds)*vs + gds*vd - Ieq
        # → Y[d,g]=+gm, Y[d,s]=-(gm+gds), Y[d,d]=+gds, b[d]+=Ieq
        #
        # Current LEAVING source through PMOS = +Id (current leaves source)
        #   = -gm*vg + (gm+gds)*vs - gds*vd + Ieq
        # → Y[s,g]=-gm, Y[s,s]=+(gm+gds), Y[s,d]=-gds, b[s]-=Ieq
        #
        # Note: Y stamp pattern is identical to NMOS! Only Ieq direction
        # differs (opposite sign on sources[]).

        # Stamp gm terms (same pattern as NMOS)
        if idx_d is not None:
            if idx_g is not None:
                Y[idx_d, idx_g] += gm
            if idx_s is not None:
                Y[idx_d, idx_s] -= gm
        if idx_s is not None:
            if idx_g is not None:
                Y[idx_s, idx_g] -= gm
            Y[idx_s, idx_s] += gm

        # Stamp gds terms (same pattern as NMOS)
        if idx_d is not None:
            Y[idx_d, idx_d] += gds
            if idx_s is not None:
                Y[idx_d, idx_s] -= gds
                Y[idx_s, idx_d] -= gds
                Y[idx_s, idx_s] += gds

        # Equivalent current source (OPPOSITE sign from NMOS)
        Ieq = Id - (gm * vsg_k) - (gds * vsd_k)
        if idx_d is not None:
            sources[idx_d] += Ieq   # current enters drain
        if idx_s is not None:
            sources[idx_s] -= Ieq   # current leaves source

    else:
        raise RuntimeError(f"Unrecognized MOSFET type: {m_type}")


def stamp_opamp(Y, sources, comp, node_map, name, p_V_guess=None, V_guess=None):
    """
    Ideal opamp equation (VCVS-like):
        V(n1) - V(n2) = A * (V(n3) - V(n4))

    Topology for V(n1)-V(n2) is already stamped by stamp_mna_connection.
    Gain ramping: Aeff = comp["value"] * comp.get("_gain_scale", 1.0)
    """
    A = float(comp.get("value", 1e5))
    k = float(comp.get("_gain_scale", 1.0))
    Aeff = A * k

    idx = node_map[name]  # branch equation row

    n3 = comp.get("n3", 0)
    n4 = comp.get("n4", 0)

    ip = get_idx(n3, node_map)
    im = get_idx(n4, node_map)

    # Row equation: V(n1)-V(n2) - A*(V(n3)-V(n4)) = 0
    if ip is not None:
        Y[idx, ip] -= Aeff
    if im is not None:
        Y[idx, im] += Aeff


# =============================================================================
# CONTROLLED SOURCES (E, F, H)
# =============================================================================
#
# SPICE controlled source summary:
#   G = VCCS  (already implemented above)     — stamps into Y directly
#   E = VCVS  — needs MNA branch current      — V_out = A * V_sense
#   F = CCCS  — needs MNA branch current       — I_out = A * I_sense
#   H = CCVS  — needs MNA branch current       — V_out = A * I_sense
#
# E, H need their own branch equation (like voltage sources).
# F needs its own branch to measure the controlling current, and stamps
#   the dependent current into the output nodes.
#

def stamp_vcvs(Y, sources, comp, node_map, name):
    """
    Stamp a Voltage-Controlled Voltage Source (E) using MNA.

    Format: Ename n+ n- nc+ nc- gain
        V(n+) - V(n-) = gain * (V(nc+) - V(nc-))

    MNA adds a branch current variable I_E. Topology (1/-1 stamps for
    n+/n- ↔ I_E) is already done by stamp_mna_connection. This function
    stamps the gain-dependent terms into the branch equation row.

    Branch equation row:
        V(n+) - V(n-) - gain*(V(nc+) - V(nc-)) = 0
    The [1, -1] for V(n+), V(n-) are from stamp_mna_connection.
    We add: -gain at nc+, +gain at nc-.
    """
    gain = comp["value"]
    idx = node_map[name]  # branch equation row

    n3 = comp.get("n3", 0)
    n4 = comp.get("n4", 0)
    ip = get_idx(n3, node_map)
    im = get_idx(n4, node_map)

    if ip is not None:
        Y[idx, ip] -= gain
    if im is not None:
        Y[idx, im] += gain


def stamp_cccs(Y, sources, comp, node_map, name):
    """
    Stamp a Current-Controlled Current Source (F) using MNA.

    Format: Fname N+ N- Vcontrol Gain
        I_out = gain * I_Vcontrol

    F does NOT get its own MNA branch. It simply stamps gain*I_ctrl
    into the output node KCL equations.
    """
    gain = comp["value"]

    # Controlling voltage source branch index
    v_ctrl = comp.get("v_control", "")
    if v_ctrl not in node_map:
        raise ValueError(
            f"CCCS '{name}': controlling source '{v_ctrl}' not found in node_map. "
            f"It must be a voltage source (V...) or inductor (L...)."
        )
    idx_ctrl = node_map[v_ctrl]

    n1, n2 = comp["n1"], comp["n2"]
    i = get_idx(n1, node_map)
    j = get_idx(n2, node_map)

    # SPICE convention: F sources current gain*I_ctrl from n- to n+
    # (current enters n+, leaves n-). In MNA, positive branch current
    # flows from + to - through the source. So positive I_ctrl means
    # current enters n+.
    # MNA Y*V = b convention: Y[i,k] term adds current LEAVING node i.
    # Current entering n+ means current leaving n+ is negative:
    if i is not None:
        Y[i, idx_ctrl] -= gain
    if j is not None:
        Y[j, idx_ctrl] += gain


def stamp_ccvs(Y, sources, comp, node_map, name):
    """
    Stamp a Current-Controlled Voltage Source (H) using MNA.

    Format: Hname n+ n- Vcontrol gain
        V(n+) - V(n-) = gain * I_Vcontrol

    MNA: H gets its own branch variable I_H.
    - Topology: stamp_mna_connection handles V(n+)-V(n-) ↔ I_H
    - Branch equation: V(n+) - V(n-) - gain*I_ctrl = 0
      → stamp -gain at (branch_row, idx_ctrl)
    """
    gain = comp["value"]
    idx_h = node_map[name]  # H's branch equation row

    v_ctrl = comp.get("v_control", "")
    if v_ctrl not in node_map:
        raise ValueError(
            f"CCVS '{name}': controlling source '{v_ctrl}' not found in node_map."
        )
    idx_ctrl = node_map[v_ctrl]

    # Branch equation row: V(n+) - V(n-) - gain*I_ctrl = 0
    Y[idx_h, idx_ctrl] -= gain


# =============================================================================
# BJT (Q) — Ebers-Moll / Gummel-Poon Level 1
# =============================================================================
#
# NPN BJT (Qname NC NB NE model):
#   - Base-Emitter junction: diode with Is, Nf
#   - Base-Collector junction: diode with Is/BF, Nr (typically reverse-biased)
#   - Collector current: Ic = BF*Ibe - Ibc  (forward active)
#   - Newton companion model: linearize both junctions, stamp gm + go + gpi
#
# The Ebers-Moll transport model:
#   If = IS * (exp(Vbe/NF/Vt) - 1)     forward transport current
#   Ir = IS * (exp(Vbc/NR/Vt) - 1)     reverse transport current
#   Ic = If - Ir                         collector current
#   Ib = If/BF + Ir/BR                  base current
#   Ie = -(Ic + Ib)                      emitter current (KCL)
#

def _bjt_junction(vj, Is, N_coeff, v_prev):
    """Compute junction current, conductance, and limited voltage."""
    Vte = max(N_coeff * Vt, 1e-12)
    vj_lim = pnjlim(vj, v_prev, 1.0)

    arg = np.clip(vj_lim / Vte, -50.0, 50.0)
    exp_v = np.exp(arg)
    I_junc = Is * (exp_v - 1.0)
    g_junc = (Is / Vte) * exp_v

    return I_junc, g_junc, vj_lim


def stamp_bjt(Y, sources, comp, node_map, name, p_V_guess, V_guess):
    """
    Stamp an NPN or PNP BJT using the Ebers-Moll transport model.

    Component data:
        n_c, n_b, n_e: collector, base, emitter nodes
        model_type: "NPN" or "PNP"
        model_params: IS, BF, BR, NF, NR, VAF, VAR, RB, RC, RE, ...

    For PNP: all terminal voltages are negated internally, stamps are
    sign-flipped so the same physics applies.

    Newton linearization at operating point (Vbe_k, Vbc_k):
        gbe = dIf/dVbe / BF + small leakage
        gbc = dIr/dVbc / BR + small leakage
        gm  = dIf/dVbe                  (transconductance)
        go  = dIr/dVbc                  (output conductance, reverse)

    MNA stamp (NPN, current INTO each terminal from device):
        I_c = +gm*vbe - go*vbc + Ic_eq
        I_b = +gbe*vbe + gbc*vbc + Ib_eq
        I_e = -(I_c + I_b)  (enforced by KCL, stamps are negative sum)
    """
    n_c = comp["n_c"]
    n_b = comp["n_b"]
    n_e = comp["n_e"]
    m_type = comp.get("model_type", "NPN")

    idx_c = get_idx(n_c, node_map)
    idx_b = get_idx(n_b, node_map)
    idx_e = get_idx(n_e, node_map)

    params = comp.get("model_params", {})
    IS = float(params.get("IS", 1e-14))
    BF = float(params.get("BF", 100.0))
    BR = float(params.get("BR", 1.0))
    NF = float(params.get("NF", 1.0))
    NR = float(params.get("NR", 1.0))
    VAF = float(params.get("VAF", 0.0))  # Forward Early voltage
    VAR = float(params.get("VAR", 0.0))  # Reverse Early voltage

    # Terminal voltages
    vc = V_guess[idx_c] if idx_c is not None else 0.0
    vb = V_guess[idx_b] if idx_b is not None else 0.0
    ve = V_guess[idx_e] if idx_e is not None else 0.0

    p_vc = p_V_guess[idx_c] if idx_c is not None else 0.0
    p_vb = p_V_guess[idx_b] if idx_b is not None else 0.0
    p_ve = p_V_guess[idx_e] if idx_e is not None else 0.0

    # PNP: negate all voltages (compute as NPN, then flip current signs)
    sign = 1.0
    if m_type == "PNP":
        sign = -1.0
        vc, vb, ve = -vc, -vb, -ve
        p_vc, p_vb, p_ve = -p_vc, -p_vb, -p_ve

    vbe = vb - ve
    vbc = vb - vc
    p_vbe = p_vb - p_ve
    p_vbc = p_vb - p_vc

    # Forward and reverse junction currents
    If, gm_f, vbe_lim = _bjt_junction(vbe, IS, NF, p_vbe)
    Ir, gm_r, vbc_lim = _bjt_junction(IS / max(BR, 1e-6), NR, vbc, p_vbc)
    # Fix: _bjt_junction takes (vj, Is, N, v_prev), re-call correctly:
    If, gm_f, vbe_lim = _bjt_junction(vbe, IS, NF, p_vbe)
    Ir, gm_r, vbc_lim = _bjt_junction(vbc, IS, NR, p_vbc)

    # Early effect (output resistance)
    if VAF > 0:
        early_fwd = 1.0 + max(vbc, -0.9 * VAF) / VAF  # avoid negative
        early_fwd = max(early_fwd, 0.1)
    else:
        early_fwd = 1.0

    if VAR > 0:
        early_rev = 1.0 + max(vbe, -0.9 * VAR) / VAR
        early_rev = max(early_rev, 0.1)
    else:
        early_rev = 1.0

    # Transport currents with Early effect
    Ic_transport = If * early_fwd - Ir * early_rev
    Ib_transport = If / BF + Ir / BR

    # Linearization conductances
    # gm = dIc/dVbe (transconductance)
    gm = gm_f * early_fwd
    # go = -dIc/dVbc (output conductance, how Ic changes with Vce via Vbc)
    go = gm_r * early_rev
    if VAF > 0:
        go += If / VAF  # Early effect adds Ic/VAF

    # gpi = dIb/dVbe (base input conductance)
    gpi = gm_f / BF
    # gmu = dIb/dVbc (base-collector feedback conductance)
    gmu = gm_r / BR

    # Equivalent currents (Norton companion)
    Ic_eq = Ic_transport - gm * vbe_lim + go * vbc_lim
    Ib_eq = Ib_transport - gpi * vbe_lim - gmu * vbc_lim

    # For PNP: the physical currents flow in the opposite direction
    # but we computed in NPN frame. Multiply Ieq by sign.
    Ic_eq *= sign
    Ib_eq *= sign

    # =========================================================
    # STAMP into Y matrix
    # =========================================================
    # We stamp in terms of Vbe = Vb-Ve and Vbc = Vb-Vc.
    # For PNP, gm/go/gpi/gmu are the same magnitude (computed in
    # NPN frame), but the controlling voltages are Veb and Vcb,
    # which reverses the sign of the off-diagonal stamps.
    # This is equivalent to keeping Y stamps identical and
    # letting sign handle Ieq.

    # --- Collector node KCL: Ic = gm*(Vb-Ve) - go*(Vb-Vc) + Ic_eq ---
    # = gm*Vb - gm*Ve - go*Vb + go*Vc + Ic_eq
    # = (gm-go)*Vb - gm*Ve + go*Vc + Ic_eq
    if idx_c is not None:
        Y[idx_c, idx_c] += go * sign
        if idx_b is not None:
            Y[idx_c, idx_b] += (gm - go) * sign
        if idx_e is not None:
            Y[idx_c, idx_e] -= gm * sign
        sources[idx_c] -= Ic_eq

    # --- Base node KCL: Ib = gpi*(Vb-Ve) + gmu*(Vb-Vc) + Ib_eq ---
    # = (gpi+gmu)*Vb - gpi*Ve - gmu*Vc + Ib_eq
    if idx_b is not None:
        Y[idx_b, idx_b] += (gpi + gmu) * sign
        if idx_e is not None:
            Y[idx_b, idx_e] -= gpi * sign
        if idx_c is not None:
            Y[idx_b, idx_c] -= gmu * sign
        sources[idx_b] -= Ib_eq

    # --- Emitter node KCL: Ie = -(Ic + Ib) by KCL ---
    # = -(gm-go+gpi+gmu)*Vb + (gm+gpi)*Ve + (go+gmu)*Vc - (Ic_eq+Ib_eq)
    Ie_eq = -(Ic_eq + Ib_eq)
    if idx_e is not None:
        Y[idx_e, idx_e] += (gm + gpi) * sign
        if idx_b is not None:
            Y[idx_e, idx_b] -= (gm - go + gpi + gmu) * sign
        if idx_c is not None:
            Y[idx_e, idx_c] += (go + gmu) * sign
        # No separate Ie_eq stamp needed — KCL is automatically
        # satisfied if C and B are correct. But for numerical
        # robustness, stamp it explicitly:
        sources[idx_e] -= Ie_eq
