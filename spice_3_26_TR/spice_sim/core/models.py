"""
Device Physics Models Module.

This module provides the mathematical evaluation functions for nonlinear
semiconductor devices. Each function returns the large-signal operating point
values, small-signal linearization parameters, and sensitivity gradients
needed by the Newton-Raphson solver and Adjoint engine.
"""

import numpy as np

# Maximum exponent argument to prevent overflow in exp().
# Standard SPICE implementations use linear extrapolation beyond this threshold.
_MAX_EXP_ARG = 40.0


def evaluate_diode(vd, Is, Vt):
    """Evaluates the Shockley diode model with overflow-safe exponential.

    For large forward bias (vd/Vt > threshold), the function switches to a
    linear extrapolation that matches the value and slope at the threshold.
    This prevents np.exp overflow during early Newton-Raphson iterations.

    Args:
        vd (float): Voltage across the diode (Anode - Cathode).
        Is (float): Saturation current (e.g., 1e-14).
        Vt (float): Thermal voltage (e.g., 0.02585).
        
    Returns:
        dict: Large-signal current (I_D), small-signal conductance (gd), 
              and the sensitivity gradient (dId_dIs).
    """
    x = vd / Vt

    if x > _MAX_EXP_ARG:
        # Linear extrapolation beyond the safe threshold
        exp_at_limit = np.exp(_MAX_EXP_ARG)
        slope = (Is / Vt) * exp_at_limit  # gd at limit
        id_at_limit = Is * (exp_at_limit - 1.0)

        id_val = id_at_limit + slope * (vd - _MAX_EXP_ARG * Vt)
        gd = slope
        did_dis = exp_at_limit - 1.0 + (exp_at_limit / Vt) * (vd - _MAX_EXP_ARG * Vt)
    elif x < -_MAX_EXP_ARG:
        # Deep reverse bias: current ≈ -Is, conductance ≈ 0
        id_val = -Is
        gd = 1e-15  # Small floor to avoid singular matrix
        did_dis = -1.0
    else:
        # Standard Shockley Equation: Id = Is * (exp(Vd/Vt) - 1)
        exp_term = np.exp(x)
        id_val = Is * (exp_term - 1.0)
        gd = (Is / Vt) * exp_term
        did_dis = exp_term - 1.0
    
    return {
        "I_D": id_val,
        "gd": gd,
        "dId_dIs": did_dis
    }


def evaluate_mosfet_level1(vgs, vds, VTO, Bn):
    """Evaluates the Level 1 MOSFET model (NMOS equations).

    This function implements the Shichman-Hodges model. It is used for both 
    NMOS and PMOS transistors — the calling code handles the polarity mapping 
    (VGS ↔ VSG, VDS ↔ VSD) before invoking this function.

    Args:
        vgs (float): Gate-to-Source voltage (polarity-adjusted).
        vds (float): Drain-to-Source voltage (polarity-adjusted).
        VTO (float): Threshold voltage (absolute value).
        Bn (float): Process transconductance parameter (W/L * KP).
        
    Returns:
        dict: Operating point values (I_D, gm, gds) and sensitivity
              gradients (dId_dBn, dId_dVTO).
    """
    vov = vgs - VTO  # Overdrive voltage
    
    Id, gm, gds = 0.0, 0.0, 1e-12
    dId_dBn = 0.0
    dId_dVTO = 0.0

    if vov <= 0:
        # --- CUTOFF ---
        pass
    elif vds < vov:
        # --- TRIODE (Linear Region) ---
        Id = Bn * (vov * vds - 0.5 * vds**2)
        gm = Bn * vds
        gds = Bn * (vov - vds) + 1e-12
        
        dId_dBn = (vov * vds - 0.5 * vds**2)
        dId_dVTO = -Bn * vds
    else:
        # --- SATURATION (Active Region) ---
        Id = 0.5 * Bn * (vov**2)
        gm = Bn * vov
        gds = 1e-12
        
        dId_dBn = 0.5 * (vov**2)
        dId_dVTO = -Bn * vov
        
    return {
        "I_D": Id, 
        "gm": gm, 
        "gds": gds, 
        "dId_dBn": dId_dBn, 
        "dId_dVTO": dId_dVTO
    }
