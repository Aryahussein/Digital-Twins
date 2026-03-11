import numpy as np

def evaluate_diode(vd, Is, Vt):
    """
    Evaluates the Shockley diode model.
    
    Args:
        vd (float): Voltage across the diode (Anode - Cathode).
        Is (float): Saturation current (e.g., 1e-14).
        Vt (float): Thermal voltage (e.g., 0.02585).
        
    Returns:
        dict: Large-signal current (I_D), small-signal conductance (gd), 
              and the sensitivity gradient (dId_dIs).
    """
    # Standard Shockley Equation: Id = Is * (exp(Vd/Vt) - 1)
    exp_term = np.exp(vd / Vt)
    id_val = Is * (exp_term - 1.0)
    
    # Small-signal conductance: gd = d(Id)/d(Vd) = (Is/Vt) * exp(Vd/Vt)
    gd = (Is / Vt) * exp_term
    
    # Gradient for sensitivity: d(Id)/d(Is) = exp(Vd/Vt) - 1
    did_dis = exp_term - 1.0
    
    return {
        "I_D": id_val,
        "gd": gd,
        "dId_dIs": did_dis
    }


def evaluate_nmos(vgs, vds, VTO, Bn):
    """Evaluates the Level 1 NMOS model with sensitivity gradients."""
    vov = vgs - VTO  # Overdrive voltage
    
    # Initialize values
    Id, gm, gds = 0.0, 0.0, 1e-12
    dId_dBn = 0.0
    dId_dVTO = 0.0

    if vov <= 0:
        # --- CUTOFF ---
        # Id, gm, gds are already initialized to zero/near-zero
        pass
    elif vds < vov:
        # --- TRIODE (Linear Region) ---
        Id = Bn * (vov * vds - 0.5 * vds**2)
        gm = Bn * vds
        gds = Bn * (vov - vds) + 1e-12
        
        # Gradients for Sensitivity
        dId_dBn = (vov * vds - 0.5 * vds**2)
        dId_dVTO = -Bn * vds
    else:
        # --- SATURATION (Active Region) ---
        Id = 0.5 * Bn * (vov**2)
        gm = Bn * vov
        gds = 1e-12 # Ideal saturation has zero output conductance
        
        # Gradients for Sensitivity
        dId_dBn = 0.5 * (vov**2)
        dId_dVTO = -Bn * vov
        
    return {
        "I_D": Id, 
        "gm": gm, 
        "gds": gds, 
        "dId_dBn": dId_dBn, 
        "dId_dVTO": dId_dVTO
    }
