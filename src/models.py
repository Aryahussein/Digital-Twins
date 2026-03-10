# models.py
import numpy as np

def evaluate_diode(vd, Is, Vt):
    """
    Evaluates the physical diode model.
    Returns the large-signal current (I_D) and small-signal conductance (gd).
    """
    exp_term = np.exp(vd / Vt)
    id_k = Is * (exp_term - 1.0)
    gd = (Is / Vt) * exp_term
    
    # We also return dId_dIs here so your sensitivity analysis can use the exact same file!
    dId_dIs = exp_term - 1.0 
    
    return {"I_D": id_k, "gd": gd, "dId_dIs": dId_dIs}

def evaluate_nmos(vgs, vds, VTO, Bn):
    """
    Evaluates the Level 1 NMOS model.
    Returns the large signal current (I_D), transconductance (gm), and output conductance (gds).
    """
    vov = vgs - VTO  # Overdrive voltage

    if vov <= 0:
        # CUTOFF
        Id, gm, gds = 0.0, 0.0, 1e-12
    elif vds < vov:
        # TRIODE
        Id = Bn * (vov * vds - 0.5 * vds**2)
        gm = Bn * vds
        gds = Bn * (vov - vds) + 1e-12
    else:
        # SATURATION
        Id = 0.5 * Bn * (vov**2)
        gm = Bn * vov
        gds = 1e-12
        
    return {"I_D": Id, "gm": gm, "gds": gds}
