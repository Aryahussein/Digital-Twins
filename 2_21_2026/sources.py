import numpy as np
import copy

def evaluate_time_source(source_dict, t):
    """Evaluate a single time-dependent source at time t."""
    stype = source_dict["type"]
    # print(source_dict)

    if stype == "PULSE":
        V1, V2 = source_dict["V1"], source_dict["V2"]
        TD, TR, TF = source_dict["TD"], source_dict["TR"], source_dict["TF"]
        PW, PER = source_dict["PW"], source_dict["PER"]

        if t < TD: return V1
        
        # Guard against PER=0 (single pulse, no repeat)
        if PER <= 0:
            t_rel = t - TD
        else:
            t_rel = (t - TD) % PER

        # Guard against TR=0 and TF=0 (instantaneous transitions)
        if TR <= 0:
            v = V2 if t_rel < 1e-30 else V2  # instant rise
            rise_done = 0.0
        else:
            rise_done = TR
            if t_rel < TR:
                return V1 + (V2 - V1) * (t_rel / TR)

        if t_rel < rise_done + PW:
            return V2
        
        if TF <= 0:
            return V1  # instant fall
        
        if t_rel < rise_done + PW + TF:
            return V2 - (V2 - V1) * ((t_rel - rise_done - PW) / TF)
        
        return V1

    elif stype == "SIN":
        return (source_dict["VOFF"] + source_dict["VAMP"] * np.sin(2 * np.pi * source_dict["FREQ"] * t + source_dict["PHASE"]))

    elif stype == "COS":
        return (source_dict["VOFF"] + source_dict["VAMP"] * np.cos(2 * np.pi * source_dict["FREQ"] * t + source_dict["PHASE"]))
    else:
        raise ValueError(f"Unknown source type: {stype}")

def evaluate_all_time_sources(components, t):
    """Return a NEW components dictionary with evaluated DC values."""
    evaluated_components = copy.deepcopy(components)
    for name, comp in evaluated_components.items():
        if "source" in comp:
            value_at_t = evaluate_time_source(comp["source"], t)
            # print("voltage at t=", t, "=", value_at_t)
            comp.pop("source")
            comp["value"] = value_at_t
    return evaluated_components
