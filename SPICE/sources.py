import numpy as np
import copy

def evaluate_time_source(source_dict, t):
    """Evaluate a single time-dependent source at time t."""
    stype = source_dict["type"]

    if stype == "PULSE":
        V1, V2 = source_dict["V1"], source_dict["V2"]
        TD, TR, TF = source_dict["TD"], source_dict["TR"], source_dict["TF"]
        PW, PER = source_dict["PW"], source_dict["PER"]

        if t < TD: return V1
        t_mod = (t - TD) % PER

        if t_mod < TR:
            return V1 + (V2 - V1) * (t_mod / TR)
        elif t_mod < TR + PW:
            return V2
        elif t_mod < TR + PW + TF:
            return V2 - (V2 - V1) * ((t_mod - TR - PW) / TF)
        else:
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
            comp.pop("source")
            comp["value"] = value_at_t
    return evaluated_components