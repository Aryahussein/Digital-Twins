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
        # print(t, TD, TR, TF, PW, PER)

        if t < TD: return V1
        t_mod = (t - TD) % PER
        # print(f"t_mod = {t_mod}")

        if t_mod < TR:
            v = V1 + (V2 - V1) * (t_mod / TR)
            # print(f"tmod < TR; v = {v}")
        elif t_mod < TR + PW:
            v = V2
            # print(f"tmod < TR + PW; v = {v}")
        elif t_mod < TR + PW + TF:
            v = V2 - (V2 - V1) * ((t_mod - TR - PW) / TF)
            # print(f"tmod < TR + PW + TF; v = {v}")
        else:
            v = V1
            # print(f"else; v = {v}")

        return v

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
