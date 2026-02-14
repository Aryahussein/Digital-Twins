import numpy as np
from txt2dictionary import parse_netlist
import copy
def evaluate_time_source(source_dict, t):
    """
    Evaluate a single time-dependent source at time t.
    """

    stype = source_dict["type"]

    if stype == "PULSE":
        V1 = source_dict["V1"]
        V2 = source_dict["V2"]
        TD = source_dict["TD"]
        TR = source_dict["TR"]
        TF = source_dict["TF"]
        PW = source_dict["PW"]
        PER = source_dict["PER"]

        if t < TD:
            return V1

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
        return (
            source_dict["VOFF"]
            + source_dict["VAMP"]
            * np.sin(2 * np.pi * source_dict["FREQ"] * t + source_dict["PHASE"])
        )

    elif stype == "COS":
        return (
            source_dict["VOFF"]
            + source_dict["VAMP"]
            * np.cos(2 * np.pi * source_dict["FREQ"] * t + source_dict["PHASE"])
        )

    else:
        raise ValueError(f"Unknown source type: {stype}")


def evaluate_all_time_sources(components, t):
    """
    Return a NEW components dictionary where all time-dependent
    sources are replaced with equivalent DC sources evaluated at time t.
    """

    # Deep copy so original dictionary remains unchanged
    evaluated_components = copy.deepcopy(components)

    for name, comp in evaluated_components.items():

        if "source" in comp:
            # Evaluate time-dependent source
            value_at_t = evaluate_time_source(comp["source"], t)

            # Replace with equivalent DC source
            comp.pop("source")
            comp["value"] = value_at_t

    return evaluated_components

if __name__ == '__main__':
    file = r"testfiles/transient_parser.txt"
    components = parse_netlist(file)
    print(components)
    t = 0.25e-3
    print(evaluate_all_time_sources(components, t))