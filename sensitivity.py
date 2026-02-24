import numpy as np
from constants import Vt


def _get_node_voltage(VI, node, node_map):
    idx = node_map.get(node)
    if idx is None:
        return 0.0
    return VI[idx]


def _get_branch_voltage(VI, n1, n2, node_map):
    v1 = _get_node_voltage(VI, n1, node_map)
    v2 = _get_node_voltage(VI, n2, node_map)
    return v1 - v2


def _get_branch_adjoint(PsiPhi, n1, n2, node_map):
    p1 = _get_node_voltage(PsiPhi, n1, node_map)
    p2 = _get_node_voltage(PsiPhi, n2, node_map)
    return p1 - p2


def get_all_sensitivities(components, VI, PsiPhi, node_map, w=0.0):
    """
    Returns sensitivities of the chosen output (encoded in PsiPhi) w.r.t. component parameters.

    Linear elements (R, C, L, G): same formulas as before, but valid at nonlinear OP too,
    because PsiPhi is solved from the final Jacobian (operating point linearization).

    Nonlinear elements:
      - Diode D: sensitivity w.r.t Is (saturation current)
      - Opamp A: sensitivity w.r.t Vsat and k
    """
    sensitivities = {}

    for name, comp in components.items():
        ctype = str(comp.get("type", "")).upper()

        # -----------------------
        # R
        # -----------------------
        if ctype == "R":
            n1, n2 = comp.get("n1", 0), comp.get("n2", 0)
            R = comp["value"]
            VI_branch = _get_branch_voltage(VI, n1, n2, node_map)
            PsiPhi_branch = _get_branch_adjoint(PsiPhi, n1, n2, node_map)
            sensitivities[name] = (1.0 / (R**2)) * (VI_branch * PsiPhi_branch)

        # -----------------------
        # C (AC only)
        # -----------------------
        elif ctype == "C":
            n1, n2 = comp.get("n1", 0), comp.get("n2", 0)
            VI_branch = _get_branch_voltage(VI, n1, n2, node_map)
            PsiPhi_branch = _get_branch_adjoint(PsiPhi, n1, n2, node_map)
            sensitivities[name] = -1j * w * (VI_branch * PsiPhi_branch)

        # -----------------------
        # L (AC only)
        # -----------------------
        elif ctype == "L":
            l_curr_idx = node_map[name]
            i_L = VI[l_curr_idx]
            i_L_hat = PsiPhi[l_curr_idx]
            sensitivities[name] = 1j * w * (i_L * i_L_hat)

        # -----------------------
        # G (VCCS)
        # -----------------------
        elif ctype == "G":
            n1, n2 = comp.get("n1", 0), comp.get("n2", 0)
            n3, n4 = comp.get("n3", 0), comp.get("n4", 0)

            VI_sense = _get_branch_voltage(VI, n3, n4, node_map)
            PsiPhi_branch = _get_branch_adjoint(PsiPhi, n1, n2, node_map)

            sensitivities[name] = -(PsiPhi_branch * VI_sense)

        # -----------------------
        # D (diode): sensitivity w.r.t Is
        # -----------------------
        elif ctype == "D":
            n1, n2 = comp.get("n1", 0), comp.get("n2", 0)
            vd = _get_branch_voltage(VI, n1, n2, node_map)
            PsiPhi_branch = _get_branch_adjoint(PsiPhi, n1, n2, node_map)

            # exp(vd/Vt) can overflow; clamp exponent
            x = float(np.clip(vd / Vt, -100.0, 40.0))
            exp_term = np.exp(x)

            # For the MNA formulation used, sensitivity w.r.t Is simplifies to:
            # d(output)/dIs = - (exp(vd/Vt) - 1) * (PsiPhi_branch)
            sensitivities[name] = -(exp_term - 1.0) * PsiPhi_branch

        # -----------------------
        # A (op-amp): sensitivities w.r.t Vsat and k
        # -----------------------
        elif ctype == "A":
            # The op-amp contributes mainly through its constraint row at index node_map[name]
            a_idx = node_map[name]
            lam_a = PsiPhi[a_idx]

            out = comp["out"]
            vp = comp["vp"]
            vm = comp["vm"]
            Vsat = float(comp.get("Vsat", 1.0))
            k_val = float(comp.get("k", 1e3))

            dV = _get_branch_voltage(VI, vp, vm, node_map)
            t = np.tanh(k_val * dV)
            sech2 = 1.0 - t * t

            # F_a = V(out) - Vsat*tanh(k*dV) = 0
            # dy/dp = -lambda^T * dF/dp
            # dF/dVsat = -tanh(k*dV) -> sens = -lam_a * (-t) = lam_a*t
            # dF/dk    = -Vsat*sech2*dV -> sens = -lam_a * (-Vsat*sech2*dV) = lam_a*Vsat*sech2*dV
            sensitivities[name + ":Vsat"] = lam_a * t
            sensitivities[name + ":k"] = lam_a * Vsat * sech2 * dV

    return sensitivities


def compute_step_sensitivities(lu, VI, components, node_map, output_nodes=None, w=0.0):
    from solver import solve_adjoint

    if output_nodes is None:
        # Only node voltage keys (ints) are meaningful output nodes
        output_nodes = [k for k in node_map.keys() if isinstance(k, int)]

    step_sensitivities = {}
    for out_node in output_nodes:
        print(f"Solving adjoint at output node {out_node}...")
        PsiPhi = solve_adjoint(lu, out_node, node_map)

        comp_sens = get_all_sensitivities(components, VI, PsiPhi, node_map, w=w)
        step_sensitivities[out_node] = comp_sens

    return step_sensitivities


def aggregate_sweep_sensitivities(components, node_map, analyses,
                                  output_nodes=None, raw_sensitivities=None,
                                  list_of_lus=None, VI_list=None, freq_list=None):
    print("Starting sensitivity data aggregation...")
    target_nodes = output_nodes if output_nodes is not None else [k for k in node_map.keys() if isinstance(k, int)]

    sensitivity_dict = {
        node: {name: [] for name in components.keys()} for node in target_nodes
    }

    if not raw_sensitivities:
        if not list_of_lus or VI_list is None:
            return sensitivity_dict

        print("Computing sensitivities from stored LU matrices...")
        is_ac = ".AC" in analyses
        raw_sensitivities = []

        for i in range(len(list_of_lus)):
            w_step = 2 * np.pi * freq_list[i] if (is_ac and freq_list is not None) else 0.0

            step_sens = compute_step_sensitivities(
                list_of_lus[i], VI_list[i], components, node_map, target_nodes, w_step
            )
            raw_sensitivities.append(step_sens)

    for step_data in raw_sensitivities:
        for node in target_nodes:
            for comp_name, sens_value in step_data[node].items():
                # initialize missing keys (because we added A:Vsat etc)
                if comp_name not in sensitivity_dict[node]:
                    sensitivity_dict[node][comp_name] = []
                sensitivity_dict[node][comp_name].append(sens_value)

    for node in sensitivity_dict:
        for comp in sensitivity_dict[node]:
            sensitivity_dict[node][comp] = np.array(sensitivity_dict[node][comp])

    print("Sensitivity aggregation complete.")
    return sensitivity_dict


def estimate_std_dev(sensitivities, components, percent_sigma=0.01):
    variance = 0.0
    for name, sens in sensitivities.items():
        if hasattr(sens, "__len__") and len(sens) == 0:
            continue

        # For A:Vsat / A:k, there is no "value" field in components dict
        base_name = name.split(":")[0]
        comp_val = components.get(base_name, {}).get("value", 0.0)
        sigma_p = comp_val * percent_sigma

        variance += np.abs(sens * sigma_p)**2

    return np.sqrt(variance)