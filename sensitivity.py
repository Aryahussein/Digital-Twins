import logging
import numpy as np
from solver import solve_adjoint
from constants import Vt
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


# -----------------------------
# Helpers for nonlinear devices
# -----------------------------


def _get_node_v(VI, idx):
    return VI[idx] if idx is not None else 0.0


def _diode_terminal_current_from_params(vd, params, Vt):
    """
    Compute terminal diode current Id flowing from n1->n2 given terminal
    voltage vd = v1 - v2. Uses same physics assumptions as stamp_diode.
    """
    Is = float(params.get("IS", 1e-14))
    N = float(params.get("N", 1.0))
    RS = float(params.get("RS", 0.0))
    BV = float(params.get("BV", 0.0))
    IBV = float(params.get("IBV", 1e-3))
    NBV = float(params.get("NBV", 1.0))

    Vte = max(N * Vt, 1e-12)
    Vtb = max(NBV * Vt, 1e-12)

    def I_of_vj(vj):
        arg_f = np.clip(vj / Vte, -50.0, 50.0)
        ef = np.exp(arg_f)
        Ifwd = Is * (ef - 1.0)

        Ibr = 0.0
        if BV > 0.0 and vj < -BV:
            x = -(vj + BV)
            arg_b = np.clip(x / Vtb, 0.0, 50.0)
            eb = np.exp(arg_b)
            Ibr = -IBV * eb

        return Ifwd + Ibr

    # Implicit solve for junction voltage if RS > 0
    vj = vd
    if RS > 0.0:
        for _ in range(10):
            Ij = I_of_vj(vj)
            vj_new = vd - RS * Ij
            vj = 0.6 * vj + 0.4 * vj_new

    return I_of_vj(vj)


def _finite_diff(fun, x0, rel=1e-6, abs_step=1e-12):
    """Central finite difference derivative for a scalar function."""
    dx = rel * max(1.0, abs(float(x0))) + abs_step
    fp = fun(x0 + dx)
    fm = fun(x0 - dx)
    return (fp - fm) / (2.0 * dx)


def _mos_current_level1(VI, node_map, comp):
    """
    Compute MOS terminal Id (current into drain) using the same simple
    Level-1 model as stamp_mosfet. Used only for sensitivity derivatives.
    Supports both NMOS and PMOS.
    """
    nd, ng, ns = comp["n_d"], comp["n_g"], comp["n_s"]
    idx_d, idx_g, idx_s = node_map.get(nd), node_map.get(ng), node_map.get(ns)

    vd = _get_node_v(VI, idx_d)
    vg = _get_node_v(VI, idx_g)
    vs = _get_node_v(VI, idx_s)

    params = comp.get("model_params", {})
    inst = comp.get("inst_params", {})
    m_type = comp.get("model_type", "NMOS")

    VTO = params.get(
        "VTO", params.get("VT0", params.get("VTH", params.get("VTH0", 0.7)))
    )
    VTO = float(VTO)

    W = float(inst.get("W", 1.0))
    L = float(inst.get("L", 1.0))
    L = max(L, 1e-12)

    if "KP" in params:
        KP = float(params["KP"])
    else:
        mu = params.get("MU", params.get("UO", params.get("U0", 0.0)))
        cox = params.get("C_OX", params.get("COX", 0.0))
        KP = float(mu) * float(cox)

    Bn = (W / L) * KP

    if m_type == "PMOS":
        # Use complementary voltages
        vov = (vs - vg) - abs(VTO)
        vds_eff = vs - vd
    else:
        # NMOS
        vov = (vg - vs) - VTO
        vds_eff = vd - vs

    if vov <= 0.0:
        return 0.0

    if vds_eff < vov:
        return Bn * (vov * vds_eff - 0.5 * vds_eff**2)
    else:
        return 0.5 * Bn * (vov**2)


def _bjt_collector_current(VI, node_map, comp):
    """
    Compute BJT collector current Ic using the Ebers-Moll transport model.
    Used only for sensitivity finite differences.
    """
    nc, nb, ne = comp["n_c"], comp["n_b"], comp["n_e"]
    idx_c = node_map.get(nc)
    idx_b = node_map.get(nb)
    idx_e = node_map.get(ne)

    vc = _get_node_v(VI, idx_c)
    vb = _get_node_v(VI, idx_b)
    ve = _get_node_v(VI, idx_e)

    m_type = comp.get("model_type", "NPN")
    if m_type == "PNP":
        vc, vb, ve = -vc, -vb, -ve

    params = comp.get("model_params", {})
    IS = float(params.get("IS", 1e-14))
    BF = float(params.get("BF", 100.0))
    BR = float(params.get("BR", 1.0))
    NF = float(params.get("NF", 1.0))
    NR = float(params.get("NR", 1.0))
    VAF = float(params.get("VAF", 0.0))

    vbe = vb - ve
    vbc = vb - vc

    Vte_f = max(NF * Vt, 1e-12)
    Vte_r = max(NR * Vt, 1e-12)

    If = IS * (np.exp(np.clip(vbe / Vte_f, -50, 50)) - 1.0)
    Ir = IS * (np.exp(np.clip(vbc / Vte_r, -50, 50)) - 1.0)

    early = 1.0
    if VAF > 0:
        early = 1.0 + max(vbc, -0.9 * VAF) / VAF
        early = max(early, 0.1)

    return If * early - Ir


def _list_sensitivity_keys(components):
    """
    Returns the list of keys that may appear in sensitivity outputs,
    including nonlinear device parameter keys like D1:IS and M1:VTO.
    """
    keys = list(components.keys())
    for name, comp in components.items():
        if name.startswith("D"):
            keys += [
                f"{name}:IS",
                f"{name}:N",
                f"{name}:RS",
                f"{name}:BV",
                f"{name}:IBV",
                f"{name}:NBV",
            ]
        elif name.startswith("M"):
            keys += [f"{name}:VTO", f"{name}:KP", f"{name}:MU", f"{name}:COX"]
        elif name.startswith("O"):
            keys += [f"{name}:A"]
        elif name.startswith("Q"):
            keys += [
                f"{name}:IS",
                f"{name}:BF",
                f"{name}:BR",
                f"{name}:NF",
                f"{name}:NR",
                f"{name}:VAF",
            ]

    # Deduplicate preserving order
    seen = set()
    out = []
    for k in keys:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


# -----------------------------
# Main sensitivity routines
# -----------------------------


def get_all_sensitivities(
    components, VI, PsiPhi, node_map, w=0.0, dt=None, Vt=0.02585
) -> Tuple[Dict[str, complex], Dict[str, Dict[str, complex]]]:
    """
    Compute sensitivity of the output w.r.t. all component parameters
    using adjoint vector PsiPhi.

    Returns:
        sensitivities: flat dict {comp_or_param_key: scalar_sensitivity}
        sensitivities_alex: nested dict {comp_name: {param_name: sensitivity}}
    """
    sensitivities: Dict[str, complex] = {}
    sensitivities_alex: Dict[str, Dict[str, complex]] = {}

    for name, comp in components.items():
        n1, n2 = comp.get("n1", 0), comp.get("n2", 0)
        idx1, idx2 = node_map.get(n1), node_map.get(n2)

        VI_branch = _get_node_v(VI, idx1) - _get_node_v(VI, idx2)
        Psi_branch = _get_node_v(PsiPhi, idx1) - _get_node_v(PsiPhi, idx2)

        # ---------------- Linear: Resistor ----------------
        if name.startswith("R"):
            R = comp["value"]
            # dG/dR = d(1/R)/dR = -1/R^2
            sens_val = -(1.0 / (R**2)) * (VI_branch * Psi_branch)
            sensitivities[name] = sens_val
            sensitivities_alex[name] = {"R": sens_val}

        # ---------------- Linear: Capacitor ----------------
        elif name.startswith("C"):
            if dt is not None:
                sens_val = -(1.0 / dt) * (VI_branch * Psi_branch)
            else:
                sens_val = -1j * w * (VI_branch * Psi_branch)
            sensitivities[name] = sens_val
            sensitivities_alex[name] = {"C": sens_val}

        # ---------------- Linear: Inductor ----------------
        elif name.startswith("L"):
            l_curr_idx = node_map[name]
            i_L = VI[l_curr_idx]
            i_L_hat = PsiPhi[l_curr_idx]
            if dt is not None:
                sens_val = (1.0 / dt) * (i_L * i_L_hat)
            else:
                sens_val = 1j * w * (i_L * i_L_hat)
            sensitivities[name] = sens_val
            sensitivities_alex[name] = {"L": sens_val}

        # ---------------- Linear: VCCS ----------------
        elif name.startswith("G"):
            n3, n4 = comp.get("n3", 0), comp.get("n4", 0)
            idx3, idx4 = node_map.get(n3), node_map.get(n4)
            v_sense = _get_node_v(VI, idx3) - _get_node_v(VI, idx4)
            sens_val = -(Psi_branch * v_sense)
            sensitivities[name] = sens_val
            sensitivities_alex[name] = {"G": sens_val}

        # ---------------- Linear: Voltage source ----------------
        elif name.startswith("V"):
            v_branch_idx = node_map[name]
            sens_val = PsiPhi[v_branch_idx]
            sensitivities[name] = sens_val
            sensitivities_alex[name] = {"V": sens_val}

        # ---------------- Linear: Current source ----------------
        elif name.startswith("I"):
            sens_val = -Psi_branch
            sensitivities[name] = sens_val
            sensitivities_alex[name] = {"I": sens_val}

        # ---------------- Opamp gain ----------------
        elif name.startswith("O"):
            idx_o = node_map.get(name, None)
            if idx_o is None:
                continue

            n3, n4 = comp.get("n3", 0), comp.get("n4", 0)
            idx3, idx4 = node_map.get(n3), node_map.get(n4)

            vdiff = _get_node_v(VI, idx3) - _get_node_v(VI, idx4)
            psi_o = _get_node_v(PsiPhi, idx_o)
            k = float(comp.get("_gain_scale", 1.0))

            sens_val = k * psi_o * vdiff
            sensitivities[f"{name}:A"] = sens_val
            sensitivities_alex[name] = {"A": sens_val}

        # ---------------- Nonlinear: Diode ----------------
        elif name.startswith("D"):
            vd = VI_branch

            params = dict(comp.get("model_params", {}))
            if "IS" not in params and "value" in comp:
                params["IS"] = comp["value"]

            def Id_with_param(pname, pval, _params=params):
                p = dict(_params)
                p[pname] = pval
                return _diode_terminal_current_from_params(vd, p, Vt)

            # Build up the alex dict incrementally (FIX: was overwriting)
            alex_entry = {}

            # IS
            Is0 = float(params.get("IS", 1e-14))
            dId_dIs = _finite_diff(lambda x: Id_with_param("IS", x), Is0)
            sensitivities[f"{name}:IS"] = -Psi_branch * dId_dIs
            alex_entry["IS"] = -Psi_branch * dId_dIs

            # N
            N0 = float(params.get("N", 1.0))
            dId_dN = _finite_diff(
                lambda x: Id_with_param("N", x), N0, rel=1e-6, abs_step=1e-9
            )
            sensitivities[f"{name}:N"] = -Psi_branch * dId_dN
            alex_entry["N"] = -Psi_branch * dId_dN

            # RS
            RS0 = float(params.get("RS", 0.0))
            dId_dRS = _finite_diff(
                lambda x: Id_with_param("RS", max(0.0, x)),
                RS0,
                rel=1e-6,
                abs_step=1e-12,
            )
            sensitivities[f"{name}:RS"] = -Psi_branch * dId_dRS
            alex_entry["RS"] = -Psi_branch * dId_dRS

            # BV
            BV0 = float(params.get("BV", 0.0))
            dId_dBV = _finite_diff(
                lambda x: Id_with_param("BV", max(0.0, x)),
                BV0,
                rel=1e-6,
                abs_step=1e-6,
            )
            sensitivities[f"{name}:BV"] = -Psi_branch * dId_dBV
            alex_entry["BV"] = -Psi_branch * dId_dBV

            # IBV
            IBV0 = float(params.get("IBV", 1e-3))
            dId_dIBV = _finite_diff(
                lambda x: Id_with_param("IBV", max(0.0, x)),
                IBV0,
                rel=1e-6,
                abs_step=1e-12,
            )
            sensitivities[f"{name}:IBV"] = -Psi_branch * dId_dIBV
            alex_entry["IBV"] = -Psi_branch * dId_dIBV

            # NBV
            NBV0 = float(params.get("NBV", 1.0))
            dId_dNBV = _finite_diff(
                lambda x: Id_with_param("NBV", max(1e-6, x)),
                NBV0,
                rel=1e-6,
                abs_step=1e-9,
            )
            sensitivities[f"{name}:NBV"] = -Psi_branch * dId_dNBV
            alex_entry["NBV"] = -Psi_branch * dId_dNBV

            sensitivities_alex[name] = alex_entry

        # ---------------- Nonlinear: MOSFET ----------------
        elif name.startswith("M"):
            nd, ns = comp["n_d"], comp["n_s"]
            idx_d, idx_s = node_map.get(nd), node_map.get(ns)
            Psi_ds = _get_node_v(PsiPhi, idx_d) - _get_node_v(PsiPhi, idx_s)

            params = dict(comp.get("model_params", {}))

            def Id_mos_with_param(pname, pval, _params=params, _comp=comp):
                c2 = dict(_comp)
                c2["model_params"] = dict(_params)
                c2["model_params"][pname] = pval
                return _mos_current_level1(VI, node_map, c2)

            alex_entry = {}

            # VTO
            VTO0 = float(
                params.get(
                    "VTO",
                    params.get("VT0", params.get("VTH", params.get("VTH0", 0.7))),
                )
            )
            dId_dVTO = _finite_diff(
                lambda x: Id_mos_with_param("VTO", x), VTO0, rel=1e-6, abs_step=1e-6
            )
            sensitivities[f"{name}:VTO"] = -Psi_ds * dId_dVTO
            alex_entry["VTO"] = -Psi_ds * dId_dVTO

            # KP or MU/COX
            if "KP" in params:
                KP0 = float(params["KP"])
                dId_dKP = _finite_diff(
                    lambda x: Id_mos_with_param("KP", x),
                    KP0,
                    rel=1e-6,
                    abs_step=1e-12,
                )
                sensitivities[f"{name}:KP"] = -Psi_ds * dId_dKP
                alex_entry["KP"] = -Psi_ds * dId_dKP
            else:
                mu0 = float(
                    params.get("MU", params.get("UO", params.get("U0", 0.0)))
                )
                cox0 = float(params.get("C_OX", params.get("COX", 0.0)))

                dId_dMU = _finite_diff(
                    lambda x: Id_mos_with_param("MU", x),
                    mu0,
                    rel=1e-6,
                    abs_step=1e-12,
                )
                dId_dCOX = _finite_diff(
                    lambda x: Id_mos_with_param("C_OX", x),
                    cox0,
                    rel=1e-6,
                    abs_step=1e-12,
                )

                sensitivities[f"{name}:MU"] = -Psi_ds * dId_dMU
                alex_entry["MU"] = -Psi_ds * dId_dMU
                sensitivities[f"{name}:COX"] = -Psi_ds * dId_dCOX
                alex_entry["COX"] = -Psi_ds * dId_dCOX

            sensitivities_alex[name] = alex_entry

        # ---------------- Nonlinear: BJT ----------------
        elif name.startswith("Q"):
            nc, ne = comp["n_c"], comp["n_e"]
            idx_c, idx_e = node_map.get(nc), node_map.get(ne)
            # Adjoint transfer across collector-emitter
            Psi_ce = _get_node_v(PsiPhi, idx_c) - _get_node_v(PsiPhi, idx_e)

            params = dict(comp.get("model_params", {}))

            def Ic_bjt_with_param(pname, pval, _params=params, _comp=comp):
                c2 = dict(_comp)
                c2["model_params"] = dict(_params)
                c2["model_params"][pname] = pval
                return _bjt_collector_current(VI, node_map, c2)

            alex_entry = {}

            for pname, default in [
                ("IS", 1e-14),
                ("BF", 100.0),
                ("BR", 1.0),
                ("NF", 1.0),
                ("NR", 1.0),
                ("VAF", 0.0),
            ]:
                p0 = float(params.get(pname, default))
                if p0 == 0.0 and pname == "VAF":
                    continue  # skip VAF if not used
                dIc_dp = _finite_diff(
                    lambda x, _pn=pname: Ic_bjt_with_param(_pn, x),
                    p0,
                    rel=1e-6,
                    abs_step=1e-12 if pname == "IS" else 1e-6,
                )
                sensitivities[f"{name}:{pname}"] = -Psi_ce * dIc_dp
                alex_entry[pname] = -Psi_ce * dIc_dp

            sensitivities_alex[name] = alex_entry

    return sensitivities, sensitivities_alex


def compute_step_sensitivities(
    lu, VI, components, node_map, output_nodes=None, w=0.0, dt=None, Vt=0.02585
) -> Tuple[Dict[str, Dict[str, complex]], Dict[str, Dict[str, Dict[str, complex]]]]:
    """Solves the adjoint system and gathers sensitivities for requested output nodes."""
    if output_nodes is None:
        output_nodes = list(node_map.keys())

    step_sensitivities: Dict[str, Dict[str, complex]] = {}
    step_sensitivities_alex: Dict[str, Dict[str, Dict[str, complex]]] = {}

    for out_node in output_nodes:
        PsiPhi = solve_adjoint(lu, out_node, node_map)
        comp_sens, comp_sens_alex = get_all_sensitivities(
            components, VI, PsiPhi, node_map, w=w, dt=dt, Vt=Vt
        )
        step_sensitivities[out_node] = comp_sens
        step_sensitivities_alex[out_node] = comp_sens_alex

    return step_sensitivities, step_sensitivities_alex


def aggregate_sweep_sensitivities(
    components,
    node_map,
    analyses,
    output_nodes=None,
    raw_sensitivities=None,
    list_of_lus=None,
    VI_list=None,
    freq_list=None,
    dt=None,
    Vt=0.02585,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Dict[str, complex]]]]:
    """
    Aggregates multi-step simulation sensitivity data into arrays.

    FIX: Previously overwrote each step instead of accumulating. Now correctly
    appends each step's data into lists, then converts to numpy arrays.
    """
    logger.info("Starting sensitivity data aggregation...")
    target_nodes = output_nodes if output_nodes is not None else list(node_map.keys())

    all_keys = _list_sensitivity_keys(components)

    # Initialize with LISTS for accumulation (was np.array([]) causing overwrite)
    sensitivity_dict: Dict[str, Dict[str, list]] = {
        node: {k: [] for k in all_keys} for node in target_nodes
    }
    sensitivity_dict_alex: Dict[str, Dict[str, Dict[str, complex]]] = {}
    raw_sensitivities_alex: List[Dict[str, Dict[str, Dict[str, complex]]]] = []

    if raw_sensitivities is None:
        if not list_of_lus or VI_list is None:
            # Convert empty lists to arrays and return
            final = {
                node: {k: np.array([]) for k in all_keys} for node in target_nodes
            }
            return final, sensitivity_dict_alex

        logger.info("Computing sensitivities from stored LU matrices...")
        is_ac = ".AC" in analyses
        raw_sensitivities = []

        for i in range(len(list_of_lus)):
            if is_ac and freq_list is not None:
                w_step = 2 * np.pi * freq_list[i]
                step_sens, step_sens_alex = compute_step_sensitivities(
                    list_of_lus[i],
                    VI_list[i],
                    components,
                    node_map,
                    target_nodes,
                    w=w_step,
                    Vt=Vt,
                )
            else:
                step_sens, step_sens_alex = compute_step_sensitivities(
                    list_of_lus[i],
                    VI_list[i],
                    components,
                    node_map,
                    target_nodes,
                    dt=dt,
                    Vt=Vt,
                )
            raw_sensitivities.append(step_sens)
            raw_sensitivities_alex.append(step_sens_alex)

    # FIX: Accumulate all steps into lists, then convert to arrays
    for step_data in raw_sensitivities:
        for node in target_nodes:
            if node not in step_data:
                continue
            for k, v in step_data[node].items():
                if k in sensitivity_dict[node]:
                    sensitivity_dict[node][k].append(v)

    # Convert accumulated lists to numpy arrays
    for node in sensitivity_dict:
        for k in sensitivity_dict[node]:
            sensitivity_dict[node][k] = np.array(sensitivity_dict[node][k])

    # Aggregate alex dict (last step wins here — this is typically used for OP only)
    for step_data in raw_sensitivities_alex:
        for node in target_nodes:
            if node in step_data:
                sensitivity_dict_alex[node] = step_data[node]

    logger.info("Sensitivity aggregation complete.")
    return sensitivity_dict, sensitivity_dict_alex


def estimate_std_dev(sensitivities, components, percent_sigma=0.01):
    """
    Computes output standard deviation from parameter sensitivities.
    Supports nonlinear param keys like D1:IS, M1:VTO, O1:A, etc.
    """
    variance = 0.0
    for key, sens in sensitivities.items():
        if isinstance(sens, np.ndarray) and sens.size == 0:
            continue
        if np.isscalar(sens) and sens == 0:
            continue

        # Determine the nominal parameter value
        if ":" in key:
            dev, pname = key.split(":", 1)
            comp = components.get(dev, {})
            params = comp.get("model_params", {})

            if dev.startswith("D"):
                base = float(
                    params.get(
                        pname, params.get(pname.upper(), comp.get("value", 0.0))
                    )
                )
            elif dev.startswith("M"):
                if pname == "VTO":
                    base = float(
                        params.get(
                            "VTO",
                            params.get(
                                "VT0",
                                params.get("VTH", params.get("VTH0", 0.0)),
                            ),
                        )
                    )
                elif pname == "KP":
                    base = float(params.get("KP", 0.0))
                elif pname == "MU":
                    base = float(
                        params.get("MU", params.get("UO", params.get("U0", 0.0)))
                    )
                elif pname == "COX":
                    base = float(params.get("C_OX", params.get("COX", 0.0)))
                else:
                    base = 0.0
            elif dev.startswith("O"):
                if pname == "A":
                    base = float(comp.get("value", 0.0))
                else:
                    base = 0.0
            else:
                base = 0.0
        else:
            base = float(components.get(key, {}).get("value", 0.0))

        sigma_p = base * percent_sigma
        variance += np.abs(sens * sigma_p) ** 2

    return np.sqrt(variance)
