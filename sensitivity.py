import numpy as np
from solver import solve_adjoint
from typing import Dict, List, Tuple

# -----------------------------
# Helpers for nonlinear devices
# -----------------------------


def _get_node_v(VI, idx):
    return VI[idx] if idx is not None else 0.0


def _diode_terminal_current_from_params(vd, params, Vt):
    """
    Compute terminal diode current Id flowing from n1->n2 given terminal voltage vd = v1 - v2.
    Uses same physics assumptions as your stamp_diode:
      - IS, N, RS
      - soft breakdown: BV, IBV, NBV
    Returns Id (terminal current).
    """
    # defaults
    Is = float(params.get("IS", 1e-14))
    N = float(params.get("N", 1.0))
    RS = float(params.get("RS", 0.0))

    BV = float(params.get("BV", 0.0))
    IBV = float(params.get("IBV", 1e-3))
    NBV = float(params.get("NBV", 1.0))

    Vte = max(N * Vt, 1e-12)
    Vtb = max(NBV * Vt, 1e-12)

    def I_of_vj(vj):
        # forward
        arg_f = np.clip(vj / Vte, -50.0, 50.0)
        ef = np.exp(arg_f)
        Ifwd = Is * (ef - 1.0)

        # soft breakdown
        Ibr = 0.0
        if BV > 0.0 and vj < -BV:
            x = -(vj + BV)  # >=0
            arg_b = np.clip(x / Vtb, 0.0, 50.0)
            eb = np.exp(arg_b)
            Ibr = -IBV * eb

        return Ifwd + Ibr

    # implicit solve for junction voltage if RS>0: vj = vd - RS*I(vj)
    vj = vd
    if RS > 0.0:
        for _ in range(10):
            Ij = I_of_vj(vj)
            vj_new = vd - RS * Ij
            vj = 0.6 * vj + 0.4 * vj_new

    return I_of_vj(vj)


def _finite_diff(fun, x0, rel=1e-6, abs_step=1e-12):
    """
    Central finite difference derivative for a scalar function.
    Step size: dx = rel*max(1,|x0|) + abs_step
    """
    dx = rel * max(1.0, abs(float(x0))) + abs_step
    fp = fun(x0 + dx)
    fm = fun(x0 - dx)
    return (fp - fm) / (2.0 * dx)


def _mos_current_level1(VI, node_map, comp):
    """
    Compute MOS terminal Id (drain current from d->s) using the SAME simple model logic
    as your stamp_mosfet (Level-1-ish).
    This is used only for local parameter derivatives for sensitivity.
    """
    # indices
    nd, ng, ns = comp["n_d"], comp["n_g"], comp["n_s"]
    idx_d, idx_g, idx_s = node_map.get(nd), node_map.get(ng), node_map.get(ns)

    vd = _get_node_v(VI, idx_d)
    vg = _get_node_v(VI, idx_g)
    vs = _get_node_v(VI, idx_s)

    vgs = vg - vs
    vds = vd - vs

    params = comp.get("model_params", {})
    inst = comp.get("inst_params", {})

    # Threshold synonyms
    VTO = params.get(
        "VTO", params.get("VT0", params.get("VTH", params.get("VTH0", 0.7)))
    )
    VTO = float(VTO)

    # Geometry
    W = float(inst.get("W", 1.0))
    L = float(inst.get("L", 1.0))
    L = max(L, 1e-12)

    # KP or MU*COX
    if "KP" in params:
        KP = float(params["KP"])
    else:
        mu = params.get("MU", params.get("UO", params.get("U0", 0.0)))
        cox = params.get("C_OX", params.get("COX", 0.0))
        KP = float(mu) * float(cox)

    Bn = (W / L) * KP

    vov = vgs - VTO
    if vov <= 0.0:
        return 0.0

    # region
    if vds < vov:
        # triode
        return Bn * (vov * vds - 0.5 * vds**2)
    else:
        # saturation
        return 0.5 * Bn * (vov**2)


def _list_sensitivity_keys(components):
    """
    Returns the list of keys that may appear in sensitivity outputs,
    including nonlinear device parameter keys like D1:IS and M1:VTO.
    """
    keys = list(components.keys())
    for name, comp in components.items():
        if name.startswith("D"):
            # include common diode params
            keys += [
                f"{name}:IS",
                f"{name}:N",
                f"{name}:RS",
                f"{name}:BV",
                f"{name}:IBV",
                f"{name}:NBV",
            ]
        elif name.startswith("M"):
            # include common MOS params
            keys += [f"{name}:VTO", f"{name}:KP", f"{name}:MU", f"{name}:COX"]
        elif name.startswith("O"):
            # include opamp gain
            keys += [f"{name}:A"]

    # de-dup but keep stable-ish order
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
    Compute sensitivity of the output w.r.t. all component parameters using adjoint PsiPhi.

    Supported linear element sensitivities:
      R, C, L, G, V, I

    Added (Option 3):
      Diode model param sensitivities per device instance:
        Dk:IS, Dk:N, Dk:RS, Dk:BV, Dk:IBV, Dk:NBV

      MOS model param sensitivities per device instance (simple level-1 current model):
        Mk:VTO and Mk:KP or Mk:MU/Mk:COX (depending on what's in the model)

      Opamp gain sensitivity:
        Ok:A
    """
    sensitivities: Dict[str, complex] = {}
    sensitivities_alex: Dict[str, Dict[str, complex]] = {}

    for name, comp in components.items():
        # Basic 2-terminal branch info (for R/C/D/etc.)
        n1, n2 = comp.get("n1", 0), comp.get("n2", 0)
        idx1, idx2 = node_map.get(n1), node_map.get(n2)

        VI_branch = _get_node_v(VI, idx1) - _get_node_v(VI, idx2)
        Psi_branch = _get_node_v(PsiPhi, idx1) - _get_node_v(PsiPhi, idx2)

        # ---------------- Linear elements ----------------
        if name.startswith("R"):
            R = comp["value"]
            sensitivities[name] = (1.0 / (R**2)) * (VI_branch * Psi_branch)
            sensitivities_alex[name] = {"R": (1.0 / (R**2)) * (VI_branch * Psi_branch)}

        elif name.startswith("C"):
            if dt is not None:
                sensitivities[name] = -(1.0 / dt) * (VI_branch * Psi_branch)
                sensitivities_alex[name] = {"C": -(1.0 / dt) * (VI_branch * Psi_branch)}

            else:
                sensitivities[name] = -1j * w * (VI_branch * Psi_branch)
                sensitivities_alex[name] = {"C": -1j * w * (VI_branch * Psi_branch)}

        elif name.startswith("L"):
            l_curr_idx = node_map[name]
            i_L = VI[l_curr_idx]
            i_L_hat = PsiPhi[l_curr_idx]
            if dt is not None:
                sensitivities[name] = (1.0 / dt) * (i_L * i_L_hat)
                sensitivities_alex[name] = {"L": (1.0 / dt) * (i_L * i_L_hat)}
            else:
                sensitivities[name] = 1j * w * (i_L * i_L_hat)
                sensitivities_alex[name] = {"L": 1j * w * (i_L * i_L_hat)}

        elif name.startswith("G"):
            n3, n4 = comp.get("n3", 0), comp.get("n4", 0)
            idx3, idx4 = node_map.get(n3), node_map.get(n4)
            v_sense = _get_node_v(VI, idx3) - _get_node_v(VI, idx4)
            sensitivities[name] = -(Psi_branch * v_sense)
            sensitivities_alex[name] = {"G": -(Psi_branch * v_sense)}

        elif name.startswith("V"):
            v_branch_idx = node_map[name]
            sensitivities[name] = PsiPhi[v_branch_idx]
            sensitivities_alex[name] = {"V": PsiPhi[v_branch_idx]}

        elif name.startswith("I"):
            sensitivities[name] = -Psi_branch
            sensitivities_alex[name] = {"I": -Psi_branch}

        # ---------------- Opamp gain ----------------
        elif name.startswith("O"):
            # mh fix the A to B
            # Sensitivity w.r.t opamp gain A in:
            #   V(n1)-V(n2) - Aeff*(V(n3)-V(n4)) = 0
            # with Aeff = k*A (gain ramping)
            idx_o = node_map.get(name, None)
            if idx_o is None:
                continue

            n3, n4 = comp.get("n3", 0), comp.get("n4", 0)
            idx3, idx4 = node_map.get(n3), node_map.get(n4)

            vdiff = _get_node_v(VI, idx3) - _get_node_v(VI, idx4)  # (V+ - V-)
            psi_o = _get_node_v(PsiPhi, idx_o)  # adjoint at opamp branch equation
            k = float(comp.get("_gain_scale", 1.0))  # ramp factor (defaults to 1)

            sensitivities[f"{name}:A"] = k * psi_o * vdiff
            sensitivities_alex[name] = {"A": k * psi_o * vdiff}
        # ---------------- Nonlinear devices: Diode ----------------
        elif name.startswith("D"):
            # Use terminal voltage across diode
            vd = VI_branch

            # Start from the diode's attached model params; if none, treat comp["value"] as IS
            params = dict(comp.get("model_params", {}))
            if "IS" not in params and "value" in comp:
                params["IS"] = comp["value"]

            # helper to compute Id with modified model param
            def Id_with_param(pname, pval):
                p = dict(params)
                p[pname] = pval
                return _diode_terminal_current_from_params(vd, p, Vt)

            # sensitivities: S = -Psi_branch * dId/dp
            # Use local finite differences (no extra circuit solves!)

            # IS saturation current
            # mh we don't need finite_diff as it can be done using analytical method
            # mh we probably only need Is0 and N, the rest can be potentially removed
            Is0 = float(params.get("IS", 1e-14))
            dId_dIs = _finite_diff(lambda x: Id_with_param("IS", x), Is0)
            sensitivities[f"{name}:IS"] = -Psi_branch * dId_dIs
            sensitivities_alex[name] = {"IS": -Psi_branch * dId_dIs}

            # N emission factor
            N0 = float(params.get("N", 1.0))
            dId_dN = _finite_diff(
                lambda x: Id_with_param("N", x), N0, rel=1e-6, abs_step=1e-9
            )
            sensitivities[f"{name}:N"] = -Psi_branch * dId_dN
            sensitivities_alex[name] = {"N": -Psi_branch * dId_dN}

            # RS
            RS0 = float(params.get("RS", 0.0))
            dId_dRS = _finite_diff(
                lambda x: Id_with_param("RS", max(0.0, x)),
                RS0,
                rel=1e-6,
                abs_step=1e-12,
            )
            sensitivities[f"{name}:RS"] = -Psi_branch * dId_dRS
            sensitivities_alex[name] = {"RS": -Psi_branch * dId_dRS}

            # BV (soft breakdown curve)
            BV0 = float(params.get("BV", 0.0))
            dId_dBV = _finite_diff(
                lambda x: Id_with_param("BV", max(0.0, x)), BV0, rel=1e-6, abs_step=1e-6
            )
            sensitivities[f"{name}:BV"] = -Psi_branch * dId_dBV
            sensitivities_alex[name] = {"BV": -Psi_branch * dId_dBV}

            # IBV / NBV
            IBV0 = float(params.get("IBV", 1e-3))
            dId_dIBV = _finite_diff(
                lambda x: Id_with_param("IBV", max(0.0, x)),
                IBV0,
                rel=1e-6,
                abs_step=1e-12,
            )
            sensitivities[f"{name}:IBV"] = -Psi_branch * dId_dIBV
            sensitivities_alex[name] = {"IBV": -Psi_branch * dId_dIBV}

            NBV0 = float(params.get("NBV", 1.0))
            dId_dNBV = _finite_diff(
                lambda x: Id_with_param("NBV", max(1e-6, x)),
                NBV0,
                rel=1e-6,
                abs_step=1e-9,
            )
            sensitivities[f"{name}:NBV"] = -Psi_branch * dId_dNBV
            sensitivities_alex[name] = {"NBV": -Psi_branch * dId_dNBV}

        # ---------------- Nonlinear devices: MOSFET ----------------
        elif name.startswith("M"):
            # Drain/source indices for adjoint weighting
            nd, ns = comp["n_d"], comp["n_s"]
            idx_d, idx_s = node_map.get(nd), node_map.get(ns)
            Psi_ds = _get_node_v(PsiPhi, idx_d) - _get_node_v(PsiPhi, idx_s)

            params = dict(comp.get("model_params", {}))

            # helper to compute Id with modified model param
            def Id_mos_with_param(pname, pval):
                c2 = dict(comp)
                c2["model_params"] = dict(params)
                c2["model_params"][pname] = pval
                return _mos_current_level1(VI, node_map, c2)

            # VTO (handle synonyms by writing VTO)
            VTO0 = float(
                params.get(
                    "VTO", params.get("VT0", params.get("VTH", params.get("VTH0", 0.7)))
                )
            )
            # mh finite diff method is not needed
            dId_dVTO = _finite_diff(
                lambda x: Id_mos_with_param("VTO", x), VTO0, rel=1e-6, abs_step=1e-6
            )
            sensitivities[f"{name}:VTO"] = -Psi_ds * dId_dVTO
            sensitivities_alex[name] = {"VTO": -Psi_ds * dId_dVTO}
            # KP or MU/COX

            # mh W and L should also be included
            if "KP" in params:
                KP0 = float(params["KP"])
                dId_dKP = _finite_diff(
                    lambda x: Id_mos_with_param("KP", x), KP0, rel=1e-6, abs_step=1e-12
                )
                sensitivities[f"{name}:KP"] = -Psi_ds * dId_dKP
                sensitivities_alex[name] = {"KP": -Psi_ds * dId_dKP}
            else:
                mu0 = float(params.get("MU", params.get("UO", params.get("U0", 0.0))))
                cox0 = float(params.get("C_OX", params.get("COX", 0.0)))

                dId_dMU = _finite_diff(
                    lambda x: Id_mos_with_param("MU", x), mu0, rel=1e-6, abs_step=1e-12
                )
                dId_dCOX = _finite_diff(
                    lambda x: Id_mos_with_param("C_OX", x),
                    cox0,
                    rel=1e-6,
                    abs_step=1e-12,
                )

                sensitivities[f"{name}:MU"] = -Psi_ds * dId_dMU
                sensitivities_alex[name] = {"MU": -Psi_ds * dId_dMU}
                sensitivities[f"{name}:COX"] = -Psi_ds * dId_dCOX
                sensitivities_alex[name] = {"COX": -Psi_ds * dId_dCOX}

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
    """Aggregates multi-step simulation sensitivity data into traced arrays."""
    print("Starting sensitivity data aggregation...")
    target_nodes = output_nodes if output_nodes is not None else list(node_map.keys())

    # include nonlinear param keys too
    all_keys = _list_sensitivity_keys(components)

    sensitivity_dict: Dict[str, Dict[str, np.ndarray]] = {
        node: {k: np.array([]) for k in all_keys} for node in target_nodes
    }
    sensitivity_dict_alex: Dict[str, Dict[str, Dict[str, complex]]] = {}
    if raw_sensitivities is None:
        if not list_of_lus or VI_list is None:
            return sensitivity_dict

        print("Computing sensitivities from stored LU matrices...")
        is_ac = ".AC" in analyses
        raw_sensitivities: List[Dict[str, Dict[str, complex]]] = []
        raw_sensitivities_alex: List[Dict[str, Dict[str, Dict[str, complex]]]] = []

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
    # mh maybe we should start this from scratch, all the data is ok , but the data structure is really bad.
    # mh step_data is a dict of nodes, each node is also a dict of componenet paramters (comp1:Is0, comp1:N, comp2:VTO, etc)
    # and its value is the sensitivity calculated from before.ls
    for step_index, step_data in enumerate(raw_sensitivities):
        for node in target_nodes:
            sensitivity_dict_alex[node] = raw_sensitivities_alex[step_index][node]
            for k, v in step_data[node].items():
                # if k not in sensitivity_dict[node]:
                sensitivity_dict[node][k] = np.array([v])
            # sensitivity_dict[node][k].append(v)

    # mh we are just changing sensitivity_dict[node][k] from a single valued list into a single values np array, WTF
    # for node in sensitivity_dict:
    #     for k in sensitivity_dict[node]:
    #         sensitivity_dict[node][k] = np.array(sensitivity_dict[node][k])

    print("Sensitivity aggregation complete.")
    return sensitivity_dict, sensitivity_dict_alex


def estimate_std_dev(sensitivities, components, percent_sigma=0.01):
    """
    Computes output standard deviation.
    Updated to support nonlinear param keys like D1:IS, M1:VTO, O1:A, etc.
    """
    variance = 0.0
    for key, sens in sensitivities.items():
        if isinstance(sens, np.ndarray) and sens.size == 0:
            continue
        if np.isscalar(sens) and sens == 0:
            continue

        # Determine sigma_p
        if ":" in key:
            dev, pname = key.split(":", 1)
            comp = components.get(dev, {})
            params = comp.get("model_params", {})
            inst = comp.get("inst_params", {})

            if dev.startswith("D"):
                base = float(
                    params.get(pname, params.get(pname.upper(), comp.get("value", 0.0)))
                )
            elif dev.startswith("M"):
                if pname == "VTO":
                    base = float(
                        params.get(
                            "VTO",
                            params.get(
                                "VT0", params.get("VTH", params.get("VTH0", 0.0))
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
