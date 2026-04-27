"""
Provides functions for ranking component sensitivities 
and generating fault tables (opens and shorts) using adjoint sensitivity 
data.

Short faults: R_short = dVout/ (v_nom * psi_nom)
    where v_nom is the voltage difference between shorting nodes in the 
    original circuit, and psi_nom is the voltage difference in the adjoint circuit.

Open faults: R_open = dVout / (i_nom * phi_nom)
    where i_nom is the branch current in the original circuit, and phi_nom 
    is the branch current in the adjoint circuit.

Large Change Sensitivity:
    Both shorts and opens use the same formula:
        v_hat = v - (v_oc / (R + R_TH)) * Y_inv_xi
    
    For shorts: R is a small positive resistance added between two nodes.
    For opens:  R = -R_branch (negative resistance cancels existing conductance).
               The conductance to cancel is whatever the component stamps into 
               the Y matrix (1/R for resistors, C/dt for capacitors, etc.)
"""

import numpy as np
from applications.large_change_sensitivity import build_xi, compute_large_change


def rank_component_sensitivities(result):
    """
    Rank order component sensitivities from largest to smallest.
    Uses the data stored in the SensitivityData tensor.
    
    Args:
        result (SimulationResult): The simulation result object with 
            sensitivities already computed.
    
    Returns:
        list[dict]: Sorted list of sensitivity rankings, or None if 
            no sensitivity data is available.
    """
    tensor = result.sensitivities
    if tensor is None:
        print("No sensitivity data found. Run simulation with sensitivity=True.")
        return None

    rank_list = []

    for p_name in tensor.param_names:
        for o_node in tensor.output_nodes:
            s_waveform = tensor.get_sweep_series(p_name, o_node)
            
            peak_idx = np.argmax(np.abs(s_waveform))
            impact = s_waveform[peak_idx]
            peak_time = tensor.sweep_axis[peak_idx]
            
            rank_list.append({
                "name": p_name,
                "output": o_node,
                "impact": impact,
                "time": peak_time
            })

    ranked = sorted(rank_list, key=lambda x: abs(x['impact']), reverse=True)

    print(f"\n{'Rank':<5} | {'Parameter':<12} | {'Impact':<12} | {'At Time':<10} | {'Target'}")
    print("-" * 65)
    for i, item in enumerate(ranked[:10]):
        print(f"{i+1:<5} | {item['name']:<12} | {item['impact']:>10.3e} | {item['time']:>8.2e}s | {item['output']}")

    return ranked


def perform_global_ranking(circuit, result):
    """
    Generate a fault table ranking both open and short faults for 
    each output node using adjoint sensitivity data.
    
    For opens: uses the existing component sensitivity waveforms.
    For shorts: computes v_diff * psi_diff for all node pairs using 
        the raw adjoint vectors stored in the SensitivityData tensor.
    
    Args:
        circuit (Circuit): The circuit object with components and node_map.
        result (SimulationResult): The simulation result with sensitivities.
    
    Returns:
        dict: { 'node_name': [ranked_faults_list] } for each output node.
    """
    tensor = result.sensitivities
    if tensor is None:
        print("No sensitivity data found. Run simulation with sensitivity=True.")
        return None

    vi_nom = result.VI
    psi_nom = tensor.adjoint_vectors

    if psi_nom is None:
        print("No raw adjoint vectors found. Cannot compute short faults.")
        use_adjoint = False
    else:
        use_adjoint = True

    branch_names = set()
    for comp in circuit.components:
        if comp.type in ["V", "L", "H", "E"]:
            branch_names.add(comp.name)

    voltage_nodes = [(name, idx) for name, idx in circuit.node_map.items()
                     if name not in branch_names]

    all_rankings = {}

    for out_node in tensor.output_nodes:
        o_idx = tensor.output_index[out_node]
        node_faults = []

        # --- 1. Open Faults (phi_b × i_b for each component branch) ---
        # For opens, rank by |phi_b × i_b| where:
        #   i_b = forward branch current through the component
        #   phi_b = adjoint branch current through the component
        # This gives uniform units (A^2) across all component types.
        num_steps = vi_nom.shape[0] if vi_nom.ndim == 2 else 1
        dt = tensor.sweep_axis[1] - tensor.sweep_axis[0] if len(tensor.sweep_axis) > 1 else 1.0
        
        for comp in circuit.components:
            # Skip independent sources — they can't "open" in the conductance sense
            if comp.type in ["V", "I"]:
                continue
            
            # Compute forward branch current and adjoint branch current
            if comp.type == "R":
                idx_i = getattr(comp, 'idx_1', None)
                idx_j = getattr(comp, 'idx_2', None)
                vi = vi_nom[:, idx_i] if idx_i is not None else 0.0
                vj = vi_nom[:, idx_j] if idx_j is not None else 0.0
                pi = psi_nom[o_idx, :, idx_i] if idx_i is not None else 0.0
                pj = psi_nom[o_idx, :, idx_j] if idx_j is not None else 0.0
                
                i_b = (vi - vj) / comp.value
                phi_b = (pi - pj)
                
            elif comp.type == "C":
                idx_i = getattr(comp, 'idx_1', None)
                idx_j = getattr(comp, 'idx_2', None)
                vi = vi_nom[:, idx_i] if idx_i is not None else 0.0
                vj = vi_nom[:, idx_j] if idx_j is not None else 0.0
                pi = psi_nom[o_idx, :, idx_i] if idx_i is not None else 0.0
                pj = psi_nom[o_idx, :, idx_j] if idx_j is not None else 0.0
                
                g_eq = 2.0 * comp.value / dt if dt > 0 else comp.value
                i_b = g_eq * (vi - vj)
                phi_b = (pi - pj)
                
            elif comp.type == "L":
                branch_idx = getattr(comp, 'branch_idx', None)
                if branch_idx is None:
                    continue
                idx_i = getattr(comp, 'idx_1', None)
                idx_j = getattr(comp, 'idx_2', None)
                
                i_b = vi_nom[:, branch_idx]
                phi_b = psi_nom[o_idx, :, branch_idx]
                
            elif comp.type in ["M_NMOS", "M_PMOS"]:
                idx_d = getattr(comp, 'idx_d', None)
                idx_g = getattr(comp, 'idx_g', None)
                idx_s = getattr(comp, 'idx_s', None)
                
                # Compute I_D from the model at each time step
                from core.models import evaluate_nmos
                open_impact = np.zeros(num_steps)
                for t_idx in range(num_steps):
                    v = vi_nom[t_idx] if vi_nom.ndim == 2 else vi_nom
                    vd = v[idx_d] if idx_d is not None else 0.0
                    vg = v[idx_g] if idx_g is not None else 0.0
                    vs = v[idx_s] if idx_s is not None else 0.0
                    vgs = comp.POLARITY * (vg - vs)
                    vds = comp.POLARITY * (vd - vs)
                    res = evaluate_nmos(vgs, vds, comp.VTO, comp.Bn)
                    i_d = res["I_D"]
                    
                    psi_d = psi_nom[o_idx, t_idx, idx_d] if idx_d is not None else 0.0
                    psi_s = psi_nom[o_idx, t_idx, idx_s] if idx_s is not None else 0.0
                    phi_ds = (psi_d - psi_s)
                    
                    open_impact[t_idx] = i_d * phi_ds
                
                peak_idx = np.argmax(np.abs(open_impact))
                node_faults.append({
                    "type": "Open",
                    "location": comp.name,
                    "impact": open_impact[peak_idx],
                    "time": tensor.sweep_axis[peak_idx]
                })
                continue
            
            else:
                continue
            
            # Compute phi_b × i_b waveform for R, C, L
            open_impact = i_b * phi_b
            peak_idx = np.argmax(np.abs(open_impact))
            node_faults.append({
                "type": "Open",
                "location": comp.name,
                "impact": open_impact[peak_idx],
                "time": tensor.sweep_axis[peak_idx]
            })

        # --- 2. Short Faults (v_diff * psi_diff for node pairs) ---
        for i in range(len(voltage_nodes)):
            for j in range(i + 1, len(voltage_nodes)):
                n1, idx1 = voltage_nodes[i]
                n2, idx2 = voltage_nodes[j]

                v_diff = vi_nom[:, idx1] - vi_nom[:, idx2]

                if use_adjoint:
                    psi_diff = psi_nom[o_idx, :, idx1] - psi_nom[o_idx, :, idx2]
                    short_sens_waveform = v_diff * psi_diff

                peak_idx = np.argmax(np.abs(short_sens_waveform))
                node_faults.append({
                    "type": "Short",
                    "location": f"{n1}<->{n2}",
                    "impact": short_sens_waveform[peak_idx],
                    "time": tensor.sweep_axis[peak_idx]
                })

        node_faults.sort(key=lambda x: abs(x["impact"]), reverse=True)
        all_rankings[out_node] = node_faults

    return all_rankings


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def _get_component_conductance(comp, dt=None, method='TR'):
    """
    Get the conductance that a component stamps into the Y matrix.
    This is what we need to cancel for an open fault.
    
    Args:
        comp: The component object.
        dt (float): Time step size (needed for C and L in transient).
        method (str): Integration method ('TR' or 'BE').
    
    Returns:
        tuple: (g, idx_k, idx_l) where g is the conductance value,
            idx_k and idx_l are the node indices. Returns (None, None, None)
            if the component type is not supported.
    """
    if comp.type == "R":
        # Resistor: stamps g = 1/R between idx_1 and idx_2
        g = 1.0 / comp.value
        return g, comp.idx_1, comp.idx_2
    
    elif comp.type == "C":
        # Capacitor: stamps g = C/dt (BE) or g = 2C/dt (TR)
        if dt is None:
            return None, None, None
        if method == 'TR':
            g = 2.0 * comp.value / dt
        else:
            g = comp.value / dt
        return g, comp.idx_1, comp.idx_2
    
    elif comp.type == "L":
        # Inductor: stamps into branch equation, not a simple conductance.
        # The inductor stamps -req at (branch_idx, branch_idx)
        # where req = 2L/dt (TR) or L/dt (BE).
        # For open fault, we'd need to cancel this, but inductors use
        # branch equations, so we use its terminal nodes instead.
        if dt is None:
            return None, None, None
        if method == 'TR':
            g = dt / (2.0 * comp.value)
        else:
            g = dt / comp.value
        return g, comp.idx_1, comp.idx_2
    
    elif comp.type in ["M_NMOS", "M_PMOS"]:
        # MOSFET: stamps gds between drain and source.
        # For a complete open, we'd cancel gds (and gm indirectly).
        # Use gds as the primary conductance to cancel.
        gds = getattr(comp, '_last_gds', None)
        if gds is not None and gds > 0:
            return gds, comp.idx_d, comp.idx_s
        return None, None, None
    
    return None, None, None


def _r_eff_to_r_added(R_eff, R_branch):
    """
    Convert a desired effective component resistance to the 
    negative R value needed for the large change formula.
    
    Args:
        R_eff (float): The desired effective resistance of the component.
        R_branch (float): The component's nominal resistance (1/g).
    
    Returns:
        float: The R value to plug into compute_large_change.
    """
    delta_g = (1.0 / R_eff) - (1.0 / R_branch)
    
    if abs(delta_g) < 1e-30:
        return np.inf
    
    return 1.0 / delta_g


# ==========================================
# FAULT THRESHOLD COMPUTATION
# ==========================================

def compute_fault_thresholds(circuit, result, output_node, delta_F,
                              top_n=10, r_sweep_shorts=None, r_sweep_opens=None,
                              dt=None, method='TR'):
    """
    Compute fault resistance thresholds using the large change sensitivity 
    formula.
    
    Args:
        circuit (Circuit): The circuit object.
        result (SimulationResult): Simulation result with LU factorizations.
        output_node (str): The output node to observe (e.g., "out").
        delta_F (float): Acceptable output tolerance in volts.
        top_n (int): Number of top-ranked faults to compute thresholds for.
        r_sweep_shorts (np.ndarray, optional): R values for shorts.
        r_sweep_opens (np.ndarray, optional): Effective R values for opens.
        dt (float, optional): Time step for companion model conductances.
            If None, extracted from the sweep axis.
        method (str): Integration method ('TR' or 'BE').
    
    Returns:
        dict: { 'shorts': [...], 'opens': [...] } with threshold data.
    """
    # --- Validate inputs ---
    if not hasattr(result, 'list_of_lus') or result.list_of_lus is None:
        print("No LU factorizations stored. Run with keep_lus=True.")
        return None

    tensor = result.sensitivities
    if tensor is None:
        print("No sensitivity data found.")
        return None

    if r_sweep_shorts is None:
        r_sweep_shorts = np.logspace(7, -1, 1000)

    # Extract dt from the sweep axis if not provided
    if dt is None and len(tensor.sweep_axis) > 1:
        dt = tensor.sweep_axis[1] - tensor.sweep_axis[0]

    # --- Get indices and dimensions ---
    n = circuit.total_dim
    out_idx = circuit.get_idx(output_node)
    num_steps = len(result.list_of_lus)

    # --- Get ranked faults ---
    all_rankings = perform_global_ranking(circuit, result)
    if all_rankings is None or output_node not in all_rankings:
        return None
    ranked_faults = all_rankings[output_node]

    # --- Process faults ---
    # threshold_results = {"shorts": [], "opens": []}
    threshold_results = {"shorts": [], "opens": [], "domain": tensor.domain}
    short_count = 0
    open_count = 0

    for fault in ranked_faults:
        if short_count >= top_n and open_count >= top_n:
            break

        # ======================
        # SHORT FAULTS
        # ======================
        if fault["type"] == "Short" and short_count < top_n:

            parts = fault["location"].split("<->")
            n1_name, n2_name = parts[0], parts[1]

            idx_k = circuit.node_map.get(n1_name)
            idx_l = circuit.node_map.get(n2_name)
            if idx_k is None:
                try: idx_k = circuit.node_map.get(int(n1_name))
                except ValueError: pass
            if idx_l is None:
                try: idx_l = circuit.node_map.get(int(n2_name))
                except ValueError: pass

            if idx_k is None and idx_l is None:
                continue

            xi_kl = build_xi(n, idx_k, idx_l)

            thresholds_per_step = []
            for t_idx in range(num_steps):
                lu = result.list_of_lus[t_idx]
                v = result.VI[t_idx] if result.VI.ndim == 2 else result.VI

                threshold_at_step = None
                for R in r_sweep_shorts:
                    delta_v = compute_large_change(lu, xi_kl, v, R, out_idx)
                    if abs(delta_v) >= delta_F:
                        threshold_at_step = R
                        break
                thresholds_per_step.append(threshold_at_step)

            valid = [(t_idx, R_th) for t_idx, R_th in enumerate(thresholds_per_step)
                     if R_th is not None]

            if valid:
                max_entry = max(valid, key=lambda x: x[1])
                best_t_idx = max_entry[0]
                best_R = max_entry[1]
                det_times = [tensor.sweep_axis[t] for t, _ in valid]

                v_at_best = result.VI[best_t_idx] if result.VI.ndim == 2 else result.VI
                v_nominal = v_at_best[out_idx]
                delta_v_at_best = compute_large_change(result.list_of_lus[best_t_idx], xi_kl, v_at_best, best_R, out_idx)
                v_faulted = v_nominal + delta_v_at_best

                threshold_results["shorts"].append({
                    "location": fault["location"],
                    "threshold_R": best_R,
                    "best_time": tensor.sweep_axis[best_t_idx],
                    "obs_window": (min(det_times), max(det_times)),
                    "v_nominal": float(np.real(v_nominal)),
                    "v_faulted": float(np.real(v_faulted)),
                    "delta_v": float(np.real(delta_v_at_best)),
                    "sensitivity_impact": fault["impact"]
                })
            else:
                threshold_results["shorts"].append({
                    "location": fault["location"],
                    "threshold_R": None, "best_time": None, "obs_window": None,
                    "v_nominal": None, "v_faulted": None, "delta_v": None,
                    "sensitivity_impact": fault["impact"]
                })
            short_count += 1

        # ======================
        # OPEN FAULTS
        # ======================
        elif fault["type"] == "Open" and open_count < top_n:

            comp_name = fault["location"]
            # Handle MOSFET parameter names (e.g., "M1_W" -> "M1")
            base_name = comp_name.split("_")[0] if "_" in comp_name else comp_name
            comp = circuit.components_dict.get(base_name)

            if comp is None:
                continue

            # Get the conductance this component stamps into the Y matrix
            g, idx_k, idx_l = _get_component_conductance(comp, dt=dt, method=method)

            if g is None or (idx_k is None and idx_l is None):
                continue

            # R_branch = 1/g (the effective resistance of this component)
            R_branch = 1.0 / g

            xi_kl = build_xi(n, idx_k, idx_l)

            # Build sweep of effective resistance values
            if r_sweep_opens is not None:
                open_sweep = r_sweep_opens
            else:
                open_sweep = np.logspace(
                    np.log10(R_branch * 1.01),
                    np.log10(R_branch * 1e4),
                    1000
                )

            thresholds_per_step = []
            for t_idx in range(num_steps):
                lu = result.list_of_lus[t_idx]
                v = result.VI[t_idx] if result.VI.ndim == 2 else result.VI

                threshold_at_step = None
                for R_eff in open_sweep:
                    R_added = _r_eff_to_r_added(R_eff, R_branch)
                    delta_v = compute_large_change(lu, xi_kl, v, R_added, out_idx)
                    if abs(delta_v) >= delta_F:
                        threshold_at_step = R_eff
                        break
                thresholds_per_step.append(threshold_at_step)

            valid = [(t_idx, R_th) for t_idx, R_th in enumerate(thresholds_per_step)
                     if R_th is not None]

            if valid:
                min_entry = min(valid, key=lambda x: x[1])
                best_t_idx = min_entry[0]
                best_R_eff = min_entry[1]
                det_times = [tensor.sweep_axis[t] for t, _ in valid]

                v_at_best = result.VI[best_t_idx] if result.VI.ndim == 2 else result.VI
                v_nominal = v_at_best[out_idx]
                R_added_best = _r_eff_to_r_added(best_R_eff, R_branch)
                delta_v_at_best = compute_large_change(result.list_of_lus[best_t_idx], xi_kl, v_at_best, R_added_best, out_idx)
                v_faulted = v_nominal + delta_v_at_best

                threshold_results["opens"].append({
                    "location": comp_name,
                    "comp_type": comp.type,
                    "nominal_R": R_branch,
                    "threshold_R": best_R_eff,
                    "best_time": tensor.sweep_axis[best_t_idx],
                    "obs_window": (min(det_times), max(det_times)),
                    "v_nominal": float(np.real(v_nominal)),
                    "v_faulted": float(np.real(v_faulted)),
                    "delta_v": float(np.real(delta_v_at_best)),
                    "sensitivity_impact": fault["impact"]
                })
            else:
                threshold_results["opens"].append({
                    "location": comp_name,
                    "comp_type": comp.type,
                    "nominal_R": R_branch,
                    "threshold_R": None, "best_time": None, "obs_window": None,
                    "v_nominal": None, "v_faulted": None, "delta_v": None,
                    "sensitivity_impact": fault["impact"]
                })
            open_count += 1

    return threshold_results


# ==========================================
# PRINT FUNCTIONS
# ==========================================

def print_fault_table(all_fault_tables, top_n=15):
    """
    Pretty-print the fault ranking tables.
    
    Args:
        all_fault_tables (dict): Output from perform_global_ranking().
        top_n (int): Number of top faults to display per node.
    """
    if all_fault_tables is None:
        return

    for node_name, ranked_list in all_fault_tables.items():
        print("\n" + "=" * 70)
        print(f"  FAULT TABLE FOR OUTPUT: V({node_name})")
        print("=" * 70)
        print(f"{'Rank':<5} | {'Type':<10} | {'Location':<15} | {'Peak Impact':<14} | {'At Time'}")
        print("-" * 70)
        for i, fault in enumerate(ranked_list[:top_n]):
            print(f"{i+1:<5} | {fault['type']:<10} | {fault['location']:<15} | {fault['impact']:>12.3e} | {fault['time']:>8.2e}s")


def print_threshold_table(threshold_results):
    """
    Pretty-print the fault threshold tables with voltage values 
    and observation windows.
    
    Args:
        threshold_results (dict): Output from compute_fault_thresholds().
    """
    if threshold_results is None:
        return

    domain = threshold_results.get("domain", "time")
    is_ac = (domain == "frequency")
    time_label = "Best Freq" if is_ac else "Best Time"
    window_label = "Freq. Window" if is_ac else "Obs. Window"
    unit = "Hz" if is_ac else "s"

    # --- Short Thresholds ---
    shorts = threshold_results.get("shorts", [])
    if shorts:
        print("\n" + "=" * 110)
        print("  SHORT FAULT THRESHOLDS  ")
        print("  Max R: the weakest short that is still detectable at the output")
        print("=" * 110)
        # print(f"{'Rank':<5} | {'Location':<15} | {'Threshold R':<14} | {'V_nom':<10} | {'V_fault':<10} | {'ΔV':<10} | {'Best Time':<12} | {'Obs. Window'}")
        print(f"{'Rank':<5} | {'Location':<15} | {'Threshold R':<14} | {'V_nom':<10} | {'V_fault':<10} | {'ΔV':<10} | {time_label:<12} | {window_label}")
        print("-" * 110)
        for i, s in enumerate(shorts):
            if s['threshold_R'] is not None:
                r_str = f"{s['threshold_R']:.2e} Ω"
                vn_str = f"{s['v_nominal']:.4f}V"
                vf_str = f"{s['v_faulted']:.4f}V"
                dv_str = f"{s['delta_v']:+.4f}V"
                # t_str = f"{s['best_time']:.2e}s"
                # obs_str = f"{s['obs_window'][0]:.2e} - {s['obs_window'][1]:.2e}s"
                t_str = f"{s['best_time']:.2e}{unit}"
                obs_str = f"{s['obs_window'][0]:.2e} - {s['obs_window'][1]:.2e}{unit}"
            else:
                r_str = "N/A"
                vn_str = "N/A"
                vf_str = "N/A"
                dv_str = "N/A"
                t_str = "N/A"
                obs_str = "N/A"
            print(f"{i+1:<5} | {s['location']:<15} | {r_str:<14} | {vn_str:<10} | {vf_str:<10} | {dv_str:<10} | {t_str:<12} | {obs_str}")

    # --- Open Thresholds ---
    opens = threshold_results.get("opens", [])
    if opens:
        print("\n" + "=" * 120)
        print("  OPEN FAULT THRESHOLDS  ")
        print("  Min R: component resistance at which degradation becomes detectable")
        print("=" * 120)
        print(f"{'Rank':<5} | {'Component':<15} | {'Type':<6} | {'Nominal R':<12} | {'Threshold R':<14} | {'V_nom':<10} | {'V_fault':<10} | {'ΔV':<10} | {window_label}")
        print("-" * 120)
        for i, o in enumerate(opens):
            nom_str = f"{o['nominal_R']:.2e} Ω"
            type_str = o.get('comp_type', '?')
            if o['threshold_R'] is not None:
                r_str = f"{o['threshold_R']:.2e} Ω"
                vn_str = f"{o['v_nominal']:.4f}V"
                vf_str = f"{o['v_faulted']:.4f}V"
                dv_str = f"{o['delta_v']:+.4f}V"
                obs_str = f"{o['obs_window'][0]:.2e} - {o['obs_window'][1]:.2e}{unit}"
            else:
                r_str = "N/A"
                vn_str = "N/A"
                vf_str = "N/A"
                dv_str = "N/A"
                obs_str = "N/A"
            print(f"{i+1:<5} | {o['location']:<15} | {type_str:<6} | {nom_str:<12} | {r_str:<14} | {vn_str:<10} | {vf_str:<10} | {dv_str:<10} | {obs_str}")