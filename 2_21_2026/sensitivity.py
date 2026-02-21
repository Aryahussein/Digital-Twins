import numpy as np

def get_all_sensitivities(components, VI, PsiPhi, node_map, w=0.0, dt=None):
    """
    Compute sensitivity of the output w.r.t. all component parameters using
    the adjoint vector PsiPhi.
    
    For AC analysis, pass w (angular frequency).
    For transient analysis, pass dt (time step) — this uses BE companion model derivatives.
    
    Supported: R, C, L, G, V (source value), I (source value)
    Not yet supported: D (diode Is sensitivity)
    """
    sensitivities = {}

    for name, comp in components.items():
        # 1. Get Indices and branch voltages
        n1, n2 = comp.get("n1", 0), comp.get("n2", 0)
        idx1, idx2 = node_map.get(n1), node_map.get(n2)

        # Helper to get the voltage difference across a branch
        # (v1 - v2). If a node is ground, its voltage is 0.
        VI_branch = (VI[idx1] if idx1 is not None else 0) - \
                   (VI[idx2] if idx2 is not None else 0)
        
        PsiPhi_branch = (PsiPhi[idx1] if idx1 is not None else 0) - \
                       (PsiPhi[idx2] if idx2 is not None else 0)

        # 2. Apply the sensitivity formula based on component type
        if name.startswith("R"):
            R = comp["value"]
            sensitivities[name] = (1.0 / (R**2)) * (VI_branch * PsiPhi_branch)

        elif name.startswith("C"):
            if dt is not None:
                # Transient (Backward Euler): companion conductance = C/dt
                # dY/dC = 1/dt, so sensitivity = -(1/dt) * dV * dPsi
                sensitivities[name] = -(1.0 / dt) * (VI_branch * PsiPhi_branch)
            else:
                # AC: dY/dC = j*w
                sensitivities[name] = -1j * w * (VI_branch * PsiPhi_branch)

        elif name.startswith("L"):
            l_curr_idx = node_map[name]
            i_L = VI[l_curr_idx]
            i_L_hat = PsiPhi[l_curr_idx]
            
            if dt is not None:
                # Transient (Backward Euler): companion impedance = -L/dt on diagonal
                # dY/dL = -1/dt, so sensitivity = (1/dt) * i_L * psi_L
                sensitivities[name] = (1.0 / dt) * (i_L * i_L_hat)
            else:
                # AC: dY/dL = -j*w
                sensitivities[name] = 1j * w * (i_L * i_L_hat)

        elif name.startswith("G"):
            n3, n4 = comp.get("n3", 0), comp.get("n4", 0)
            idx3, idx4 = node_map.get(n3), node_map.get(n4)
            
            v_sense = (VI[idx3] if idx3 is not None else 0) - \
                      (VI[idx4] if idx4 is not None else 0)
            
            sensitivities[name] = - (PsiPhi_branch * v_sense)

        elif name.startswith("V"):
            v_branch_idx = node_map[name]
            sensitivities[name] = PsiPhi[v_branch_idx]

        elif name.startswith("I"):
            sensitivities[name] = -PsiPhi_branch

        # Note: Diode (D) sensitivity w.r.t. Is is not implemented.

    return sensitivities

def compute_step_sensitivities(lu, VI, components, node_map, output_nodes=None, w=0.0, dt=None):
    """Solves the adjoint system and gathers sensitivities for requested output nodes.
    
    For AC analysis, pass w (angular frequency).
    For transient analysis, pass dt (time step).
    """
    from solver import solve_adjoint
    if output_nodes is None:
        output_nodes = list(node_map.keys())

    step_sensitivities = {}
    for out_node in output_nodes:
        PsiPhi = solve_adjoint(lu, out_node, node_map)
        comp_sens = get_all_sensitivities(components, VI, PsiPhi, node_map, w=w, dt=dt)
        step_sensitivities[out_node] = comp_sens

    return step_sensitivities

def aggregate_sweep_sensitivities(components, node_map, analyses, 
                                  output_nodes=None, raw_sensitivities=None, 
                                  list_of_lus=None, VI_list=None, freq_list=None, dt=None):
    """Aggregates multi-step simulation data into traced arrays.
    
    For post-processing mode (raw_sensitivities=None), pass freq_list for AC
    or dt for transient analysis.
    """
    print("Starting sensitivity data aggregation...")
    target_nodes = output_nodes if output_nodes is not None else list(node_map.keys())
    
    # Initialize nested dict: {node: {comp_name: []}}
    sensitivity_dict = {
        node: {name: [] for name in components.keys()} for node in target_nodes
    }

    # If raw sensitivities weren't computed during the simulation loop, compute them now
    if raw_sensitivities is None:
        if not list_of_lus or VI_list is None:
            return sensitivity_dict
            
        print("Computing sensitivities from stored LU matrices...")
        is_ac = ".AC" in analyses
        raw_sensitivities = []
        
        for i in range(len(list_of_lus)):
            if is_ac and freq_list is not None:
                w_step = 2 * np.pi * freq_list[i]
                step_sens = compute_step_sensitivities(
                    list_of_lus[i], VI_list[i], components, node_map, target_nodes, w=w_step
                )
            else:
                # Transient mode: pass dt
                step_sens = compute_step_sensitivities(
                    list_of_lus[i], VI_list[i], components, node_map, target_nodes, dt=dt
                )
            raw_sensitivities.append(step_sens)

    # Transpose data: raw_sensitivities is a list of dicts -> [{node: {comp: val}}]
    for step_data in raw_sensitivities:
        for node in target_nodes:
            for comp_name, sens_value in step_data[node].items():
                sensitivity_dict[node][comp_name].append(sens_value)

    # Convert lists to numpy arrays for easier plotting/math later
    for node in sensitivity_dict:
        for comp in sensitivity_dict[node]:
            sensitivity_dict[node][comp] = np.array(sensitivity_dict[node][comp])

    print("Sensitivity aggregation complete.")
    # print(sensitivity_dict)
    return sensitivity_dict

def estimate_std_dev(sensitivities, components, percent_sigma=0.01):
    """
    Computes output standard deviation.
    Works for scalars (Single Step) OR numpy arrays (Sweeps).
    """
    variance = 0.0
    for name, sens in sensitivities.items():
        # Skip empty arrays
        if isinstance(sens, np.ndarray) and sens.size == 0:
            continue
        # Skip zero scalar sensitivities
        if np.isscalar(sens) and sens == 0:
            continue

        # Safely get component value, default to 0 if it doesn't have a standard "value"
        comp_val = components[name].get("value", 0.0) 
        sigma_p = comp_val * percent_sigma
        
        # Add to variance (np.abs handles both reals and complex AC magnitudes safely)
        variance += np.abs(sens * sigma_p)**2
        
    return np.sqrt(variance)

        
