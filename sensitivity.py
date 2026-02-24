import numpy as np

def get_all_sensitivities(components, VI, PsiPhi, node_map, w=0.0):
    sensitivities = {}

    for name, comp in components.items():
        # 1. Get Indices and branch voltages
        n1, n2 = comp.get("n1", 0), comp.get("n2", 0)
        idx1, idx2 = node_map.get(n1), node_map.get(n2) # get_idx logic

        # Helper to get the voltage difference across a branch
        # (v1 - v2). If a node is ground, its voltage is 0.
        VI_branch = (VI[idx1] if idx1 is not None else 0) - \
                   (VI[idx2] if idx2 is not None else 0)
        
        PsiPhi_branch = (PsiPhi[idx1] if idx1 is not None else 0) - \
                       (PsiPhi[idx2] if idx2 is not None else 0)

        # 2. Apply the sensitivity formula based on component type
        if name.startswith("R"):
            # Sensitivity w.r.t Resistance (R):
            # dY/dR = (1/R^2) * VI_branch * PsiPhi_branch
            R = comp["value"]
            sensitivities[name] = (1.0 / (R**2)) * (VI_branch * PsiPhi_branch)

        elif name.startswith("C"):
            # Sensitivity w.r.t Capacitance (C):
            # dY/dC = j*w. Sensitivity = - (j*w * VI_branch * PsiPhi_branch)
            sensitivities[name] = -1j * w * (VI_branch * PsiPhi_branch)

        elif name.startswith("L"):
            # For Inductors, we use the MNA branch current
            # Row index for the inductor current is its entry in node_map
            l_curr_idx = node_map[name]
            i_L = VI[l_curr_idx]
            i_L_hat = PsiPhi[l_curr_idx]
            
            # dY/dL for an inductor row is -j*w
            # Sensitivity = - (-j*w * i_L * i_L_hat) = j*w * i_L * i_L_hat
            sensitivities[name] = 1j * w * (i_L * i_L_hat)

        elif name.startswith("G"):
            # VCCS: I = gm * (VI_sense_pos - VI_sense_neg)
            # Sensitive w.r.t transconductance (gm)
            n3, n4 = comp.get("n3", 0), comp.get("n4", 0)
            idx3, idx4 = node_map.get(n3), node_map.get(n4)
            
            v_sense = (VI[idx3] if idx3 is not None else 0) - \
                      (VI[idx4] if idx4 is not None else 0)
            
            # For VCCS, sensitivity = - (PsiPhi_branch_current * VI_sense)
            # In adjoint, this is PsiPhi_branch (voltage at the current source nodes)
            sensitivities[name] = - (PsiPhi_branch * v_sense)

    return sensitivities

def compute_step_sensitivities(lu, VI, components, node_map, output_nodes=None, w=0.0):
    """Solves the adjoint system and gathers sensitivities for requested output nodes."""
    from solver import solve_adjoint
    # print(node_map)
    # print(output_nodes)
    if output_nodes is None:
        output_nodes = list(node_map.keys())

    # print(f"Computing sensitivities for {output_nodes}")

    step_sensitivities = {}
    for out_node in output_nodes:
        # 1. Solve the adjoint circuit for this specific output node
        print(f"Solvin adjoint at output node {out_node}...")
        PsiPhi = solve_adjoint(lu, out_node, node_map)
        
        # 2. Compute component sensitivities
        comp_sens = get_all_sensitivities(components, VI, PsiPhi, node_map, w=w)
        
        # 3. Store in dictionary keyed by the output node
        step_sensitivities[out_node] = comp_sens

    return step_sensitivities

def aggregate_sweep_sensitivities(components, node_map, analyses, 
                                  output_nodes=None, raw_sensitivities=None, 
                                  list_of_lus=None, VI_list=None, freq_list=None):
    """Aggregates multi-step simulation data into traced arrays."""
    print("Starting sensitivity data aggregation...")
    target_nodes = output_nodes if output_nodes is not None else list(node_map.keys())
    
    # Initialize nested dict: {node: {comp_name: []}}
    sensitivity_dict = {
        node: {name: [] for name in components.keys()} for node in target_nodes
    }

    # If raw sensitivities weren't computed during the simulation loop, compute them now
    if not raw_sensitivities:
        if not list_of_lus or VI_list is None:
            return sensitivity_dict
            
        print("Computing sensitivities from stored LU matrices...")
        is_ac = ".AC" in analyses
        raw_sensitivities = []
        
        for i in range(len(list_of_lus)):
            w_step = 2 * np.pi * freq_list[i] if (is_ac and freq_list is not None) else 0.0
            # print(f"w_step = {w_step}")
            
            step_sens = compute_step_sensitivities(
                list_of_lus[i], VI_list[i], components, node_map, target_nodes, w_step
            )
            # print(step_sens)
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
        # SKIP EMPTY LISTS/ARRAYS HERE
        if len(sens) == 0:
            continue

        # Safely get component value, default to 0 if it doesn't have a standard "value"
        comp_val = components[name].get("value", 0.0) 
        sigma_p = comp_val * percent_sigma
        
        # Add to variance (np.abs handles both reals and complex AC magnitudes safely)
        variance += np.abs(sens * sigma_p)**2
        
    return np.sqrt(variance)

        
