"""
Provides functions for ranking component sensitivities 
and generating fault tables (opens and shorts) using adjoint sensitivity 
data from the simulator.

Short faults: R_short = dVout/ (v_nom * psi_nom)
    where v_nom is the voltage difference between shorting nodes in the 
    original circuit, and psi_nom is the voltage difference in the adjoint circuit.

Open faults: R_open = dVout / (i_nom * phi_nom)
    where i_nom is the branch current in the original circuit, and phi_nom 
    is the branch current in the adjoint circuit.
"""

import numpy as np

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
            
            # Use the maximum absolute value as the ranking metric
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

    # Print the table
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
    psi_nom = tensor.adjoint_vectors  # Shape: (n_outputs, n_steps, total_dim)
    
    # Check that adjoint vectors were stored
    if psi_nom is None:
        print("No raw adjoint vectors found. Cannot compute short faults.")
        use_adjoint = False
    else:
        use_adjoint = True
    
    # Filter out branch current entries to keep only physical voltage nodes
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

        # --- 1. Rank Existing Components/Branches (Open Faults) ---
        # An open fault means a component's resistance goes very high.
        # The sensitivity waveform tells us how much the output changes
        # per unit change in that component's parameter.
        for p_name in tensor.param_names:
            s_waveform = tensor.get_sweep_series(p_name, out_node)
            peak_idx = np.argmax(np.abs(s_waveform))
            node_faults.append({
                "type": "Open", 
                "location": p_name, 
                "impact": s_waveform[peak_idx], 
                "time": tensor.sweep_axis[peak_idx]
            })

        # --- 2. Rank Node-Pair Shorts ---
        # v_diff = voltage difference between the two nodes in the original circuit
        # psi_diff = voltage difference between the two nodes in the adjoint circuit
        for i in range(len(voltage_nodes)):
            for j in range(i + 1, len(voltage_nodes)):
                n1, idx1 = voltage_nodes[i]
                n2, idx2 = voltage_nodes[j]
                
                # Forward voltage difference across the potential short
                v_diff = vi_nom[:, idx1] - vi_nom[:, idx2]
                
                if use_adjoint:
                    # Adjoint voltage difference (psi) for this output node
                    psi_diff = psi_nom[o_idx, :, idx1] - psi_nom[o_idx, :, idx2]
                    
                    short_sens_waveform = v_diff * psi_diff
                
                peak_idx = np.argmax(np.abs(short_sens_waveform))
                node_faults.append({
                    "type": "Short", 
                    "location": f"{n1}<->{n2}", 
                    "impact": short_sens_waveform[peak_idx],
                    "time": tensor.sweep_axis[peak_idx]
                })

        # --- 3. Sort by absolute impact ---
        node_faults.sort(key=lambda x: abs(x["impact"]), reverse=True)
        all_rankings[out_node] = node_faults

    return all_rankings


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