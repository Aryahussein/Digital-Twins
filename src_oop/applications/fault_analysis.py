import numpy as np

def rank_component_sensitivities(result):
    """
    Rank order component sensitivities from largest to smallest.
    Uses the data already stored in the SensitivityCube.
    """
    cube = result.sensitivities.get("Sensitivity_Cube")
    if not cube:
        print("No Sensitivity Cube found. Run simulation with sensitivity=True.")
        return

    rank_list = []

    #Iterate through every parameter stored in the cube
    for p_name in cube.param_names:
        #Get the time-series waveform for this parameter
        for o_node in cube.output_nodes:
            s_waveform = cube.get_time_series(p_name, o_node)
            
            #Use the maximum absolute value as the 'Ranking Metric'
            #This identifies the "Peak Impact" this component has on the output
            # Find the peak and the time it happened
            peak_idx = np.argmax(np.abs(s_waveform))
            impact = s_waveform[peak_idx] # Keep the sign (+/-)
            peak_time = cube.time_array[peak_idx]
            
            rank_list.append({
                "name": p_name,
                "output": o_node,
                "impact": impact,
                "time": peak_time
            })

    #Sort the list: Largest to Smallest
    ranked = sorted(rank_list, key=lambda x: abs(x['impact']), reverse=True)

    print(f"\n{'Rank':<5} | {'Parameter':<12} | {'Impact':<12} | {'At Time':<10} | {'Target'}")
    print("-" * 65)
    for i, item in enumerate(ranked[:10]):
        print(f"{i+1:<5} | {item['name']:<12} | {item['impact']:>10.3e} | {item['time']:>8.2e}s | {item['output']}")

    return ranked


def perform_global_ranking(circuit, result):
    cube = result.sensitivities["Sensitivity_Cube"]
    vi_nom = result.VI
    psi_nom = cube.adjoint_vectors 
    
    # Filter: identify branch current entries by checking component types
    branch_names = set()
    for comp in circuit.components:
        if comp.type in ["V", "L", "H", "E"]:
            branch_names.add(comp.name)
    
    # Keep only physical voltage nodes
    voltage_nodes = [(name, idx) for name, idx in result.node_map.items()
                     if name not in branch_names]
    
    # We will return a dictionary: { 'node_name': [ranked_faults_list] }
    all_rankings = {}

    # Iterate through every output node you specified in main.py
    for out_idx, out_node_name in enumerate(cube.output_nodes):
        node_faults = []

        # --- 1. Rank Existing Components (Opens) for THIS node ---
        for p_name in cube.param_names:
            s_waveform = cube.get_time_series(p_name, out_node_name)
            peak_idx = np.argmax(np.abs(s_waveform))
            node_faults.append({
                "type": "Open", 
                "location": p_name, 
                "impact": s_waveform[peak_idx], 
                "time": cube.time_array[peak_idx]
            })

        # --- 2. Rank Node-Pair Shorts for THIS node ---
        for i in range(len(voltage_nodes)):
            for j in range(i + 1, len(voltage_nodes)):
                n1, idx1 = voltage_nodes[i]
                n2, idx2 = voltage_nodes[j]
                
                v_diff = vi_nom[:, idx1] - vi_nom[:, idx2]
                psi_diff = psi_nom[out_idx, :, idx1] - psi_nom[out_idx, :, idx2] 
                
                short_sens_waveform = v_diff * psi_diff
                
                peak_idx = np.argmax(np.abs(short_sens_waveform))
                node_faults.append({
                    "type": "Short", 
                    "location": f"{n1}<->{n2}", 
                    "impact": short_sens_waveform[peak_idx],
                    "time": cube.time_array[peak_idx]
                })

        # --- 3. Final Sort for THIS node ---
        node_faults.sort(key=lambda x: abs(x["impact"]), reverse=True)
        all_rankings[out_node_name] = node_faults

    return all_rankings