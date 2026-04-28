import numpy as np

def get_top_k_parameters(circuit, sensitivities, target_node, k=3):
    """Ranks and extracts the most influential parameters using NORMALIZED sensitivity."""
    o_idx = sensitivities.output_index[target_node]
    
    ranking = []
    for p_idx, param in enumerate(sensitivities.param_names):
        # 1. Raw sensitivity array
        raw_sens = sensitivities.data[p_idx, o_idx, :]
        
        # 2. Extract nominal value via OOP delegation
        comp_name = param.split("_")[0]
        comp = next(c for c in circuit.components if c.name == comp_name)
        p_nom = comp.get_nominal_value(param)
        
        # 3. Normalized sensitivity (Volts per 100% change)
        norm_sens = raw_sens * p_nom
        max_sens = np.max(np.abs(norm_sens))
        
        ranking.append((param, max_sens))
        
    # Sort descending
    ranking.sort(key=lambda x: x[1], reverse=True)
    top_k = [r[0] for r in ranking[:k]]
    
    print(f"\n--- Top {k} Sensitive Parameters for '{target_node}' (Normalized) ---")
    for p, val in ranking[:k]:
        print(f"  {p}: Max Impact = {val:.4e} V")
        
    return top_k

def _get_nominal_values(circuit, param_names):
    """Helper to safely extract nominal physical values via OOP delegation."""
    p_nom = np.zeros(len(param_names))
    for i, param in enumerate(param_names):
        comp_name = param.split("_")[0]
        comp = next(c for c in circuit.components if c.name == comp_name)
        p_nom[i] = comp.get_nominal_value(param)
    return p_nom

def generate_worst_case_deltas(circuit, sensitivities, target_node, alpha_array, k=3):
    """Generates Delta P matrix for Worst-Case Corner Analysis (Uniform scaling)."""
    top_params = get_top_k_parameters(circuit, sensitivities, target_node, k)
    o_idx = sensitivities.output_index[target_node]
    
    # Calculate Mean Raw Sensitivities across the sweep
    S_raw = np.array([np.mean(sensitivities.data[sensitivities.param_index[p], o_idx, :]) for p in top_params])
    p_nom = _get_nominal_values(circuit, top_params)
    
    # Worst-Case Weights (Direction only: +1.0 or -1.0)
    S_norm = S_raw * p_nom
    weights = np.sign(S_norm) 
    
    # Build Delta P Matrix: (Alphas, k)
    dp_matrix = alpha_array[:, None] * weights[None, :] * p_nom[None, :]
    return top_params, dp_matrix
