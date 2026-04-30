import numpy as np
from scipy.stats import norm

def _get_nominal_values(circuit, param_names):
    """Helper to safely extract nominal physical values via OOP delegation.
    
    Args:
        circuit (Circuit): The main circuit object containing components.
        param_names (list): List of parameter string names (e.g., ['R1_value', 'M1_W']).
        
    Returns:
        np.ndarray: A 1D array of the nominal physical values.
    """
    p_nom = np.zeros(len(param_names))
    for i, param in enumerate(param_names):
        # Extract the base component name (e.g., 'M1' from 'M1_W')
        comp_name = param.split("_")[0]
        
        # Find the component in the circuit registry
        comp = next((c for c in circuit.components if c.name == comp_name), None)
        if comp is None:
            raise ValueError(f"Component '{comp_name}' not found in circuit.")
            
        p_nom[i] = comp.get_nominal_value(param)
        
    return p_nom


def get_top_k_parameters(circuit, sensitivities, target_node, k=3):
    """Ranks and extracts the most influential parameters using NORMALIZED sensitivity.
    
    Raw sensitivity (dV/dp) is mathematically correct, but practically misleading 
    because a 1-ohm shift in a 1M-ohm resistor is tiny, while a 1-meter shift 
    in a 1-micron transistor is catastrophic. Multiplying by the nominal value 
    normalizes the metric to Volts per 100% parameter change.
    
    Args:
        circuit (Circuit): The main circuit object.
        sensitivities (SensitivityData): The adjoint sensitivity results object.
        target_node (str): The node being analyzed.
        k (int): Number of top parameters to extract.
        
    Returns:
        list: The string names of the top `k` most sensitive parameters.
    """
    o_idx = sensitivities.output_index[target_node]
    
    ranking = []
    for p_idx, param in enumerate(sensitivities.param_names):
        # 1. Raw sensitivity array across the sweep
        raw_sens = sensitivities.data[p_idx, o_idx, :]
        
        # 2. Extract nominal value via OOP delegation
        comp_name = param.split("_")[0]
        comp = next((c for c in circuit.components if c.name == comp_name), None)
        if not comp: continue
        
        p_nom = comp.get_nominal_value(param)
        
        # 3. Normalized sensitivity (Volts per 100% change)
        norm_sens = raw_sens * p_nom
        
        # Use the maximum absolute impact across the sweep to rank it
        max_sens = np.max(np.abs(norm_sens))
        ranking.append((param, max_sens))
        
    # Sort descending based on maximum impact
    ranking.sort(key=lambda x: x[1], reverse=True)
    
    # Cap k at the maximum available parameters
    k_actual = min(k, len(ranking))
    top_k = [r[0] for r in ranking[:k_actual]]
    
    print(f"\n--- Top {k_actual} Sensitive Parameters for V({target_node}) (Normalized) ---")
    for p, val in ranking[:k_actual]:
        print(f"  {p:<15} : Max Impact = {val:.4e} V / 100% \u0394")
        
    return top_k


def generate_worst_case_deltas(circuit, sensitivities, target_node, alpha_array, k=3):
    """Generates the Delta P matrix for Worst-Case Corner Analysis.
    
    Evaluates the mean sensitivity direction of the top `k` components to 
    construct a synchronized variation matrix. If a component's sensitivity 
    is positive, it sweeps in the +alpha direction. If negative, it sweeps 
    in the -alpha direction. This guarantees we hit the absolute worst-case corners.
    
    Args:
        circuit (Circuit): The main circuit object.
        sensitivities (SensitivityData): The adjoint sensitivity results object.
        target_node (str): The node being analyzed.
        alpha_array (np.ndarray): 1D array of variation percentages (e.g., -0.2 to +0.2).
        k (int): Number of top parameters to extract.
        
    Returns:
        tuple: (list of param_names, np.ndarray of shape (len(alpha_array), k) containing physical deltas)
    """
    top_params = get_top_k_parameters(circuit, sensitivities, target_node, k)
    if not top_params:
        return [], np.array([])

    o_idx = sensitivities.output_index[target_node]
    
    # 1. Calculate Mean Raw Sensitivities across the sweep
    S_raw = np.zeros(len(top_params))
    for i, p in enumerate(top_params):
        p_idx = sensitivities.param_index[p]
        S_raw[i] = np.mean(sensitivities.data[p_idx, o_idx, :])
        
    p_nom = _get_nominal_values(circuit, top_params)
    
    # 2. Worst-Case Weights (Direction only: +1.0 or -1.0)
    # We multiply by p_nom just to ensure directionality remains physically aligned
    S_norm = S_raw * p_nom
    
    # np.sign returns 0 if S_norm is exactly 0. We default 0 to +1 to avoid dead sweeps.
    weights = np.sign(S_norm) 
    weights[weights == 0] = 1.0 
    
    # 3. Build Delta P Matrix: Shape -> (len(alpha_array), k)
    # alpha_array[:, None] transforms shape (N,) to (N, 1)
    # weights[None, :] transforms shape (k,) to (1, k)
    # p_nom[None, :] transforms shape (k,) to (1, k)
    # Broadcasting multiplies them into a perfect (N, k) matrix of physical deltas!
    dp_matrix = alpha_array[:, None] * weights[None, :] * p_nom[None, :]
    
    return top_params, dp_matrix

def calculate_analytical_yield(circuit, result, target_node, step_idx, spec_min, spec_max, tolerance_pct=0.10, sigma_level=3):
    """
    Projects component PDFs onto the output node using Propagation of Variance.
    """
    sensitivities = result.sensitivities
    o_idx = sensitivities.output_index[target_node]
    param_names = sensitivities.param_names
    
    # Extract raw sensitivities at the specific time step
    S_raw = np.array([sensitivities.data[i, o_idx, step_idx] for i in range(len(param_names))])
    
    # Extract nominal values
    p_nom = _get_nominal_values(circuit, param_names)
    
    # 1. Calculate the physical standard deviation (sigma) of the components.
    p_sigma = (p_nom * tolerance_pct) / sigma_level
    
    # 2. Propagation of Variance Formula
    variance_out = np.sum((S_raw * p_sigma)**2)
    sigma_out = np.sqrt(variance_out)
    
    # 3. Calculate Yield Percentage using the CDF
    # FIX: Pull the voltage directly from the 'result' object!
    mean_out = result.VI[step_idx][result.node_map[target_node]]
    
    # The probability it falls below the max spec, MINUS the probability it falls below the min spec
    prob_passing = norm.cdf(spec_max, loc=mean_out, scale=sigma_out) - \
                   norm.cdf(spec_min, loc=mean_out, scale=sigma_out)
                   
    yield_pct = prob_passing * 100.0
    dpmo = (1.0 - prob_passing) * 1_000_000
    
    print(f"\n--- Analytical Yield Estimation ---")
    print(f"Output Mean:  {mean_out:.4f} V")
    print(f"Output Sigma: {sigma_out:.4e} V")
    print(f"Spec Window:  [{spec_min:.4f} V, {spec_max:.4f} V]")
    print(f"Yield:        {yield_pct:.6f}%")
    print(f"DPMO:         {dpmo:.2f}")

    return mean_out, sigma_out, yield_pct
