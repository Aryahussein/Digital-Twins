import numpy as np
from scipy.stats import norm

def _find_component_for_param(circuit, param):
    """Safely finds the component associated with a parameter string."""
    sorted_comps = sorted(circuit.components, key=lambda c: len(c.name), reverse=True)
    for comp in sorted_comps:
        if param == comp.name or param.startswith(comp.name + "_"):
            return comp
    return None

def _get_nominal_values(circuit, param_names):
    """Extracts nominal values by delegating directly to the components."""
    p_vals = []
    for param in param_names:
        comp = _find_component_for_param(circuit, param)
        if not comp:
            raise ValueError(f"FATAL: Component for parameter '{param}' not found in circuit.")
        p_vals.append(comp.get_nominal_value(param))
    return np.array(p_vals)

def analyze_and_rank_sensitivities_globally(circuit, result, target_node, candidate_params, k=3, tolerance_pct=0.05):
    sensitivities = result.sensitivities
    o_idx = sensitivities.output_index[target_node]
    p_noms = _get_nominal_values(circuit, candidate_params)

    full_ranking = []
    
    num_steps = len(result.sweep_axis)
    max_individual_dv_waveform = np.zeros(num_steps)
    total_variance_waveform = np.zeros(num_steps)

    for param, p_nom in zip(candidate_params, p_noms):
        if p_nom == 0: continue
        comp_name = param.split('_')[0].upper()
        if comp_name.startswith('V') or comp_name.startswith('I'): continue
            
        p_idx = sensitivities.param_index[param]
        raw_sens_waveform = sensitivities.data[p_idx, o_idx, :]
        
        # dV/dp * delta_p
        expected_dv_waveform = np.abs(raw_sens_waveform * p_nom * tolerance_pct)
        
        # Build Metric 2 (Sum of Variances)
        total_variance_waveform += expected_dv_waveform**2
        
        # Build Metric 1 (Max Individual Impact)
        mask = expected_dv_waveform > max_individual_dv_waveform
        max_individual_dv_waveform[mask] = expected_dv_waveform[mask]
        
        max_dv_expected = np.max(expected_dv_waveform)
        
        full_ranking.append({
            'param': param, 
            'dv_expected': max_dv_expected,
            'rel_sens': np.max(np.abs(raw_sens_waveform * p_nom)) 
        })
        
    full_ranking.sort(key=lambda x: x['dv_expected'], reverse=True)
    k_actual = min(k, len(full_ranking))
    top_params = [data['param'] for data in full_ranking[:k_actual]]
    
    # Extract the exact time steps for the metrics
    step_metric1 = int(np.argmax(max_individual_dv_waveform))
    step_metric2 = int(np.argmax(total_variance_waveform))
    
    return top_params, full_ranking, step_metric1, step_metric2


def generate_worst_case_deltas(circuit, result, target_node, alpha_array, top_params, eval_step=None):
    """
    Generates the Delta P matrix based strictly on the sensitivity direction.
    If eval_step is None, it dynamically finds the global worst-case direction.
    """
    sensitivities = result.sensitivities
    o_idx = sensitivities.output_index[target_node]
    
    S_raw = np.zeros(len(top_params))
    for i, p in enumerate(top_params):
        p_idx = sensitivities.param_index[p]
        
        if eval_step is not None:
            # We know the exact step we care about
            S_raw[i] = sensitivities.data[p_idx, o_idx, eval_step]
        else:
            # Scouting Mode: Find the time index where THIS specific parameter 
            # had its absolute maximum impact, and grab the sensitivity there.
            idx_max = np.argmax(np.abs(sensitivities.data[p_idx, o_idx, :]))
            S_raw[i] = sensitivities.data[p_idx, o_idx, idx_max]
            
    p_nom = _get_nominal_values(circuit, top_params)
    S_norm = S_raw * p_nom
    
    weights = np.sign(S_norm) 
    weights[weights == 0] = 1.0 
    dp_matrix = alpha_array[:, None] * weights[None, :] * p_nom[None, :]
    
    return dp_matrix

def calculate_analytical_yield(circuit, result, target_node, step_idx, spec_min, spec_max, tolerance_pct=0.10, sigma_level=3):
    """Projects component PDFs onto the output node evaluated exactly at the step index."""
    sensitivities = result.sensitivities
    o_idx = sensitivities.output_index[target_node]
    param_names = sensitivities.param_names
    
    S_raw = np.zeros(len(param_names))
    for i in range(len(param_names)):
        S_raw[i] = sensitivities.data[i, o_idx, step_idx]
    
    p_nom = _get_nominal_values(circuit, param_names)
    p_sigma = (p_nom * tolerance_pct) / sigma_level
    
    variance_out = np.sum((S_raw * p_sigma)**2)
    sigma_out = np.sqrt(variance_out)
    
    mean_out = result.VI[step_idx][result.node_map[target_node]]
    
    prob_passing = norm.cdf(spec_max, loc=mean_out, scale=sigma_out) - norm.cdf(spec_min, loc=mean_out, scale=sigma_out)
    yield_pct = prob_passing * 100.0
    dpmo = (1.0 - prob_passing) * 1_000_000
    
    print(f"\n--- Analytical Yield Estimation ---")
    print(f"Output Mean:  {mean_out:.4f} V")
    print(f"Output Sigma: {sigma_out:.4e} V")
    print(f"Spec Window:  [{spec_min:.4f} V, {spec_max:.4f} V]")
    print(f"Yield:        {yield_pct:.6f}%")
    print(f"DPMO:         {dpmo:.2f}")

    return mean_out, sigma_out, yield_pct

def calculate_large_change_yield(circuit, base_result, target_node, step_idx, top_params, spec_min, spec_max, tolerance_pct=0.10, sigma_level=3):
    """
    Projects component PDFs onto the output node using EXACT non-linear voltage shifts.
    Architecturally pure: Evaluates each parameter in isolation and returns a dict of distinct data vaults.
    """
    from engines.large_change_engine import LargeChangeEngine
    
    v_nom = base_result.VI[step_idx][base_result.node_map[target_node]]
    actual_delta_Vs = np.zeros(len(top_params))
    method = getattr(base_result, "method", "TR") if base_result.analysis_type == ".TRAN" else "TR"
    
    # Dictionary to hold the independent data vaults for each parameter
    lc_vaults = {}
    woodbury_deltas = {}
    
    print("\n--- Executing Isolated Woodbury +1\u03c3 Jumps ---")
    
    for i, param in enumerate(top_params):
        # 1. Calculate the exact +1 sigma shift for THIS parameter only
        p_nom = _get_nominal_values(circuit, [param])[0]
        p_sigma = (p_nom * tolerance_pct) / sigma_level
        
        # 2. Build a localized 1x1 sweep matrix (One parameter, One variation)
        dp_matrix = np.array([[p_sigma]])
        
        # 3. Fire the engine for JUST this parameter
        # The variation axis is cleanly set to [1.0] representing 1 standard deviation
        engine = LargeChangeEngine(circuit)
        
        # Temporarily suppress the engine's internal print statement to keep the console clean
        import sys, os
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            lc_result = engine.compute(
                result=base_result, 
                param_names=[param], 
                dp_matrix=dp_matrix,
                variation_axis=np.array([1.0]),  # A true, pure numerical axis!
                method=method
            )
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
            
        # Store the dedicated vault
        lc_vaults[param] = lc_result
        
        # 4. Extract the voltage shift at the exact evaluation step
        # Since there's only 1 variation in this vault, var_idx is 0
        v_shifted = np.real(lc_result(target_node, var_idx=0, step_idx=step_idx))
        actual_delta_Vs[i] = v_shifted - v_nom
        woodbury_deltas[param] = actual_delta_Vs[i]
        # # ===== INJECT THIS DEBUG BLOCK =====
        # print(f"  [DEBUG Woodbury Jump - {param}]")
        # print(f"    Nominal V: {v_nom:.6f} V")
        # print(f"    Shifted V: {v_shifted:.6f} V")
        # print(f"    Delta V:   {actual_delta_Vs[i]*1000:.2f} mV")
        # # ===================================
        
        print(f"  {param:<10}: +1\u03c3 \u0394V = {actual_delta_Vs[i]:+.4e} V")

    # 5. Non-Linear Propagation of Variance
    variance_out = np.sum(actual_delta_Vs**2)
    sigma_out = np.sqrt(variance_out)
    
    # 6. Calculate Yield Percentage using the CDF
    prob_passing = norm.cdf(spec_max, loc=v_nom, scale=sigma_out) - norm.cdf(spec_min, loc=v_nom, scale=sigma_out)
    yield_pct = prob_passing * 100.0
    dpmo = (1.0 - prob_passing) * 1_000_000
    
    print(f"\n--- LARGE CHANGE Analytical Yield Estimation ---")
    print(f"Output Mean:  {v_nom:.4f} V")
    print(f"Output Sigma: {sigma_out:.4e} V")
    print(f"Spec Window:  [{spec_min:.4f} V, {spec_max:.4f} V]")
    print(f"Yield:        {yield_pct:.6f}%")
    print(f"DPMO:         {dpmo:.2f}")

    # Return the dictionary of vaults so they can be plotted individually if desired!
    return v_nom, sigma_out, yield_pct, lc_vaults, woodbury_deltas
