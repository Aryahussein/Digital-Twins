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

def analyze_and_rank_sensitivities(circuit, result, target_node, candidate_params, eval_step=None, k=3, tolerance_pct=0.05):
    """
    DRY Unified Engine: 
    1. Finds the global worst-case evaluation step (ignoring static power supplies).
    2. Ranks all parameters based on their impact at that specific step.
    """
    sensitivities = result.sensitivities
    o_idx = sensitivities.output_index[target_node]
    
    # Pre-fetch all nominal values once
    p_noms = _get_nominal_values(circuit, candidate_params)

    # ========================================================
    # Phase 1: Determine Evaluation Step (If not user-forced)
    # ========================================================
    if eval_step is None:
        global_max_impact = -1.0
        driving_param = None
        
        for param, p_nom in zip(candidate_params, p_noms):
            if p_nom == 0: continue
            
            # --- THE FIX: Ignore Voltage/Current sources for determining the switching step ---
            # SPICE convention: Voltage sources start with 'V', Current with 'I'
            comp_name = param.split('_')[0].upper()
            if comp_name.startswith('V') or comp_name.startswith('I'):
                continue
            # --------------------------------------------------------------------------------
                
            p_idx = sensitivities.param_index[param]
            raw_sens = sensitivities.data[p_idx, o_idx, :]
            
            # Normalize to Volts
            rel_sens_waveform = np.abs(raw_sens * p_nom)
            
            local_max_val = np.max(rel_sens_waveform)
            if local_max_val > global_max_impact:
                global_max_impact = local_max_val
                eval_step = int(np.argmax(rel_sens_waveform))
                driving_param = param
                
        # Fallback just in case only VDD was passed
        if eval_step is None:
            eval_step = 0
            
        print(f"Evaluating Yield at MAXIMUM Relative Sensitivity (Step {eval_step}, driven by {driving_param})")
    else:
        print(f"Evaluating Yield at User-Specified Step: {eval_step}")

    # ========================================================
    # Phase 2: Rank Parameters exactly at the Evaluation Step
    # ========================================================
    ranking = []
    for param, p_nom in zip(candidate_params, p_noms):
        if p_nom == 0: continue
        
        p_idx = sensitivities.param_index[param]
        dv_dp_eval = sensitivities.data[p_idx, o_idx, eval_step]
        
        rel_sens = dv_dp_eval * p_nom
        dv_expected = np.abs(rel_sens * tolerance_pct)
        
        ranking.append({'param': param, 'rel_sens': rel_sens, 'dv_expected': dv_expected})
        
    ranking.sort(key=lambda x: x['dv_expected'], reverse=True)
    k_actual = min(k, len(ranking))
    top_k_data = ranking[:k_actual]
    
    print(f"\n--- Top {k_actual} Sensitive Parameters for V({target_node}) at Step {eval_step} ---")
    print(f"{'Parameter':<12} | {'Rel Sens (V/100%)':<18} | {'Expected \u0394V (@ ' + str(int(tolerance_pct*100)) + '%)':<20}")
    print("-" * 75)
    for data in top_k_data:
        print(f"  {data['param']:<10} | {data['rel_sens']:>16.4e} V | {data['dv_expected']:>16.4e} V")
        
    return eval_step, [data['param'] for data in top_k_data]


def generate_worst_case_deltas(circuit, result, target_node, alpha_array, top_params, eval_step):
    """Generates the Delta P matrix based strictly on the sensitivity direction at the evaluation step."""
    sensitivities = result.sensitivities
    o_idx = sensitivities.output_index[target_node]
    
    S_raw = np.zeros(len(top_params))
    for i, p in enumerate(top_params):
        p_idx = sensitivities.param_index[p]
        S_raw[i] = sensitivities.data[p_idx, o_idx, eval_step]
        
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
    return v_nom, sigma_out, yield_pct, lc_vaults
