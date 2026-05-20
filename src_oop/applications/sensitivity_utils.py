import numpy as np
from scipy.stats import norm

def _find_component_for_param(circuit, param):
    sorted_comps = sorted(circuit.components, key=lambda c: len(c.name), reverse=True)
    for comp in sorted_comps:
        if param == comp.name or param.startswith(comp.name + "_"):
            return comp
    return None

def _get_nominal_values(circuit, param_names):
    p_vals = []
    for param in param_names:
        comp = _find_component_for_param(circuit, param)
        if not comp:
            raise ValueError(f"FATAL: Component for parameter '{param}' not found in circuit.")
        p_vals.append(comp.get_nominal_value(param))
    return np.array(p_vals)

def _get_param_specs(circuit, param_name):
    comp = circuit.param_to_component_map.get(param_name)
    param_base = param_name.split('_')[-1] 
    
    if comp and "stat_params" in comp.data:
        if param_base in comp.data["stat_params"]:
            stats = comp.data["stat_params"][param_base]
            return stats["tol"], stats["sigma"]
            
        if "VALUE" in comp.data["stat_params"]:
            stats = comp.data["stat_params"]["VALUE"]
            return stats["tol"], stats["sigma"]
            
    return 0.0, 3.0


def analyze_and_rank_sensitivities_globally(circuit, result, target_node, candidate_params, k=3):
    sensitivities = result.sensitivities
    o_idx = sensitivities.output_index[target_node]
    p_noms = _get_nominal_values(circuit, candidate_params)

    full_ranking = []
    num_steps = len(result.sweep_axis)
    max_individual_dv_waveform = np.zeros(num_steps)
    total_variance_waveform = np.zeros(num_steps)
    all_dv_waveforms = {}

    for param, p_nom in zip(candidate_params, p_noms):
        if p_nom == 0: continue
            
        p_idx = sensitivities.param_index[param]
        raw_sens_waveform = sensitivities.data[p_idx, o_idx, :]
        p_tol, p_sig_level = _get_param_specs(circuit, param)
        
        # Calculate TRUE 1-Sigma Impact: S_raw * (p_nom * tol / sigma_level)
        p_sigma = (p_nom * p_tol) / p_sig_level
        expected_dv_waveform = np.abs(raw_sens_waveform * p_sigma)
        
        all_dv_waveforms[param] = expected_dv_waveform
        total_variance_waveform += expected_dv_waveform**2
        
        mask = expected_dv_waveform > max_individual_dv_waveform
        max_individual_dv_waveform[mask] = expected_dv_waveform[mask]
        
        full_ranking.append({
            'param': param, 
            'dv_expected': np.max(expected_dv_waveform),
            'rel_sens': np.max(np.abs(raw_sens_waveform * p_nom)) 
        })
        
    full_ranking.sort(key=lambda x: x['dv_expected'], reverse=True)
    k_actual = min(k, len(full_ranking))
    top_params = [data['param'] for data in full_ranking[:k_actual]]
    
    sim_start = 1 if getattr(result, "analysis_type", "") == ".TRAN" else 0
    step_metric1 = int(np.argmax(max_individual_dv_waveform[sim_start:])) + sim_start
    step_metric2 = int(np.argmax(total_variance_waveform[sim_start:])) + sim_start
    
    return top_params, full_ranking, step_metric1, step_metric2, max_individual_dv_waveform, total_variance_waveform, all_dv_waveforms

def generate_worst_case_deltas(circuit, result, target_node, alpha_array, top_params, eval_step=None, use_1sigma=False):
    sensitivities = result.sensitivities
    o_idx = sensitivities.output_index[target_node]
    
    S_raw = np.zeros(len(top_params))
    for i, p in enumerate(top_params):
        p_idx = sensitivities.param_index[p]
        if eval_step is not None:
            S_raw[i] = sensitivities.data[p_idx, o_idx, eval_step]
        else:
            idx_max = np.argmax(np.abs(sensitivities.data[p_idx, o_idx, :]))
            S_raw[i] = sensitivities.data[p_idx, o_idx, idx_max]
            
    p_nom = _get_nominal_values(circuit, top_params)
    S_norm = S_raw * p_nom
    
    weights = np.sign(S_norm) 
    weights[weights == 0] = 1.0 
    
    param_tols = []
    for p in top_params:
        tol, sig = _get_param_specs(circuit, p)
        if use_1sigma and sig > 0:
            param_tols.append(tol / sig) 
        else:
            param_tols.append(tol)       
            
    param_tols = np.array(param_tols)
    dp_matrix = alpha_array[:, None] * weights[None, :] * param_tols[None, :] * p_nom[None, :]
    
    return dp_matrix


def calculate_analytical_yield(circuit, result, target_node, step_idx, spec_min, spec_max):
    sensitivities = result.sensitivities
    o_idx = sensitivities.output_index[target_node]
    param_names = sensitivities.param_names
    
    S_raw = np.zeros(len(param_names))
    p_sigma = np.zeros(len(param_names))
    
    for i, p in enumerate(param_names):
        S_raw[i] = sensitivities.data[i, o_idx, step_idx]
        p_nom = _get_nominal_values(circuit, [p])[0]
        p_tol, p_sig = _get_param_specs(circuit, p)
        p_sigma[i] = (p_nom * p_tol) / p_sig
    
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


def calculate_large_change_yield(circuit, base_result, target_node, step_idx, top_params, spec_min, spec_max):
    from engines.large_change_engine import LargeChangeEngine
    
    v_nom = base_result.VI[step_idx][base_result.node_map[target_node]]
    actual_delta_Vs = np.zeros(len(top_params))
    method = getattr(base_result, "method", "TR") if base_result.analysis_type == ".TRAN" else "TR"
    
    lc_vaults = {}
    woodbury_deltas = {}
    
    print("\n--- Executing Isolated Woodbury Worst-Case Jumps ---")
    
    for i, param in enumerate(top_params):
        p_nom = _get_nominal_values(circuit, [param])[0]
        p_tol, p_sigma_level = _get_param_specs(circuit, param)
        
        p_wc_delta = p_nom * p_tol
        dp_matrix = np.array([[p_wc_delta]])
        
        engine = LargeChangeEngine(circuit)
        
        # Unsuppressed evaluation
        lc_result = engine.compute(
            result=base_result, 
            param_names=[param], 
            dp_matrix=dp_matrix,
            variation_axis=np.array([1.0]), 
            method=method
        )
            
        lc_vaults[param] = lc_result
        
        v_shifted_wc = np.real(lc_result(target_node, var_idx=0, step_idx=step_idx))
        delta_v_wc = v_shifted_wc - v_nom
        
        actual_delta_Vs[i] = delta_v_wc / p_sigma_level
        woodbury_deltas[param] = actual_delta_Vs[i]
        
        print(f"  {param:<10}: ΔV_wc = {delta_v_wc:+.4e} V -> Effective 1σ ΔV = {actual_delta_Vs[i]:+.4e} V")

    variance_out = np.sum(actual_delta_Vs**2)
    sigma_out = np.sqrt(variance_out)
    
    prob_passing = norm.cdf(spec_max, loc=v_nom, scale=sigma_out) - norm.cdf(spec_min, loc=v_nom, scale=sigma_out)
    yield_pct = prob_passing * 100.0
    dpmo = (1.0 - prob_passing) * 1_000_000
    
    print(f"\n--- LARGE CHANGE Analytical Yield Estimation ---")
    print(f"Output Mean:  {v_nom:.4f} V")
    print(f"Output Sigma: {sigma_out:.4e} V")
    print(f"Spec Window:  [{spec_min:.4f} V, {spec_max:.4f} V]")
    print(f"Yield:        {yield_pct:.6f}%")
    print(f"DPMO:         {dpmo:.2f}")

    return v_nom, sigma_out, yield_pct, lc_vaults, woodbury_deltas
