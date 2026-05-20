import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
from engines.large_change_engine import LargeChangeEngine
from utils.plotting import (
    plot_combined_yield_pdf, 
    plot_transient_envelope, 
    plot_unified_pareto,
    plot_individual_sensitivities, 
    plot_sigma_metrics             
)
from applications.sensitivity_utils import (
    generate_worst_case_deltas, 
    calculate_analytical_yield, 
    calculate_large_change_yield, 
    analyze_and_rank_sensitivities_globally, 
    _get_nominal_values, 
    _get_param_specs
)

def evaluate_deep_yield_at_step(
    circuit, result, target_node, step_idx, top_params, full_ranking, 
    dynamic_min, dynamic_max,
    out_sigma_req, max_allowable_sigma, folder, base_name, metric_label, env_min_v=None, env_max_v=None
):
    print(f"\n[{metric_label}] Evaluating Deep Yield at Step {step_idx} (t={result.sweep_axis[step_idx]*1e9:.2f}ns)")
    
    # 1. Calculate LC Yield 
    mean_lc, sigma_lc, yield_lc, _, woodbury_deltas = calculate_large_change_yield(
        circuit=circuit, base_result=result, target_node=target_node, 
        step_idx=step_idx, top_params=top_params, spec_min=dynamic_min, 
        spec_max=dynamic_max
    )

    # 2. Rigorous Spec Check
    print(f"  Target Max Sigma ({out_sigma_req}-Sigma): {max_allowable_sigma:.4e} V")
    print(f"  Actual Circuit Sigma:     {sigma_lc:.4e} V")
    if sigma_lc <= max_allowable_sigma:
        print("  RESULT: PASS (Circuit meets rigorous Yield Spec!)")
    else:
        print("  RESULT: FAIL (Circuit variance is too high)")

    # =========================================================================
    # Generate a localized, 1-Sigma Adjoint ranking for THIS step
    # =========================================================================
    step_specific_ranking = []
    sensitivities = result.sensitivities
    o_idx = sensitivities.output_index[target_node]
    p_noms = _get_nominal_values(circuit, [d['param'] for d in full_ranking])

    for d, p_nom in zip(full_ranking, p_noms):
        param = d['param']
        p_idx = sensitivities.param_index[param]
        raw_sens = sensitivities.data[p_idx, o_idx, step_idx]
        
        p_tol, p_sigma_level = _get_param_specs(circuit, param)
        
        p_sigma = (p_nom * p_tol) / p_sigma_level
        expected_dv = np.abs(raw_sens * p_sigma)
        
        step_specific_ranking.append({
            'param': param,
            'dv_expected': expected_dv,
            'rel_sens': raw_sens * p_nom
        })
        
    step_specific_ranking.sort(key=lambda x: x['dv_expected'], reverse=True)

    combined_adjoint_v = 0.0
    combined_woodbury_v = 0.0
    
    if env_min_v is not None and env_max_v is not None:
        v_nom = result.VI[step_idx][result.node_map[target_node]]
        
        combined_woodbury_v = max(abs(env_max_v - v_nom), abs(env_min_v - v_nom))
        combined_adjoint_v = sum([d['dv_expected'] for d in step_specific_ranking])

    # =========================================================================
    # Generate Plots specifically for this step
    # =========================================================================
    plot_name = f"{base_name}_{metric_label.replace(' ', '_')}"
    
    plot_unified_pareto(
        step_specific_ranking=step_specific_ranking, 
        woodbury_deltas=woodbury_deltas, 
        target_node=target_node, 
        step_idx=step_idx, 
        folder=folder, 
        name=f"{plot_name}_unified_pareto",
        combined_adjoint_v=combined_adjoint_v,
        combined_woodbury_v=combined_woodbury_v
    )
    
    plot_combined_yield_pdf(
        mean_out=mean_lc, sigma_out=sigma_lc, target_node=target_node,
        spec_min=dynamic_min, spec_max=dynamic_max, folder=folder,
        name=f"{plot_name}_pdf", step_idx=step_idx
    )
    
    return mean_lc, sigma_lc, woodbury_deltas

def perform_sdwc_yield_analysis(circuit, result, target_node, calculated_params, netlist_name, folder_map, out_spec=0.05, out_sigma_level=6, k_params=3):
    out_folder = "../figures/yield"; os.makedirs(out_folder, exist_ok=True)
    method = getattr(result, "method", "TR") if result.analysis_type == ".TRAN" else "TR"

    # Phase A: Adjoint Scouting
    top_params, full_ranking, step_m1, step_m2, m1_wave, m2_wave, all_dv_waveforms = analyze_and_rank_sensitivities_globally(
        circuit, result, target_node, calculated_params, k=k_params
    )

    # Phase B: Woodbury Scouting
    engine = LargeChangeEngine(circuit); corner_alphas = np.array([-1.0, 1.0])
    
    # M3: Physical spread
    dp_m3 = generate_worst_case_deltas(circuit, result, target_node, corner_alphas, top_params, use_1sigma=False)
    lc_m3 = engine.compute(result, top_params, dp_m3, corner_alphas, method)
    spread_m3 = np.abs(np.max(lc_m3.data[:,:,result.node_map[target_node]], axis=0) - np.min(lc_m3.data[:,:,result.node_map[target_node]], axis=0))

    # M4: True 1-Sigma spread
    dp_m4 = generate_worst_case_deltas(circuit, result, target_node, corner_alphas, top_params, use_1sigma=True)
    lc_m4 = engine.compute(result, top_params, dp_m4, corner_alphas, method)
    n_idx = result.node_map[target_node]
    v_min_m4, v_max_m4 = np.min(lc_m4.data[:,:,n_idx], axis=0), np.max(lc_m4.data[:,:,n_idx], axis=0)
    spread_m4 = np.abs(v_max_m4 - v_min_m4)

    # Identify Steps
    if result.analysis_type == ".TRAN":
        v_nom = result.VI[:, n_idx]
        active_mask = (np.abs(np.gradient(v_nom)) / (np.max(np.abs(np.gradient(v_nom))) or 1.0)) > 0.01
        step_m3 = int(np.argmax(spread_m3 * active_mask))
        step_m4 = int(np.argmax(spread_m4 * active_mask))
    else:
        step_m3, step_m4 = int(np.argmax(spread_m3)), int(np.argmax(spread_m4))

    # Transparency Plots
    plot_individual_sensitivities(result, all_dv_waveforms, out_folder, f"{netlist_name}_all_sensitivities")
    plot_sigma_metrics(result, m1_wave, m2_wave, spread_m3, spread_m4, step_m1, step_m2, step_m3, step_m4, out_folder, f"{netlist_name}_sigma_metrics")

    # UN-GROUPED EVALUATION LIST
    eval_list = [(step_m1, "M1_Max_Sens"), (step_m2, "M2_Max_Var"), (step_m3, "M3_Max_Phys"), (step_m4, "M4_Max_1Sig")]
    all_woodbury_deltas = {}

    for step, label in eval_list:
        v_nom_step = result.VI[step][n_idx]
        d_min, d_max = v_nom_step * (1.0 - out_spec), v_nom_step * (1.0 + out_spec)
        
        _, _, deltas = evaluate_deep_yield_at_step(
            circuit, result, target_node, step, top_params, full_ranking,
            d_min, d_max, out_sigma_level, (v_nom_step * out_spec)/out_sigma_level,
            out_folder, netlist_name, label, env_min_v=v_min_m4[step], env_max_v=v_max_m4[step]
        )
        all_woodbury_deltas[f"{label}_step_{step}"] = deltas

    plot_transient_envelope(result, lc_m4, target_node, eval_list, out_folder, f"{netlist_name}_master_tran_envelope")

    return {"envelope": lc_m4, "all_woodbury_deltas": all_woodbury_deltas, "v_min": v_min_m4, "v_max": v_max_m4}

