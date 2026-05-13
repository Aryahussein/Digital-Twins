import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
from engines.large_change_engine import LargeChangeEngine
from utils.plotting import plot_combined_yield_pdf, plot_transient_envelope ,plot_unified_pareto
from applications.sensitivity_utils import (
    generate_worst_case_deltas, 
    calculate_analytical_yield, calculate_large_change_yield, analyze_and_rank_sensitivities_globally, _get_nominal_values, _get_param_specs
)

def evaluate_deep_yield_at_step(
    circuit, result, target_node, step_idx, top_params, full_ranking, 
    dynamic_min, dynamic_max, factory_tol, manufacturing_sigma, 
    out_sigma_req, max_allowable_sigma, folder, base_name, metric_label, env_min_v=None, env_max_v=None
):
    print(f"\n[{metric_label}] Evaluating Deep Yield at Step {step_idx} (t={result.sweep_axis[step_idx]*1e9:.2f}ns)")
    
    # 1. Calculate LC Yield
    mean_lc, sigma_lc, yield_lc, _, woodbury_deltas = calculate_large_change_yield(
        circuit=circuit, base_result=result, target_node=target_node, 
        step_idx=step_idx, top_params=top_params, spec_min=dynamic_min, 
        spec_max=dynamic_max, tolerance_pct=factory_tol, sigma_level=manufacturing_sigma
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
        full_envelope_dev = max(abs(env_max_v - v_nom), abs(env_min_v - v_nom))
        combined_woodbury_v = full_envelope_dev / manufacturing_sigma
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
    
    # CRITICAL: Return the woodbury_deltas!
    return mean_lc, sigma_lc, woodbury_deltas


def perform_sdwc_yield_analysis(
    circuit, result, target_node, calculated_params, netlist_name, 
    folder_map, factory_tolerance=0.05, sigma_level=6,
    out_spec=0.05, out_sigma_level=6, k_params=3
):
    out_folder = "../figures/yield"
    os.makedirs(out_folder, exist_ok=True)
    method = getattr(result, "method", "TR") if result.analysis_type == ".TRAN" else "TR"

    top_params, full_ranking, step_m1, step_m2 = analyze_and_rank_sensitivities_globally(
        circuit=circuit, result=result, target_node=target_node, 
        candidate_params=calculated_params, k=k_params, tolerance_pct=factory_tolerance
    )
    print(f"\nSweeping top {len(top_params)} parameters: {top_params}")

    print("\n--- Phase B: Scouting true physical envelope with Woodbury ---")
    engine = LargeChangeEngine(circuit)
    
    corner_alphas = np.array([-1.0, 1.0]) 
    dp_matrix_corners = generate_worst_case_deltas(
        circuit, result, target_node, corner_alphas, top_params, eval_step=None 
    )
    
    lc_envelope = engine.compute(
        result=result, param_names=top_params, dp_matrix=dp_matrix_corners,
        variation_axis=corner_alphas, method=method
    )

    n_idx = result.node_map[target_node]
    all_waveforms = lc_envelope.data[:, :, n_idx]
    v_min = np.min(all_waveforms, axis=0)
    v_max = np.max(all_waveforms, axis=0)

    if result.analysis_type == ".TRAN":
        v_nom = result.VI[:, n_idx]
        slew_rate = np.abs(np.gradient(v_nom))
        max_slew = np.max(slew_rate) if np.max(slew_rate) > 0 else 1.0
        active_mask = (slew_rate / max_slew) > 0.01
        
        if not np.any(active_mask):
            active_mask = np.ones(len(v_nom), dtype=bool)
            
        # Only look for the maximum physical spread during a transition
        masked_spread = np.abs(v_max - v_min) * active_mask
        step_m3 = int(np.argmax(masked_spread))
    else:
        step_m3 = int(np.argmax(np.abs(v_max - v_min)))

    # sim_start = 0
    # if getattr(result, "analysis_type", "") == ".TRAN":
    #     sim_start = 2

    # step_m3 = int(np.argmax(np.abs(v_max - v_min)[sim_start:])) + sim_start


    evaluation_steps = {}
    def add_step(step, label):
        if step not in evaluation_steps: evaluation_steps[step] = []
        evaluation_steps[step].append(label)

    add_step(step_m1, "M1_Max_Individual_Sens")
    add_step(step_m2, "M2_Max_Total_Variance")
    add_step(step_m3, "M3_Max_Physical_Spread")

    all_woodbury_deltas = {} # <--- CAPTURE ALL

    for step, labels in evaluation_steps.items():
        combined_label = "+".join(labels)
        
        v_nom = result.VI[step][n_idx]
        dynamic_min = v_nom * (1.0 - out_spec)
        dynamic_max = v_nom * (1.0 + out_spec)
        max_allow_sigma = (v_nom * out_spec) / out_sigma_level

        _, _, current_deltas = evaluate_deep_yield_at_step(
            circuit=circuit, result=result, target_node=target_node, 
            step_idx=step, top_params=top_params, full_ranking=full_ranking,
            dynamic_min=dynamic_min, dynamic_max=dynamic_max,
            factory_tol=factory_tolerance, manufacturing_sigma=sigma_level,
            out_sigma_req=out_sigma_level, max_allowable_sigma=max_allow_sigma,
            folder=out_folder, base_name=netlist_name, metric_label=combined_label,
            env_min_v=v_min[step], env_max_v=v_max[step]
        )

        # Map the specific step to its isolated Woodbury jumps
        all_woodbury_deltas[step] = current_deltas

    if result.analysis_type == ".TRAN":
        plot_transient_envelope(
            result, lc_envelope, target_node, evaluation_steps,
            out_folder, f"{netlist_name}_master_tran_envelope"
        )

    # Send EVERYTHING to the GUI
    return {
        "envelope": lc_envelope,
        "evaluation_steps": evaluation_steps,       # <--- Pass step names
        "all_woodbury_deltas": all_woodbury_deltas, # <--- Pass all deltas
        "default_step": step_m3,                    # <--- Where to start
        "v_min": v_min,
        "v_max": v_max,
        "top_params": top_params,
        "full_ranking": full_ranking,
        "orig_tol": factory_tolerance,
        "orig_sigma": sigma_level
    }
