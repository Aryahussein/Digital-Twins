import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
from engines.large_change_engine import LargeChangeEngine
from utils.plotting import plot_combined_yield_pdf, plot_transient_envelope ,plot_unified_pareto
from applications.sensitivity_utils import (
    generate_worst_case_deltas, 
    calculate_analytical_yield, calculate_large_change_yield, analyze_and_rank_sensitivities_globally, _get_nominal_values
)

def evaluate_deep_yield_at_step(
    circuit, result, target_node, step_idx, top_params, full_ranking, 
    dynamic_min, dynamic_max, factory_tol, manufacturing_sigma, 
    out_sigma_req, max_allowable_sigma, folder, base_name, metric_label, env_min_v=None, env_max_v=None
):
    print(f"\n[{metric_label}] Evaluating Deep Yield at Step {step_idx} (t={result.sweep_axis[step_idx]*1e9:.2f}ns)")
    
    # 1. Calculate LC Yield (This internally runs the pure +1σ Woodbury jumps from t=0)
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
        
        # Grab the exact Adjoint derivative for THIS specific time step
        raw_sens = sensitivities.data[p_idx, o_idx, step_idx]
        
        # Calculate the physical 1-Sigma shift
        p_sigma = (p_nom * factory_tol) / manufacturing_sigma
        
        # Expected linear delta V
        expected_dv = np.abs(raw_sens * p_sigma)
        
        step_specific_ranking.append({
            'param': param,
            'dv_expected': expected_dv,
            'rel_sens': raw_sens * p_nom
        })
        
    step_specific_ranking.sort(key=lambda x: x['dv_expected'], reverse=True)

    combined_adjoint_mv = 0.0
    combined_woodbury_mv = 0.0
    
    if env_min_v is not None and env_max_v is not None:
        v_nom = result.VI[step_idx][result.node_map[target_node]]
        
        # Woodbury Truth: The max deviation to the edge of the physical envelope
        combined_woodbury_mv = max(abs(env_max_v - v_nom), abs(env_min_v - v_nom)) * 1000
        
        # Adjoint Prediction: The linear sum of all parameters shifted by 5%
        # (dv_expected is currently 1-sigma. Multiply by sigma_level to get the full 5% factory tol)
        combined_adjoint_mv = sum([d['dv_expected'] * manufacturing_sigma * 1000 for d in step_specific_ranking])

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
        combined_adjoint_mv=combined_adjoint_mv,
        combined_woodbury_mv=combined_woodbury_mv
    )
    
    plot_combined_yield_pdf(
        mean_out=mean_lc, sigma_out=sigma_lc, target_node=target_node,
        spec_min=dynamic_min, spec_max=dynamic_max, folder=folder,
        name=f"{plot_name}_pdf", step_idx=step_idx
    )
    
    return mean_lc, sigma_lc

def perform_sdwc_yield_analysis(
    circuit, result, target_node, calculated_params, netlist_name, 
    folder_map, factory_tolerance=0.05, sigma_level=6,
    out_spec=0.05, out_sigma_level=6, k_params=3
):
    out_folder = "../figures/yield"
    os.makedirs(out_folder, exist_ok=True)
    method = getattr(result, "method", "TR") if result.analysis_type == ".TRAN" else "TR"

    # =========================================================================
    # PHASE 1: Adjoint Metrics (M1: Max Individual, M2: Max Variance)
    # =========================================================================
    top_params, full_ranking, step_m1, step_m2 = analyze_and_rank_sensitivities_globally(
        circuit=circuit, result=result, target_node=target_node, 
        candidate_params=calculated_params, k=k_params, tolerance_pct=factory_tolerance
    )
    print(f"\nSweeping top {len(top_params)} parameters: {top_params}")

    # =========================================================================
    # PHASE 2: Woodbury Scout Metric (M3: Max Physical Envelope Spread)
    # =========================================================================
    print("\n--- Phase B: Scouting true physical envelope with Woodbury ---")
    engine = LargeChangeEngine(circuit)
    
    corner_alphas = np.array([-factory_tolerance, factory_tolerance])
    dp_matrix_corners = generate_worst_case_deltas(
        circuit, result, target_node, corner_alphas, top_params, eval_step=None 
    )

    print(f"\n--- [DEBUG] SCOUT DELTA MATRIX ---")
    print(f"Alphas used: {corner_alphas}")
    print(f"Parameters:  {top_params}")
    print(f"DP Matrix:\n{dp_matrix_corners}")
    print(f"----------------------------------")
    
    lc_envelope = engine.compute(
        result=result, param_names=top_params, dp_matrix=dp_matrix_corners,
        variation_axis=corner_alphas, method=method
    )

    n_idx = result.node_map[target_node]
    all_waveforms = lc_envelope.data[:, :, n_idx]
    v_min = np.min(all_waveforms, axis=0)
    v_max = np.max(all_waveforms, axis=0)
    
    step_m3 = int(np.argmax(np.abs(v_max - v_min)))

    # =========================================================================
    # PHASE 3: Deduplication & Deep Evaluation
    # =========================================================================
    # Map the identified steps to their conceptual labels.
    # If two metrics find the exact same step, this dictionary neatly combines them!
    evaluation_steps = {}
    
    def add_step(step, label):
        if step not in evaluation_steps: evaluation_steps[step] = []
        evaluation_steps[step].append(label)

    add_step(step_m1, "M1_Max_Individual_Sens")
    add_step(step_m2, "M2_Max_Total_Variance")
    add_step(step_m3, "M3_Max_Physical_Spread")

    for step, labels in evaluation_steps.items():
        combined_label = "+".join(labels)
        
        # Calculate strict dynamic specs for THIS specific step
        v_nom = result.VI[step][n_idx]
        dynamic_min = v_nom * (1.0 - out_spec)
        dynamic_max = v_nom * (1.0 + out_spec)
        max_allow_sigma = (v_nom * out_spec) / out_sigma_level

        # Fire the reusable helper function
        evaluate_deep_yield_at_step(
            circuit=circuit, result=result, target_node=target_node, 
            step_idx=step, top_params=top_params, full_ranking=full_ranking,
            dynamic_min=dynamic_min, dynamic_max=dynamic_max,
            factory_tol=factory_tolerance, manufacturing_sigma=sigma_level,
            out_sigma_req=out_sigma_level, max_allowable_sigma=max_allow_sigma,
            folder=out_folder, base_name=netlist_name, metric_label=combined_label,
            env_min_v=v_min[step], env_max_v=v_max[step]
        )

    # =========================================================================
    # PHASE 4: The Master Transient Envelope Plot
    # =========================================================================
    if result.analysis_type == ".TRAN":
        # Pass the dictionary of evaluation steps so the plotter can draw lines for all of them
        plot_transient_envelope(
            result, lc_envelope, target_node, evaluation_steps,
            out_folder, f"{netlist_name}_master_tran_envelope"
        )
