import numpy as np
from engines.large_change_engine import LargeChangeEngine
from applications.sensitivity_utils import analyze_and_rank_sensitivities, generate_worst_case_deltas, calculate_analytical_yield, calculate_large_change_yield
from utils.plotting import plot_combined_yield_pdf, plot_transient_envelope

def perform_sdwc_yield_analysis(
    circuit, result, target_node, calculated_params, netlist_name, 
    folder_map, user_eval_step=None, factory_tolerance=0.05,
    k_params=3, alpha_range=(-0.20, 0.20), alpha_steps=50
):
    eval_step, top_params = analyze_and_rank_sensitivities(
        circuit=circuit, 
        result=result, 
        target_node=target_node, 
        candidate_params=calculated_params, 
        eval_step=user_eval_step, 
        k=k_params, 
        tolerance_pct=factory_tolerance
    )

    print(f"\nSweeping the finalized top {len(top_params)} sensitive parameters: {top_params}")
    print(f"Sweep Range: {alpha_range[0]*100}% to {alpha_range[1]*100}%")

    # 3. Dynamic Specifications
    n_idx = result.node_map[target_node]
    v_nominal_at_step = result.VI[eval_step][n_idx]
    
    if np.abs(v_nominal_at_step) < 1e-6:
        dynamic_min, dynamic_max = -0.1, 0.1
    else:
        dynamic_min = v_nominal_at_step * 0.90
        dynamic_max = v_nominal_at_step * 1.10

    # 4. Execute Yield Estimations side-by-side for comparison!
    print("\n--- Phase A: Adjoint (Linear) Yield Estimation ---")
    mean_adj, sigma_adj, yield_adj = calculate_analytical_yield(
        circuit=circuit, result=result, target_node=target_node, 
        step_idx=eval_step, spec_min=dynamic_min, spec_max=dynamic_max, 
        tolerance_pct=factory_tolerance, sigma_level=3
    )

    print("\n--- Phase B: Large Change (Non-Linear) Yield Estimation ---")
    mean_lc, sigma_lc, yield_lc, lc_vaults_dict = calculate_large_change_yield( # <-- Add lc_vaults_dict here!
        circuit=circuit, base_result=result, target_node=target_node, 
        step_idx=eval_step, top_params=top_params,
        spec_min=dynamic_min, spec_max=dynamic_max, 
        tolerance_pct=factory_tolerance, sigma_level=3
    )

    # 5. Setup & Execute Woodbury Engine for the full Sweep Visualizations
    alpha_sweep = np.linspace(alpha_range[0], alpha_range[1], alpha_steps)
    method = getattr(result, "method", "TR") if result.analysis_type == ".TRAN" else "TR"
    
    dp_matrix = generate_worst_case_deltas(
        circuit=circuit, result=result, target_node=target_node, 
        alpha_array=alpha_sweep, top_params=top_params, eval_step=eval_step
    )
    
    engine = LargeChangeEngine(circuit)
    lc_results = engine.compute(
        result=result, param_names=top_params, dp_matrix=dp_matrix,
        variation_axis=alpha_sweep, method=method
    )

    # 6. Plotting (Feeding it the superior Large Change statistics)
    out_folder = "../figures/yield"
    plot_name_pdf = f"{netlist_name}_{folder_map.get(result.analysis_type, 'misc')}_combined_yield"

    plot_combined_yield_pdf(
        mean_out=mean_lc, sigma_out=sigma_lc, lc_results=lc_results, # <-- Using LC stats
        alpha_sweep=alpha_sweep, target_node=target_node,
        spec_min=dynamic_min, spec_max=dynamic_max,
        factory_tolerance=factory_tolerance, folder=out_folder,
        name=plot_name_pdf, step_idx=eval_step
    )

    if result.analysis_type == ".TRAN":
        plot_transient_envelope(
            result, lc_results, target_node, eval_step, 
            dynamic_min, dynamic_max, out_folder, f"{netlist_name}_tran_yield_envelope"
        )

