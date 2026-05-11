import numpy as np
import matplotlib.pyplot as plt
from engines.large_change_engine import LargeChangeEngine
from applications.sensitivity_utils import generate_worst_case_deltas, calculate_analytical_yield
from utils.plotting import plot_combined_yield_pdf, plot_transient, plot_transient_sensitivity_corrected_units, plot_transient_envelope

def perform_sdwc_yield_analysis(
    circuit, 
    result, 
    target_node, 
    calculated_params, 
    netlist_name, 
    folder_map, 
    user_eval_step=None, 
    factory_tolerance=0.05,
    k_params=3,                  # <-- Defaulted to top 3 components
    alpha_range=(-0.20, 0.20),   # <-- Swiping from -20% to +20%
    alpha_steps=50
):
    """
    Executes Yield Analysis by sweeping the top sensitive parameters and 
    visualizing the transient envelope against 3-sigma specification thresholds.
    """
    if not (result.sensitivities and getattr(result, "list_of_lus", None)):
        print("Yield analysis skipped: Requires both sensitivities and LU factorization history.")
        return

    print("\n=== EXECUTING TRANSIENT YIELD ANALYSIS ===")

    # Ensure we only take the top K parameters (e.g., the 3 most sensitive)
    top_params = calculated_params[:k_params]
    print(f"Sweeping the top {len(top_params)} sensitive parameters: {top_params}")
    print(f"Sweep Range: {alpha_range[0]*100}% to {alpha_range[1]*100}%")

    # ==========================================
    # 1. Determine the Evaluation Target Point
    # ==========================================
    eval_step = 0
    if user_eval_step is not None:
        eval_step = int(user_eval_step)
    else:
        if result.analysis_type in [".TRAN", ".AC", ".DC"]:
            num_steps = len(result.VI)
            aggregate_sensitivity = np.zeros(num_steps)
            for param in top_params:
                sens_waveform = np.atleast_1d(result.get_sensitivity(target_node, param))
                aggregate_sensitivity += np.abs(sens_waveform)
            eval_step = int(np.argmax(aggregate_sensitivity))
            print(f"Evaluating Yield at MAXIMUM Adjoint Sensitivity (Step {eval_step})")

    # ==========================================
    # 2. Setup & Execute Woodbury Engine (SDWC)
    # ==========================================
    alpha_sweep = np.linspace(alpha_range[0], alpha_range[1], alpha_steps)
    method = getattr(result, "method", "TR") if result.analysis_type == ".TRAN" else "TR"
    
    # Generate the worst-case trajectory through the parameter space
    params, dp_matrix = generate_worst_case_deltas(
        circuit=circuit, sensitivities=result.sensitivities,
        target_node=target_node, alpha_array=alpha_sweep, k=k_params
    )
    
    engine = LargeChangeEngine(circuit)
    lc_results = engine.compute(
        result=result, param_names=params, dp_matrix=dp_matrix,
        variation_axis=alpha_sweep, method=method
    )

    # ==========================================
    # 3. Dynamic Specifications (The 3-Sigma Threshold)
    # ==========================================
    n_idx = result.node_map[target_node]
    v_nominal_at_step = result.VI[eval_step][n_idx]
    
    # Define the pass/fail threshold window
    if np.abs(v_nominal_at_step) < 1e-6:
        dynamic_min, dynamic_max = -0.1, 0.1
    else:
        # Example: 10% tolerance around the nominal point
        dynamic_min = v_nominal_at_step * 0.90
        dynamic_max = v_nominal_at_step * 1.10

    # ==========================================
    # 4. Analytical Yield Estimation
    # ==========================================
    mean_out, sigma_out, yield_pct = calculate_analytical_yield(
        circuit=circuit, 
        result=result,                
        target_node=target_node, 
        step_idx=eval_step, 
        spec_min=dynamic_min, 
        spec_max=dynamic_max, 
        tolerance_pct=factory_tolerance, 
        sigma_level=3
    )

    out_folder = "../figures/yield"
    plot_name_pdf = f"{netlist_name}_{folder_map.get(result.analysis_type, 'misc')}_combined_yield"

    plot_combined_yield_pdf(
        mean_out=mean_out, sigma_out=sigma_out, lc_results=lc_results,
        alpha_sweep=alpha_sweep, target_node=target_node,
        spec_min=dynamic_min, spec_max=dynamic_max,
        factory_tolerance=factory_tolerance, folder=out_folder,
        name=plot_name_pdf, step_idx=eval_step
    )

    # ==========================================
    # 5. NEW: Transient Yield Envelope Visualization
    # ==========================================
    if result.analysis_type == ".TRAN":
        plot_transient_envelope(
            result, lc_results, target_node, eval_step, 
            dynamic_min, dynamic_max, out_folder, f"{netlist_name}_tran_yield_envelope"
        )

