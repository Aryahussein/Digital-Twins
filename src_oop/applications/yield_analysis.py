import numpy as np

from engines.large_change_engine import LargeChangeEngine
from utils.plotting import plot_worst_case_corners, plot_combined_yield_pdf, plot_transient, plot_transient_sensitivity_corrected_units
from applications.sensitivity_utils import generate_worst_case_deltas, calculate_analytical_yield

def perform_sdwc_yield_analysis(
    circuit, 
    result, 
    target_node, 
    calculated_params, 
    netlist_name, 
    folder_map, 
    user_eval_step=None, 
    factory_tolerance=0.05,
    k_params=2,
    alpha_range=(-0.20, 0.20),
    alpha_steps=50
):
    """
    Executes both Analytical and SDWC Yield Analysis, combining them 
    into a single statistical reality-check plot.
    """
    if not (result.sensitivities and getattr(result, "list_of_lus", None)):
        print("Yield analysis skipped: Requires both sensitivities and LU factorization history.")
        return

    print("\n=== EXECUTING YIELD ANALYSIS ===")

    # ==========================================
    # 1. Determine the Evaluation Target Point
    # ==========================================
    eval_step = 0
    if user_eval_step is not None:
        eval_step = int(user_eval_step)
        print(f"Evaluating at user-specified step: {eval_step}")
    else:
        if result.type in [".TRAN", ".AC", ".DC"]:
            num_steps = len(result.VI)
            aggregate_sensitivity = np.zeros(num_steps)
            for param in calculated_params:
                sens_waveform = np.atleast_1d(result.get_sensitivity(target_node, param))
                aggregate_sensitivity += np.abs(sens_waveform)
            eval_step = int(np.argmax(aggregate_sensitivity))
            print(f"Evaluating {result.type} Yield at MAXIMUM Adjoint Sensitivity (Step {eval_step})")
        else:
            eval_step = 0 

    if result.type == ".TRAN":
        # 1. Base Transient + Marker
        plot_transient(
            result=result, 
            output_nodes=target_node, 
            folder="../figures/yield", 
            name=f"{netlist_name}_tran_marked", 
            mark_step=eval_step
        )
        
        # 2. Ranked Sensitivities + Marker
        # calculated_params contains your top ranked parameters from Section 4
        plot_transient_sensitivity_corrected_units(
            circuit=circuit,           # <--- Pass the circuit object here!
            result=result,
            output_node=target_node,
            target_component=calculated_params, 
            folder="../figures/yield",
            name=f"{netlist_name}_tran_sensitivities_marked",
            mark_step=eval_step,
            delta_pct=0.05             # Sync this with your factory tolerance if you want!
        )

    # ==========================================
    # 2. Setup & Execute Woodbury Engine (SDWC)
    # ==========================================
    alpha_sweep = np.linspace(alpha_range[0], alpha_range[1], alpha_steps)
    method = getattr(result, "method", "TR") if result.type == ".TRAN" else "TR"
    
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
    # 3. Dynamic Specifications
    # ==========================================
    n_idx = result.node_map[target_node]
    v_nominal_at_step = result.VI[eval_step][n_idx]
    
    if np.abs(v_nominal_at_step) < 1e-6:
        dynamic_min, dynamic_max = -0.1, 0.1
    else:
        dynamic_min = v_nominal_at_step * 0.90
        dynamic_max = v_nominal_at_step * 1.10

    # ==========================================
    # 4. Analytical Yield Estimation
    # ==========================================
    # Calculates continuous PDF probabilities assuming circuit linearity
    mean_out, sigma_out, yield_pct = calculate_analytical_yield(
        circuit=circuit, 
        result=result,                # <--- Pass the full result object here!
        target_node=target_node, 
        step_idx=eval_step, 
        spec_min=dynamic_min, 
        spec_max=dynamic_max, 
        tolerance_pct=factory_tolerance, 
        sigma_level=3
    )

    # ==========================================
    # 5. Combined Master Plot
    # ==========================================
    out_folder = "../figures/yield"
    plot_name = f"{netlist_name}_{folder_map.get(result.type, 'misc')}_combined_yield"

    plot_combined_yield_pdf(
        mean_out=mean_out,
        sigma_out=sigma_out,
        lc_results=lc_results,
        alpha_sweep=alpha_sweep,
        target_node=target_node,
        spec_min=dynamic_min,
        spec_max=dynamic_max,
        factory_tolerance=factory_tolerance,
        folder=out_folder,
        name=plot_name,
        step_idx=eval_step
    )
    print(f"Master Yield plot saved to {out_folder}/")
