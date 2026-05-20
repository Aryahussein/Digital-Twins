"""Main Execution Script.

This file serves as the primary entry point for the EDA simulator. It orchestrates
the entire pipeline: parsing the SPICE netlist, constructing the OOP circuit graph, 
executing the forward numerical engines, calculating Adjoint sensitivities, and 
visualizing the results through Yield and Fault analysis.
"""

from utils.parser import NetlistParser
from core.circuit import Circuit
from engines.simulator import Simulator
from engines.large_change_engine import LargeChangeEngine

import matplotlib.pyplot as plt
import numpy as np

from utils.plotting import (
    plot_transient,
    plot_transient_sensitivity,
    plot_dc_sweep,
    plot_dc_sensitivity,
    plot_ac_sensitivity,
    plot_transient_sensitivity_to_radiation,
    print_solution,
    make_bode_plot,
    plot_fault_comparison,
    plot_all_faults,
    plot_worst_case_corners,
    plot_combined_yield_pdf,
)

from applications.fault_analysis import (
    rank_component_sensitivities,
    perform_global_ranking,
    print_fault_table,
    compute_fault_thresholds,
    print_threshold_table,
)

from applications.yield_analysis import (
    perform_sdwc_yield_analysis, 
    calculate_analytical_yield
)


def run_simulation_core(
    netlist_path,
    output_nodes=None,
    sensitivity=False,
    global_adjoint=False,
    keep_lus=False,
):
    """Runs the complete SPICE simulation pipeline.

    Args:
        netlist_path (str): The file path to the SPICE netlist text file.
        output_nodes (list[str | int], optional): Target nodes for Adjoint 
            sensitivities. Defaults to None (tracks all nodes).
        sensitivity (bool, optional): Toggles the calculation of the unified 
            continuous sensitivity tensor. Defaults to False.
        global_adjoint (bool, optional): Toggles the backward global integral 
            (only applicable for .TRAN analysis). Defaults to False.
        keep_lus (bool, optional): Forces LU factorization caching for advanced 
            post-processing. Defaults to False.

    Returns:
        tuple[Circuit, SimulationResult]: The initialized circuit object and 
        the secure data vault containing all simulation outputs.
    """
    # 1. Parse the text file
    parser = NetlistParser()
    raw_components, analyses = parser.parse(netlist_path)

    print(analyses)

    # 2. Construct the Object-Oriented Circuit
    circuit = Circuit(raw_components)

    # 3. Hand the Circuit to the Simulator Manager
    sim = Simulator(circuit, analyses, output_nodes)

    # 4. Execute Analysis
    result = sim.execute_analysis(
        sensitivity=sensitivity, 
        global_adjoint=global_adjoint, 
        keep_lus=keep_lus
    )

    return circuit, result


if __name__ == "__main__":
    
    # =========================================================================
    # 1. CONFIGURATION & SETTINGS
    # =========================================================================
    # netlist = "ring_oscilator"  # e.g., nmos_inverter, 1T1C_dram_cell
    #
    # plot_nodes = ["n1", "n2", "n3"]
    # output_nodes = ["n3"]
    # target_node = "n3"
    # target_parameter = "M3_W"
    #
    # netlist = "CMOS_inverter_CAP_tighter_tol"  # e.g., nmos_inverter, 1T1C_dram_cell
    #
    # plot_nodes = ["IN", "OUT"]
    # output_nodes = ["OUT"]
    # target_node = "OUT"
    # target_parameter = "CL"

    netlist = "rc_transient"  # e.g., nmos_inverter, 1T1C_dram_cell

    plot_nodes = ["in", "out"]
    output_nodes = ["out"]
    target_node = "out"
    target_parameter = "C1"
    
    file_path = f"../testfiles/{netlist}.txt"
    # --- Application Toggles ---
    run_sensitivity = True     # Calculates the Adjoint matrix
    
    plot_base_sweep = True     # Plots the standard V(t), V(f), or V(v) curve
    plot_sensitivity = True    # Plots the dV/dp curves
    
    radiation_sim = False      # Specific transient radiation feature
    fault_analysis = False     # Component Ranking and Thresholding
    yield_analysis = True      # SDWC Woodbury analysis

    # --- Yield Analysis Settings ---
    # Set to an integer (e.g., 50) to evaluate a specific step, or None for Auto
    user_eval_step = None

    number_of_large_change_params = 4 # Number of parameters to sweep in large change analysis

    output_spec = 0.05             # Output voltage must stay within ±5%
    output_sigma_req = 6          # We demand a 6-Sigma yield from the circuit

    # =========================================================================
    # 2. EXECUTION CORE
    # =========================================================================
    circuit, result = run_simulation_core(
        file_path,
        output_nodes=output_nodes,
        sensitivity=run_sensitivity,
        keep_lus=yield_analysis,           # Yield requires LU caching
        global_adjoint=run_sensitivity,    # TRAN requires global integral for fault/yield
    )

    if result is None:
        print("Simulation failed or returned no results.")
        exit()

    # Define standard output folders based on analysis type
    folder_map = {".TRAN": "tran", ".AC": "ac", ".DC": "dc", ".OP": "op"}
    analysis_key = result.analysis_type
    
    out_dir = f"../figures/{folder_map.get(analysis_key, 'misc')}"
    base_name = f"{netlist}_{folder_map.get(analysis_key, 'misc')}"

    # =========================================================================
    # 3. BASE SOLUTION & PLOTTING
    # =========================================================================
    if analysis_key == ".OP":
        print_solution(result)
        
    elif plot_base_sweep:
        if analysis_key == ".TRAN":
            plot_transient(result, output_nodes=plot_nodes, folder=out_dir, name=base_name)
        elif analysis_key == ".AC":
            make_bode_plot(result, output_nodes=plot_nodes, folder=out_dir, name=base_name)
        elif analysis_key == ".DC":
            plot_dc_sweep(result, output_nodes=plot_nodes, folder=out_dir, name=base_name)

    # =========================================================================
    # 4. SENSITIVITY POST-PROCESSING
    # =========================================================================
    calculated_params = result.get_sensitivity_parameters(target_node) if result.sensitivities else []

    if calculated_params:
        print(f"\n=== SENSITIVITIES FOR V({target_node}) ===")
        
        # Print OP values or TRAN integrated values
        if analysis_key == ".OP":
            for param in calculated_params:
                sens_val = result.get_sensitivity(target_node, param)
                print(f"  d({target_node})/d({param}) = {sens_val:+.6e}")

        elif analysis_key == ".TRAN" and result.global_sensitivities:
            for param in calculated_params:
                val = result.global_sensitivities["Integrated_Transient"][target_node].get(param, 0.0)
                print(f"  {param:<15} : {val:+.6e} (Integrated)")
                
        # --- Plot Sensitivities ---
        if plot_sensitivity and target_parameter in calculated_params:
            if analysis_key == ".TRAN":
                plot_transient_sensitivity(result, target_node, target_parameter, folder=out_dir, name=f"{base_name}_sens")
                
                if radiation_sim:
                    plot_transient_sensitivity_to_radiation(
                        result, target_node, target_parameter,
                        radiation=[1.6e-19 * 10**i for i in range(4, 8, 1)],
                        folder=out_dir, name=f"{base_name}_sens_radiated"
                    )
            elif analysis_key == ".AC":
                plot_ac_sensitivity(result, target_node, target_parameter, folder=out_dir, name=f"{base_name}_sens")
            elif analysis_key == ".DC":
                plot_dc_sensitivity(result, target_node, target_parameter, folder=out_dir, name=f"{base_name}_sens")

    # =========================================================================
    # 5. FAULT ANALYSIS
    # =========================================================================
    if fault_analysis and result.sensitivities:
        print("\n=== EXECUTING FAULT ANALYSIS ===")
        rank_component_sensitivities(result)
        all_fault_tables = perform_global_ranking(circuit, result)
        print_fault_table(all_fault_tables)

        delta_F = 0.1  # Output tolerance in Volts
        
        for node in output_nodes:
            thresholds = compute_fault_thresholds(circuit, result, node, delta_F)
            print_threshold_table(thresholds)
            
            # Fault plotting usually makes the most sense for Transient and DC sweeps
            if analysis_key in [".TRAN", ".DC"]:
                plot_fault_comparison(
                    circuit, result, node, thresholds, delta_F, 
                    folder="../figures/fault", name=f"{netlist}_fault_{node}"
                )

    # =========================================================================
    # 6. YIELD ANALYSIS (SDWC Woodbury)
    # =========================================================================
    if yield_analysis:
        print("\n=== EXECUTING YIELD ANALYSIS ===")
        res =  perform_sdwc_yield_analysis(
            circuit=circuit, 
            result=result, 
            target_node=target_node, 
            calculated_params=calculated_params,
            netlist_name=netlist,
            folder_map=folder_map,
            out_spec=output_spec,
            out_sigma_level=output_sigma_req,
            k_params=number_of_large_change_params
        )
