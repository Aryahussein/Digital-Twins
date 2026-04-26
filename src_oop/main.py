"""
Main Execution Script.

This file serves as the primary entry point for the EDA simulator. It orchestrates
the pipeline: parsing the netlist, constructing the OOP circuit, executing the 
numerical engines, and visualizing the results.
"""

from utils.parser import NetlistParser
from core.circuit import Circuit
from engines.simulator import Simulator
from utils.plotting import plot_transient, plot_transient_sensitivity, plot_dc_sweep, plot_dc_sensitivity, plot_ac_sensitivity, print_solution, make_bode_plot,plot_fault_comparison, plot_all_faults
import numpy as np
from applications.fault_analysis import rank_component_sensitivities, perform_global_ranking, print_fault_table,compute_fault_thresholds, print_threshold_table


def run_simulation_core(netlist_path, output_nodes=None, sensitivity=False, global_adjoint=False, keep_lus=False):
    """Runs the complete SPICE simulation pipeline.

    Args:
        netlist_path (str): The file path to the SPICE netlist text file.
        output_nodes (list, optional): Target nodes for Adjoint sensitivities. 
            Defaults to None.
        sensitivity (bool, optional): Toggles the unified continuous sensitivity tensor. Defaults to False.
        global_adjoint (bool, optional): Toggles the backward global integral for TRAN. Defaults to False.
        keep_lus (bool, optional): Forces LU factorization caching. Defaults to False.

    Returns:
        tuple: (Circuit, SimulationResult) containing the initialized circuit 
        object and the secure data vault with all simulation outputs.
    """
    # 1. Parse the text file
    parser = NetlistParser()
    raw_components, analyses = parser.parse(netlist_path)

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

    # 5. Return the objects directly
    return circuit, result


if __name__ == "__main__":
    # --- Configuration ---
    netlist = "rc_transient" # nmos_inverter, rc_lowpass, etc.
    file_path = f"../testfiles/{netlist}.txt"
    target_node = "out"
    target_component = "C1" 

    # --- Execution ---
    output_nodes = ["in","out"]
    circuit, result = run_simulation_core(
        file_path, 
        output_nodes=output_nodes, 
        sensitivity=True,
        keep_lus=True,
        global_adjoint=False  # Ensure we calculate the global backward integrals for TRAN
    )

    # --- Post-Processing & Visualization ---
    if result is None:
        print("Simulation failed or returned no results.")
        exit()

    # ==========================================
    # TRANSIENT ANALYSIS
    # ==========================================
    if result.type == ".TRAN":
        plot_transient(
            result, 
            output_nodes=target_node, 
            folder="../figures/tran", 
            name=f"{netlist}_tran"
        )
            
        calculated_params = result.get_sensitivity_parameters(target_node)
            
        if calculated_params:
            print(f"\n=== TRANSIENT SENSITIVITIES FOR V({target_node}) ===")
            for param in calculated_params:
                # Print the total global scalar integral (if calculated)
                if result.global_sensitivities:
                    val = result.global_sensitivities["Integrated_Transient"][target_node].get(param, 0.0)
                    print(f"  {param:<15} : {val:+.6e} (Integrated)")
            
            if target_component in calculated_params:
                # Plot the continuous tensor series
                # Note: You can remove the 'format' argument from your plotting function!
                plot_transient_sensitivity(
                    result, 
                    target_node, 
                    target_component=target_component, 
                    folder="../figures/tran", 
                    name=f"{netlist}_tran_sens"
                )
        
        #fault table ranking
        rank_component_sensitivities(result)
        all_fault_tables = perform_global_ranking(circuit, result)
        print_fault_table(all_fault_tables)

        # Short thresholds (delta_F = output tolerance in volts)
        delta_F = 0.1
        for node in output_nodes:
            thresholds = compute_fault_thresholds(circuit, result, node, delta_F)
            print_threshold_table(thresholds)
            plot_fault_comparison(circuit, result, node, thresholds, delta_F,
                                folder="../figures/fault", name=f"{netlist}_fault_{node}")
            plot_all_faults(circuit, result, node, thresholds, delta_F,
                            folder="../figures/fault", name=f"{netlist}_all_faults_{node}")

        # Quick tensor test: Print matrices at first and last time steps
        if result.sensitivities:
            result.sensitivities.print_matrix_at_step(0)
            result.sensitivities.print_matrix_at_step(-1)

    # ==========================================
    # AC ANALYSIS (Frequency Domain)
    # ==========================================
    elif result.type == ".AC":
        make_bode_plot(
            result, 
            output_nodes=target_node, 
            folder="../figures/ac", 
            name=f"{netlist}_ac"
        )

        calculated_params = result.get_sensitivity_parameters(target_node)
        
        if calculated_params and target_component in calculated_params:
            plot_ac_sensitivity(
                result, 
                target_node, 
                target_component=target_component, 
                folder="../figures/ac", 
                name=f"{netlist}_ac_sens"
            )
        # ---- FAULT ANALYSIS ----
        rank_component_sensitivities(result)
        all_fault_tables = perform_global_ranking(circuit, result)
        print_fault_table(all_fault_tables)

        delta_F = 0.1
        for node in output_nodes:
            thresholds = compute_fault_thresholds(circuit, result, node, delta_F=delta_F)
            print_threshold_table(thresholds)

    # ==========================================
    # DC SWEEP ANALYSIS
    # ==========================================
    elif result.type == ".DC":
        plot_dc_sweep(
            result, 
            output_nodes=target_node, 
            folder="../figures/dc", 
            name=f"{netlist}_dc"
        )

        calculated_params = result.get_sensitivity_parameters(target_node)
        
        if calculated_params and target_component in calculated_params:
            plot_dc_sensitivity(
                result, 
                target_node, 
                target_component=target_component, 
                folder="../figures/dc", 
                name=f"{netlist}_dc_sens"
            )
         # ---- FAULT ANALYSIS ----
        rank_component_sensitivities(result)
        all_fault_tables = perform_global_ranking(circuit, result)
        print_fault_table(all_fault_tables)

        delta_F = 0.1
        for node in output_nodes:
            thresholds = compute_fault_thresholds(circuit, result, node, delta_F=delta_F)
            print_threshold_table(thresholds)
            plot_fault_comparison(circuit, result, node, thresholds, delta_F,
                                  folder="../figures/fault", name=f"{netlist}_fault_{node}")

    # ==========================================
    # DC OPERATING POINT (.OP)
    # ==========================================
    elif result.type == ".OP":
        # Use the utility function to print node voltages and branch currents
        print_solution(result)

        # Print the exact Adjoint sensitivities for the target node
        calculated_params = result.get_sensitivity_parameters(target_node)
        
        if calculated_params:
            print(f"\n=== DC SENSITIVITIES FOR V({target_node}) ===")
            for param in calculated_params:
                sens_val = result.get_sensitivity(target_node, param)
                print(f"d({target_node})/d({param}) = {sens_val:+.6e}")

        # ---- FAULT ANALYSIS ----
        rank_component_sensitivities(result)
        all_fault_tables = perform_global_ranking(circuit, result)
        print_fault_table(all_fault_tables)

        delta_F = 0.1
        for node in output_nodes:
            thresholds = compute_fault_thresholds(circuit, result, node, delta_F=delta_F)
            print_threshold_table(thresholds)
