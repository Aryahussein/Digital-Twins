"""
Main Execution Script.

This file serves as the primary entry point for the EDA simulator. It orchestrates
the pipeline: parsing the netlist, constructing the OOP circuit, executing the 
numerical engines, and visualizing the results.
"""

from utils.parser import NetlistParser
from core.circuit import Circuit
from engines.simulator import Simulator
from utils.plotting import (
    plot_transient, plot_transient_sensitivity, 
    plot_dc_sweep, plot_dc_sensitivity, 
    plot_ac_sensitivity, print_solution, make_bode_plot
)
import numpy as np


def run_simulation_core(netlist_path, output_nodes=None, sensitivity=False, keep_lus=False, method="BE"):
    """Runs the complete SPICE simulation pipeline.

    Args:
        netlist_path (str): The file path to the SPICE netlist text file.
        output_nodes (list, optional): Target nodes for Adjoint sensitivities.
        sensitivity (bool, optional): Toggles the backward Adjoint pass. Defaults to False.
        keep_lus (bool, optional): Forces LU factorization caching. Defaults to False.
        method (str, optional): Transient integration method — 'BE' or 'TR'. Defaults to 'BE'.

    Returns:
        tuple: (Circuit, SimulationResult)
    """
    parser = NetlistParser()
    raw_components, analyses = parser.parse(netlist_path)

    circuit = Circuit(raw_components)

    sim = Simulator(circuit, analyses, output_nodes)

    result = sim.execute_analysis(sensitivity=sensitivity, keep_lus=keep_lus, method=method)

    return circuit, result


if __name__ == "__main__":
    # --- Configuration ---
    netlist = "rc_lowpass"
    file_path = f"testfiles/{netlist}.txt"
    target_node = None
    target_component = "" 
    integration_method = "BE"  # Choose "BE" or "TR"

    # --- Execution ---
    circuit, result = run_simulation_core(
        file_path, 
        output_nodes=[target_node], 
        sensitivity=False,
        method=integration_method
    )

    # --- Post-Processing & Visualization ---
    if result is None:
        print("Simulation failed or returned no results.")
        exit()

    if result.type == ".TRAN":
        plot_transient(
            result, output_nodes=target_node, 
            folder="figures/tran", name=f"{netlist}_tran"
        )
            
        calculated_params = result.get_sensitivity_parameters(target_node)
        if calculated_params:
            print(f"\n=== INTEGRATED TRANSIENT SENSITIVITIES FOR V({target_node}) ===")
            for param in calculated_params:
                val = result.get_sensitivity(target_node, param, output_format="integrated")
                print(f"  {param:<15} : {val:+.6e}")
            
            if target_component in calculated_params:
                plot_transient_sensitivity(
                    result, target_node, target_component=target_component, 
                    folder="figures/tran", name=f"{netlist}_tran_sens"
                )

    elif result.type == ".AC":
        make_bode_plot(
            result, output_nodes=target_node, 
            folder="figures/ac", name=f"{netlist}_ac"
        )
        calculated_params = result.get_sensitivity_parameters(target_node)
        if calculated_params and target_component in calculated_params:
            plot_ac_sensitivity(
                result, target_node, target_component=target_component, 
                folder="figures/ac", name=f"{netlist}_ac_sens"
            )

    elif result.type == ".DC":
        plot_dc_sweep(
            result, output_nodes=target_node, 
            folder="figures/dc", name=f"{netlist}_dc"
        )
        calculated_params = result.get_sensitivity_parameters(target_node)
        if calculated_params and target_component in calculated_params:
            plot_dc_sensitivity(
                result, target_node, target_component=target_component, 
                folder="figures/dc", name=f"{netlist}_dc_sens"
            )

    elif result.type == ".OP":
        print_solution(result)
        calculated_params = result.get_sensitivity_parameters(target_node)
        if calculated_params:
            print(f"\n=== DC SENSITIVITIES FOR V({target_node}) ===")
            for param in calculated_params:
                sens_val = result.get_sensitivity(target_node, param)
                print(f"  d({target_node})/d({param}) = {sens_val:+.6e}")
