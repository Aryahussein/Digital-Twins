"""
Main Execution Script.

This file serves as the primary entry point for the EDA simulator. It orchestrates
the pipeline: parsing the netlist, constructing the OOP circuit, executing the 
numerical engines, and visualizing the results.
"""

from utils.parser import NetlistParser
from core.circuit import Circuit
from engines.simulator import Simulator
from utils.plotting import plot_transient, plot_transient_sensitivity # Assuming renamed to plotting.py
import numpy as np

def run_simulation_core(netlist_path, output_nodes=None, sensitivity=False, keep_lus=False):
    """Runs the complete SPICE simulation pipeline.

    Args:
        netlist_path (str): The file path to the SPICE netlist text file.
        output_nodes (list, optional): Target nodes for Adjoint sensitivities. 
            Defaults to None.
        sensitivity (bool, optional): Toggles the backward Adjoint pass. Defaults to False.
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
    sim = Simulator(circuit, analyses, output_nodes, ramp=1)

    # 4. Execute Analysis
    result = sim.execute_analysis(sensitivity=sensitivity, keep_lus=keep_lus)

    # 5. Return the objects directly! No more messy dictionaries.
    return circuit, result


if __name__ == "__main__":
    # --- Configuration ---
    netlist = "test_differential_pair"
    file_path = f"../testfiles/{netlist}.txt"
    target_node = "out"  # The plotting tools now handle single strings or lists automatically!
    target_component = "C1" 

    # --- Execution ---
    circuit, result = run_simulation_core(
        file_path, 
        output_nodes=[target_node], 
        sensitivity=True
    )

    # --- Post-Processing & Visualization ---
    if result.type == ".TRAN":
            plot_transient(
                result, 
                output_nodes=target_node, 
                folder="../figures/tran", 
                name=f"{netlist}_tran"
            )
            
            # Use our new OOP helper to check if data exists!
            calculated_params = result.get_sensitivity_parameters(target_node)
            
            if calculated_params:
                print("\n=== INTEGRATED TRANSIENT SENSITIVITIES ===")
                
                for param in calculated_params:
                    # Clean, pure OOP getters
                    val = result.get_sensitivity(target_node, param, output_format="integrated")
                    print(f"  {param:<15} : {val:+.6e}")
                
                plot_transient_sensitivity(
                    result, 
                    target_node, 
                    target_component=target_component, 
                    folder="../figures/tran", 
                    name=f"{netlist}_tran_sens"
                )
            
    elif result.type == ".AC":
        from utils.plotting import make_bode_plot
        make_bode_plot(result, output_nodes=target_node, folder="../figures/ac", name=f"{netlist}_ac")
