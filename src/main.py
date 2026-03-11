from parser import NetlistParser
from circuit import Circuit
from simulator import Simulator  # Your newly refactored Manager!
from tools import plot_transient, plot_transient_sensitivity
import numpy as np

def run_simulation_core(netlist_path, output_nodes=None, sensitivity=False, keep_lus=False):
    # 1. Parse the text file
    parser = NetlistParser()
    raw_components, analyses = parser.parse(netlist_path)

    # 2. CONSTRUCT THE OBJECT-ORIENTED CIRCUIT
    circuit = Circuit(raw_components)

    # 3. Hand the Circuit to the Simulator Manager
    sim = Simulator(circuit, analyses, output_nodes, ramp=1)

    # 4. Execute Analysis
    result = sim.execute_analysis(sensitivity=sensitivity, keep_lus=keep_lus)

    # 5. Return data (extracting node_map from the circuit)
    sens_post_proc = result.sensitivities if hasattr(result, "sensitivities") else None

    return {
        "analyses": analyses, 
        "circuit": circuit,          # Pass the whole object!
        "x_axis": result.sweep_axis, 
        "VI": result.VI, 
        "sens_post_proc": sens_post_proc,
        "output_nodes": output_nodes
    }

if __name__ == "__main__":
    netlist = "rc_transient" 
    file_path = f"../testfiles/{netlist}.txt"
    target_node = ["out"]
    target_component = "C1" 

    results = run_simulation_core(file_path, output_nodes=target_node, sensitivity=True)
    
    # Unpack results
    analyses = results["analyses"]
    x_axis = results["x_axis"]
    VI = results["VI"]
    node_map = results["circuit"].node_map # <--- Pulled right from the circuit
    sensitivities_dict = results["sens_post_proc"]

    if ".TRAN" in analyses:
        plot_transient(x_axis, VI, node_map, target_node, folder="../figures/tran", name=f"{netlist}_tran")
        
        if sensitivities_dict:
            # Process the cumulative integral 
            integ_sens = sensitivities_dict["Integrated_Transient"]
            time_series = sensitivities_dict["Time_Series"]

            primary_node = target_node[0]
            cumulative_series = {primary_node: {}}
            dt = analyses[".TRAN"]["step"]
            
            for param, arr in time_series[primary_node].items():
                cumulative_series[primary_node][param] = np.cumsum(arr) * dt
            
            print("\n=== INTEGRATED TRANSIENT SENSITIVITIES ===")
            for param, val in integ_sens.items():
                print(f"  {param:<15} : {val:+.6e}")
            
            plot_transient_sensitivity(
                x_axis, VI, cumulative_series, node_map, target_node, 
                target_component=target_component, folder="../figures/tran", name=f"{netlist}_tran_sens"
            )
