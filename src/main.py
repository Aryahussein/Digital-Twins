from parser import NetlistParser
from simulations import Simulator
from node_index import build_node_index
from assembleYmatrix import stamp_linear_components, stamp_static_components, initialize_stamps
from sources import evaluate_all_time_sources
from sensitivity import aggregate_sweep_sensitivities, compute_step_sensitivities
from tools import make_bode_plot, plot_ac_sensitivity, plot_transient, plot_transient_sensitivity
import numpy as np

def run_simulation_core(netlist_path, output_nodes=None, sensitivity=False, sensitivity_post=False, keep_lus=False):
    # 1. Parse
    parser = NetlistParser()
    components, analyses = parser.parse(netlist_path)
    print(components)
    node_map = build_node_index(components)

    ramp = 1
    # 2. Setup Solver (Matrix and Nonlinearity logic is now INSIDE the class)
    sim = Simulator(components, analyses, node_map, output_nodes, ramp=ramp)

    # 3. Solve (The class routes the analysis internally)
    x_axis, VI, list_of_lus, raw_sens = sim.execute_analysis(
        sensitivity=sensitivity, 
        keep_lus=(keep_lus or sensitivity_post)
    )


    sens_post_proc = raw_sens

    # 4. Handle Post-Processing Sensitivity
    if sensitivity_post or sensitivity:
        if output_nodes is None:
            output_nodes = list(node_map.keys())

        # =====================================================================
        # NEW BYPASS: Skip aggregation if we already have True Adjoint results!
        # =====================================================================
        if isinstance(raw_sens, dict) and "Integrated_Transient" in raw_sens:
            sens_post_proc = raw_sens

        elif ".TRAN" in analyses or ".AC" in analyses:
            sens_post_proc = aggregate_sweep_sensitivities(
                components, node_map, analyses, raw_sensitivities=raw_sens, 
                output_nodes=output_nodes, list_of_lus=list_of_lus, 
                VI_list=VI, freq_list=x_axis if ".AC" in analyses else None
            )
        else: #OP, no sweep needed, only one step
            if sensitivity_post:
                sens_post_proc = compute_step_sensitivities(
                    list_of_lus, VI, components, node_map, output_nodes, 
                    w=(analyses.get(".OP", {}).get("freq", 0.0) * 2 * np.pi)
                )

    return {
        "analyses": analyses, "components": components, "node_map": node_map,
        "x_axis": x_axis, "VI": VI, "sens_post_proc": sens_post_proc,
        "output_nodes": output_nodes, "list_of_lus": list_of_lus
    }

if __name__ == "__main__":
    
    # ==========================================
    # TOGGLE THIS TO SWITCH BETWEEN GUI AND CLI
    USE_GUI = False
    # ==========================================

    if USE_GUI:
        import tkinter as tk
        from gui import CircuitSimulatorGUI # Make sure your gui code is saved as gui.py
        
        root = tk.Tk()
        app = CircuitSimulatorGUI(root, run_simulation_core) # Pass the core function to the GUI
        root.mainloop()
        
    else:
        netlist = "rc_transient" # Choose your netlist here
        target_node = ["out"]
        target_node_for_plotting = ["out"]
        target_component = "C1" # Let's look at the capacitor!
        
        file_path = f"../testfiles/{netlist}.txt"

        keep_lus = False

        # Choose how to compute the sensitivity
        sensitivity = True # Good for when you only need to have the sensitivity at a few output nodes (less memory needed)
        sens_post_proc = False # Good when you need to have the sensitivity at all output nodes

        if sens_post_proc:
            sensitivity = False
            keep_lus = True
        elif sensitivity:
            sens_post_proc = False
            keep_lus = False
        else:
            keep_lus = False
            sens_post_proc = False
            sensitivity = False


        results = run_simulation_core(
            file_path, output_nodes=target_node, 
            sensitivity=sensitivity, sensitivity_post=sens_post_proc, keep_lus=keep_lus
        )
        
        # Unpack results
        analyses = results["analyses"]
        x_axis = results["x_axis"]
        VI = results["VI"]
        node_map = results["node_map"]

        sensitivities_list = results["sens_post_proc"]

        print(sensitivities_list)
        
        # Visualize
        if ".AC" in analyses:
            make_bode_plot(x_axis, VI, node_map, target_node_for_plotting, folder="../figures/ac", name=f"{netlist}_bode")
            if sens_post_proc or sensitivity:
                plot_ac_sensitivity(x_axis, VI, sensitivities_list, node_map, target_node_for_plotting, target_component=target_component, folder="../figures/ac", name=f"{netlist}_ac_sens")

        elif ".TRAN" in analyses:
            plot_transient(x_axis, VI, node_map, target_node_for_plotting, folder="../figures/tran", name=f"{netlist}_tran")
            
            if sens_post_proc or sensitivity:
                # Check if this is our new True Adjoint dictionary
                if isinstance(sensitivities_list, dict) and "Integrated_Transient" in sensitivities_list:
                    integ_sens = sensitivities_list["Integrated_Transient"]
                    time_series = sensitivities_list["Time_Series"]
                    
                    # 1. Print the Integrated Scalars
                    print(f"\n=======================================================")
                    print(f" INTEGRATED TRANSIENT SENSITIVITIES")
                    print(f" Objective: Output Voltage at t={x_axis[-1]:.4f}s")
                    print(f"=======================================================")
                    for param, val in integ_sens.items():
                        print(f"  {param:<15} : {val:+.6e}")
                    print(f"=======================================================\n")
                    
                    # 2. Plot the Time-Series Integrand
                    plot_transient_sensitivity(
                        x_axis, VI, time_series, node_map, target_node, 
                        target_component=target_component, folder="../figures/tran", name=f"{netlist}_tran_sens"
                    )
                else:
                    # Fallback for standard step-wise post-processing
                    plot_transient_sensitivity(
                        x_axis, VI, sensitivities_list, node_map, target_node, 
                        target_component=target_component, folder="../figures/tran", name=f"{netlist}_tran_sens"
                    )

        else:
            if sens_post_proc or sensitivity:
                if target_node is None:
                    target_node = list(node_map.keys())
                print(f"Sensitivity for all components and output nodes {target_node}")
                print(f"Sensitivity: {sensitivities_list}")
