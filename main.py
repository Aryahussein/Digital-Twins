from txt2dictionary import parse_netlist
from node_index import build_node_index
from solver import solve_nonlinear_circuit, solve_linear_circuit, solve_adjoint
from postprocessing import map_voltages
from assembleYmatrix import generate_stamps
import numpy as np
from tools import run_bode_plot, print_solution, get_all_sensitivities, plot_sensitivity_sweep
from constants import *

def estimate_std_dev(sensitivities, components, percent_sigma=0.01):
    variance = 0
    for name, sens in sensitivities.items():
        # Assume 1% is the standard deviation of the component
        sigma_p = components[name]["value"] * percent_sigma
        variance += np.abs(sens * sigma_p)**2
        
    return np.sqrt(variance)

def do_sensitivity_analysis(lu, VI, output_node, node_map, total_dim, w=0.0):
    # get all node and branch voltages of adjoint circuit
    PsiPhi = solve_adjoint(lu, output_node, node_map, total_dim)

    # get the sensitivities of all components
    sensitivities = get_all_sensitivities(components, VI, PsiPhi, node_map, w=w)
    # print(sensitivities)

    # find the standard deviation on the output voltage
    tolerance_on_components = 0.01
    std_dev = estimate_std_dev(sensitivities, components, percent_sigma=tolerance_on_components)
    # print(std_dev)
    return sensitivities, std_dev

if __name__ == "__main__":
    test_directory = "testfiles/"
    netlist = test_directory + "/test_lots_of_diodes.txt"


    #-----------------------------------------------------------------------------------
    # Build circuit
    #-----------------------------------------------------------------------------------
    # get components
    components = parse_netlist(netlist)
    print(components)

    nonlinear = False
    for name, comp in components.items():
        if name.startswith("D"):
            nonlinear = True

    node_map, total_dim = build_node_index(components)

    # PUT THE FREQUENCY SOMEWHERE ELSE!!
    w = 2*np.pi * 60
    if nonlinear:
        w = 0.0

    Y, sources = generate_stamps(components, node_map, total_dim, w=w)
    
    #-----------------------------------------------------------------------------------
    # solve circuit
    #-----------------------------------------------------------------------------------
    if nonlinear:
        V_guess = np.zeros(total_dim)
        max_iter = 100
        tol = 1e-9
        num_ramp_steps = 10
        lu, VI = solve_nonlinear_circuit(Y, sources, components, node_map, total_dim, V_guess, max_iter=max_iter, tol=tol, num_steps=num_ramp_steps)
    else:
        lu, VI = solve_linear_circuit(Y, sources)

    print_solution(VI, node_map, w=w)

<<<<<<< HEAD
    # select output node for adjoint input
    output_node_for_sensitivity = 2
    sensitivities, std_dev = do_sensitivity_analysis(lu, VI, output_node_for_sensitivity, node_map, total_dim, w=w)
    print("Sensitivities of components:")
    for name, sens in sensitivities.items():
        print(f"{name}: {sens:.4f} V/unit")
    print(f"output voltage V = {VI[node_map[output_node_for_sensitivity]]} $\pm$ {std_dev} V")
=======
    #-----------------------------------------------------------------------------------
    # get sensitivities
    #-----------------------------------------------------------------------------------
>>>>>>> non_linear_solver

    # # select output node for adjoint input
    # output_node_for_sensitivity = 3
    # sensitivities, std_dev = do_sensitivity_analysis(lu, VI, output_node_for_sensitivity, node_map, total_dim, w=w)
    # print(f"output voltage V = {VI[node_map[output_node_for_sensitivity]]} $\pm$ {std_dev} V")

    
    #-----------------------------------------------------------------------------------
    # ac stuff
    #-----------------------------------------------------------------------------------
    # # Example Bode Plot
    # run_bode_plot(test_directory + "ac_lowpass.txt", output_node=2, start_freq=10, stop_freq=100000, points=200, name = "lowpass")
    # run_bode_plot(test_directory + "ac_resonance.txt", output_node=3, start_freq=10, stop_freq=100000, points=200, name = "resonance")

<<<<<<< HEAD
    # Example ac sensitivity
    # print("do sweep")
    plot_sensitivity_sweep(components, output_node_for_sensitivity, "C1", start_f=10, end_f=1000, name="lowpass_C1_sensitivity")
=======
    # # Example ac sensitivity
    # print("do sweep")
    # plot_sensitivity_sweep(components, output_node_for_sensitivity, "R1", start_f=10, end_f=1000, name="sensitiviy")
>>>>>>> non_linear_solver

    









