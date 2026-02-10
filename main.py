from txt2dictionary import parse_netlist
from node_index import build_node_index
from solver import solve_sparse, solve_LU, solve_adjoint, get_node_and_branch_currents
from postprocessing import map_voltages
from assembleYmatrix import generate_stamps
import numpy as np
from tools import run_bode_plot, print_solution, get_all_sensitivities, plot_sensitivity_sweep

def build_matrix(componenents, w=0):
    node_map, total_dim = build_node_index(components)

    # Build Y matrix and I vector from components
    Y, sources = generate_stamps(components, node_map, total_dim, w=w)

    return Y, sources, node_map, total_dim

def estimate_std_dev(sensitivities, components, percent_sigma=0.01):
    variance = 0
    for name, sens in sensitivities.items():
        # Assume 1% is the standard deviation of the component
        sigma_p = components[name]["value"] * percent_sigma
        variance += np.abs(sens * sigma_p)**2
        
    return np.sqrt(variance)

def do_sensitivity_analysis(lu, VI, output_node, node_map, total_dim, w=0):
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
    netlist = test_directory + "/ac_lowpass.txt"

    w = 2*np.pi * 60
    # w = 0

    # get components
    components = parse_netlist(netlist)
    print(components)

    # get matrix, sources, and mapping of components to matrix indeces
    Y, sources, node_map, total_dim = build_matrix(components, w=w)
    print(node_map)

    # solve! get LU, node voltages and branch currents of original circuit
    lu = solve_LU(Y)
    VI = get_node_and_branch_currents(lu, sources)
    print_solution(VI, node_map, w=w)

    # select output node for adjoint input
    output_node_for_sensitivity = 2
    sensitivities, std_dev = do_sensitivity_analysis(lu, VI, output_node_for_sensitivity, node_map, total_dim, w=w)
    print("Sensitivities of components:")
    for name, sens in sensitivities.items():
        print(f"{name}: {sens:.4f} V/unit")
    print(f"output voltage V = {VI[node_map[output_node_for_sensitivity]]} $\pm$ {std_dev} V")

    # # Example Bode Plot
    # run_bode_plot(test_directory + "ac_lowpass.txt", output_node=2, start_freq=10, stop_freq=100000, points=200, name = "lowpass")
    # run_bode_plot(test_directory + "ac_resonance.txt", output_node=3, start_freq=10, stop_freq=100000, points=200, name = "resonance")

    # Example ac sensitivity
    # print("do sweep")
    plot_sensitivity_sweep(components, output_node_for_sensitivity, "C1", start_f=10, end_f=1000, name="lowpass_C1_sensitivity")

    









