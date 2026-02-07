from txt2dictionary import parse_netlist
from node_index import build_node_index
from solver import solve_sparse, solve_LU
from postprocessing import map_voltages
from assembleYmatrix import generate_stamps
import matplotlib.pyplot as plt
import numpy as np
from tools import run_bode_plot, print_solution

def build_matrix(componenents, w=0):
    node_map, total_dim = build_node_index(components)

    # Build Y matrix and I vector from components
    Y, sources = generate_stamps(components, node_map, total_dim, w=w)

    return Y, sources, node_map, total_dim

def get_node_and_branch_currents(lu, sources):
    VI = lu.solve(sources)
    return VI

def get_all_sensitivities(components, VI, PsiPhi, node_map, w=0):
    sensitivities = {}

    for name, comp in components.items():
        # 1. Get Indices and branch voltages
        n1, n2 = comp.get("n1", 0), comp.get("n2", 0)
        idx1, idx2 = node_map.get(n1), node_map.get(n2) # get_idx logic

        # Helper to get the voltage difference across a branch
        # (v1 - v2). If a node is ground, its voltage is 0.
        VI_branch = (VI[idx1] if idx1 is not None else 0) - \
                   (VI[idx2] if idx2 is not None else 0)
        
        PsiPhi_branch = (PsiPhi[idx1] if idx1 is not None else 0) - \
                       (PsiPhi[idx2] if idx2 is not None else 0)

        # 2. Apply the sensitivity formula based on component type
        if name.startswith("R"):
            # Sensitivity w.r.t Resistance (R):
            # dY/dR = (1/R^2) * VI_branch * PsiPhi_branch
            R = comp["value"]
            sensitivities[name] = (1.0 / (R**2)) * (VI_branch * PsiPhi_branch)

        elif name.startswith("C"):
            # Sensitivity w.r.t Capacitance (C):
            # dY/dC = j*w. Sensitivity = - (j*w * VI_branch * PsiPhi_branch)
            sensitivities[name] = -1j * w * (VI_branch * PsiPhi_branch)

        elif name.startswith("L"):
            # For Inductors, we use the MNA branch current
            # Row index for the inductor current is its entry in node_map
            l_curr_idx = node_map[name]
            i_L = VI[l_curr_idx]
            i_L_hat = PsiPhi[l_curr_idx]
            
            # dY/dL for an inductor row is -j*w
            # Sensitivity = - (-j*w * i_L * i_L_hat) = j*w * i_L * i_L_hat
            sensitivities[name] = 1j * w * (i_L * i_L_hat)

        elif name.startswith("G"):
            # VCCS: I = gm * (VI_sense_pos - VI_sense_neg)
            # Sensitive w.r.t transconductance (gm)
            n3, n4 = comp.get("n3", 0), comp.get("n4", 0)
            idx3, idx4 = node_map.get(n3), node_map.get(n4)
            
            v_sense = (VI[idx3] if idx3 is not None else 0) - \
                      (VI[idx4] if idx4 is not None else 0)
            
            # For VCCS, sensitivity = - (PsiPhi_branch_current * VI_sense)
            # In adjoint, this is PsiPhi_branch (voltage at the current source nodes)
            sensitivities[name] = - (PsiPhi_branch * v_sense)

    return sensitivities

def estimate_std_dev(sensitivities, components, percent_sigma=0.01):
    variance = 0
    for name, sens in sensitivities.items():
        # Assume 1% is the standard deviation of the component
        sigma_p = components[name]["value"] * percent_sigma
        variance += np.abs(sens * sigma_p)**2
        
    return np.sqrt(variance)

def solve_adjoint(lu, target, var_map, total_dim):
    """
    target can be a node number (e.g. 2) or a name (e.g. 'V1')
    """
    d = np.zeros(total_dim)
    idx = var_map[target]  # Works for both nodes and branches!
    d[idx] = 1.0
    return lu.solve(d, trans='T')

if __name__ == "__main__":
    test_directory = "testfiles/"
    netlist = test_directory + "/test_book_floating_voltage_source.txt"

    w = 0

    # get components
    components = parse_netlist(netlist)
    print(components)

    # get matrix, sources, and mapping of components to matrix indeces
    Y, sources, node_map, total_dim = build_matrix(components)
    print(node_map)

    # solve! get LU, node voltages and branch currents of original circuit
    lu = solve_LU(Y)
    VI = get_node_and_branch_currents(lu, sources)
    print_solution(VI, node_map)

    # select output node for adjoint input
    target_node_for_sensitivity = 1

    # get all node and branch voltages of adjoint circuit
    PsiPhi = solve_adjoint(lu, target_node_for_sensitivity, node_map, total_dim)

    # get the sensitivities of all components
    sensitivities = get_all_sensitivities(components, VI, PsiPhi, node_map)
    print(sensitivities)

    # find the standard deviation on the output voltage
    std_dev = estimate_std_dev(sensitivities, components)
    print(std_dev)

    print(f"output voltage V = {VI[target_node_for_sensitivity]} $\pm$ {std_dev} V")

    # # Example Bode Plot
    run_bode_plot(test_directory + "ac_lowpass.txt", output_node=2, start_freq=10, stop_freq=100000, points=200, name = "lowpass")
    run_bode_plot(test_directory + "ac_resonance.txt", output_node=3, start_freq=10, stop_freq=100000, points=200, name = "resonance")
