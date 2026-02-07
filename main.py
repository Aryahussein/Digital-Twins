from txt2dictionary import parse_netlist
from node_index import build_node_index
from solver import solve_sparse
from postprocessing import map_voltages
from assembleYmatrix import generate_stamps
import matplotlib.pyplot as plt
import numpy as np
from tools import run_bode_plot

def dc_nodal_analysis(netlist, w = 0):

    # Convert netlist to components
    components = parse_netlist(netlist)

    # print(components)

    # Set ground as node 0 and re-assign node indices for the rest
    node_index, mna_index, total_dim = build_node_index(components)

    # Build Y matrix and I vector from components
    Y, I = generate_stamps(components, node_index, mna_index, total_dim, w=w)
    # print(Y.todense())
    
    # Show sparsity pattern of Y matrix (optional)
    plt.spy(Y)
    plt.show()

    # Use existing sparse matrix solver
    V = solve_sparse(Y, I)

    # 1. Node Voltages
    voltages = map_voltages(V, node_index)
    for node in sorted(voltages):
        val = voltages[node]
        if w == 0:
            print(f"Node {node}: {val.real:.6f} V")
        else:
            mag = np.abs(val)
            phase = np.degrees(np.angle(val))
            print(f"Node {node}: {mag:.6f} V ∠ {phase:.2f}°")

    # 2. Branch Currents (from Voltage Sources & Inductors)
    if mna_index:
        print("\n--- Branch Currents ---")
        sorted_mna = sorted(mna_index.items(), key=lambda item: item[1])
        for name, idx in sorted_mna:
            val = V[idx]
            if w == 0:
                print(f"{name}: {val.real:.6f} A")
            else:
                mag = np.abs(val)
                phase = np.degrees(np.angle(val))
                print(f"{name}: {mag:.6f} A ∠ {phase:.2f}°")
    


if __name__ == "__main__":
    test_directory = "testfiles/"
    netlist = test_directory + "/test_book_floating_voltage_source.txt"
    dc_nodal_analysis(netlist)
    # Example Bode Plot
    run_bode_plot(test_directory + "ac_lowpass.txt", output_node=2, start_freq=10, stop_freq=100000, points=200, name = "lowpass")
    run_bode_plot(test_directory + "ac_resonance.txt", output_node=3, start_freq=10, stop_freq=100000, points=200, name = "resonance")
