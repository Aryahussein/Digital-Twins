from txt2dictionary import parse_netlist
from node_index import build_node_index
from solver import solve_sparse
from postprocessing import map_voltages
from netlist2Ymatrix import generate_stamps
import matplotlib.pyplot as plt

def dc_nodal_analysis(netlist):

    # Convert netlist to components
    components = parse_netlist(netlist)

    print(components)


    # Set ground as node 0 and re-assign node indices for the rest
    node_index, N = build_node_index(components)

    # Build Y matrix and I vector from components
    Y, I = generate_stamps(components, node_index)
    print(Y.todense())
    
    # Show sparsity pattern of Y matrix (optional)
    plt.spy(Y)
    plt.show()

    # Use existing sparse matrix solver
    V = solve_sparse(Y, I)

    # Map calculated voltages to user-defined nodes
    voltages = map_voltages(V, node_index)
    for node in sorted(voltages):
        print(f"Node {node}: {voltages[node]:.6f} V")
    

if __name__ == "__main__":
    netlist = "testfiles/example_vccs.txt"
    dc_nodal_analysis(netlist)