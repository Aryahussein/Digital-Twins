from node_index import invert_node_index


def map_voltages(V, node_index):
    """
    Maps the internal matrix indices back to user-defined node numbers.
    V: solution vector (contains Node Voltages AND Branch Currents)
    node_index: dict {node_number: index}
    """
    voltages = {0: 0.0} # Initialize ground
    for node, idx in node_index.items():
        voltages[node] = V[idx]
        
    return voltages
