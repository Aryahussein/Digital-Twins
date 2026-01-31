from node_index import invert_node_index

def map_voltages(V, node_index):
    """
       Maps the internal nodes back to the user-defined
       nodes and returns the voltage corresponding to
       that user-defined node:
           node_index: dict {node_number: index}
           V: vector containing nodal voltages
       """
    index_node = invert_node_index(node_index)
    voltages = {0: 0.0}
    for i, v in enumerate(V):
        voltages[index_node[i]] = v
    return voltages