def build_node_index(components):
    """
    Builds a single mapping for all unknowns (voltages and currents).
    
    Returns:
        var_map: dict {name_or_node: matrix_index}
        total_dim: total size of the matrix
    """
    nodes = set()
    for comp in components.values():
        for key in ["n1", "n2", "n3", "n4"]:
            val = comp.get(key, 0)
            if val != 0:
                nodes.add(val)

    node_list = sorted(nodes)
    node_map = {}
    current_idx = 0

    # 1. Map Nodes (Voltages)
    for node in node_list:
        node_map[node] = current_idx
        current_idx += 1

    # 2. Map MNA Components (Branch Currents)
    for name in components:
        if name.startswith(("V", "L")):
            node_map[name] = current_idx
            current_idx += 1
            
    return node_map

def invert_node_index(node_index):
    return {i: node for node, i in node_index.items()}
