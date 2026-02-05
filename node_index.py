def build_node_index(components):
    """
    Builds mappings for nodes AND MNA components (Voltage sources, Inductors).
    
    Returns:
        node_index: dict {node_number: matrix_index}
        mna_index: dict {component_name: matrix_index}
        total_dim: total size of the matrix (Num Nodes + Num MNA components)
    """
    nodes = set()
    for comp in components.values():
        # Check all potential node connections
        for key in ["n1", "n2", "n3", "n4"]:
            if key in comp and comp[key] != 0:
                nodes.add(comp[key])

    node_list = sorted(nodes)
    node_index = {node: idx for idx, node in enumerate(node_list)}
    num_nodes = len(node_list)

    # Identify MNA Components (Voltage Sources and Inductors)
    # These components add extra rows/columns to the matrix
    mna_index = {}
    current_idx = num_nodes

    for name in components:
        # Check if component is Voltage Source (V) or Inductor (L)
        if name.startswith("V") or name.startswith("L"):
            mna_index[name] = current_idx
            current_idx += 1
            
    total_dim = current_idx

    return node_index, mna_index, total_dim

def invert_node_index(node_index):
    return {i: node for node, i in node_index.items()}
