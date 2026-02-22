def build_node_index(components):
    """
    Builds a single mapping for all unknowns (voltages and currents).
    
    Ground (node 0) is implicitly handled — it is never in the map.
    All other nodes that appear in components get sequential indices.
    Voltage sources (V) and inductors (L) get extra indices for branch currents.
    
    Returns:
        node_map: dict {node_number_or_component_name: matrix_index}
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


def validate_node(node, node_map):
    """
    Validate that a node exists in the circuit.
    Returns the index for non-ground nodes, None for ground (0),
    and raises KeyError for unknown nodes.
    """
    if node == 0:
        return None
    if node not in node_map:
        raise KeyError(f"Node {node} not found in circuit. Known nodes: {sorted(k for k in node_map if isinstance(k, int))}")
    return node_map[node]

def invert_node_index(node_index):
    return {i: node for node, i in node_index.items()}
