def build_node_index(components):
    """Builds mapping for unknowns (voltages and currents)."""
    nodes = set()
    for comp in components.values():
        for key in ["n1", "n2", "n3", "n4"]:
            val = comp.get(key, 0)
            if val != 0: nodes.add(val)

    node_list = sorted(nodes)
    node_map = {}
    current_idx = 0

    # 1. Map Nodes
    for node in node_list:
        node_map[node] = current_idx
        current_idx += 1

    # 2. Map Branch Currents (V sources and Inductors)
    for name in components:
        if name.startswith(("V", "L")):
            node_map[name] = current_idx
            current_idx += 1
            
    return node_map, current_idx