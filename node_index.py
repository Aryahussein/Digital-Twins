def build_node_index(components):
    """
    Builds a mapping from node numbers (excluding ground '0') to matrix indices.
    Returns:
        node_index: dict {node_number: index}
        N: total number of unknown nodes
    """
    nodes = set()
    for comp in components.values():
        if comp["n1"] != 0:
            nodes.add(comp["n1"])
        if comp["n2"] != 0:
            nodes.add(comp["n2"])
        if comp["n3"] != 0:
            nodes.add(comp["n3"])
        if comp["n4"] != 0:
            nodes.add(comp["n4"])

    node_list = sorted(nodes)
    node_index = {node: idx for idx, node in enumerate(node_list)}
    N = len(node_list)
    return node_index, N

def invert_node_index(node_index):
    return {i: node for node, i in node_index.items()}
