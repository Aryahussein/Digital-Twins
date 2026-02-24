def build_node_index(components):
    """
    Builds a single mapping for all unknowns (node voltages + extra MNA currents).

    Returns:
        node_map: dict mapping
            - node number (int) -> matrix index
            - component name (str) for MNA branch current -> matrix index

    Notes:
        - Ground is node 0 and is NOT included as an unknown.
        - MNA extra variables are added for:
            * Independent voltage sources (V*)
            * Inductors (L*)  [treated like a 0V source in DC/MNA]
            * Op-amps (A*)    [our nonlinear VCVS-like constraint element]
    """
    nodes = set()

    # 1) Collect node numbers used by all components
    for comp in components.values():
        ctype = comp.get("type", "").upper()

        # Traditional node keys used by most elements
        for key in ("n1", "n2", "n3", "n4"):
            val = comp.get(key, 0)
            if isinstance(val, int) and val != 0:
                nodes.add(val)

        # Custom op-amp element uses out/vp/vm keys
        if ctype == "A":
            for key in ("out", "vp", "vm"):
                val = comp.get(key, 0)
                if isinstance(val, int) and val != 0:
                    nodes.add(val)

    node_list = sorted(nodes)
    node_map = {}
    current_idx = 0

    # 2) Map node voltages (unknowns)
    for node in node_list:
        node_map[node] = current_idx
        current_idx += 1

    # 3) Map extra MNA unknowns (branch currents / constraint currents)
    #    Add one per V, L, and A element.
    for name, comp in components.items():
        ctype = comp.get("type", "").upper()

        # Keep backward compatibility with old naming-based check
        if name.startswith(("V", "L")) or ctype in ("V", "L"):
            node_map[name] = current_idx
            current_idx += 1

        # New: op-amp element (A)
        elif name.startswith("A") or ctype == "A":
            node_map[name] = current_idx
            current_idx += 1

    return node_map


def invert_node_index(node_index):
    return {i: node for node, i in node_index.items()}