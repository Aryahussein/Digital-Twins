def parse_value(value_str):
    """
    Convert a value string with units like 'k', 'm', 'u' to float.
    """
    multipliers = {'k': 1e3, 'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12}
    value_str = value_str.lower().strip()

    if value_str[-1] in multipliers:
        return float(value_str[:-1]) * multipliers[value_str[-1]]
    else:
        return float(value_str)


def parse_netlist(file_path):
    """
    Reads a netlist from a text file and returns a dictionary of components.

    Returns:
        dict: {component_name: {"n1": int, "n2": int, "value": float}, ...}
    """
    components = {}

    with open(file_path, 'r') as f:
        for line in f:
            # Skip empty lines and comments
            line = line.strip()
            if not line or line.startswith('*'):
                continue

            tokens = line.split()
            if len(tokens) < 4:
                raise ValueError(f"Invalid netlist line: {line}")

            name, n1, n2, value_str = tokens[0], tokens[1], tokens[2], tokens[3]

            # Convert nodes to integers if possible (assuming '0' is ground)
            n1 = int(n1)
            n2 = int(n2)
            value = parse_value(value_str)

            components[name] = {"n1": n1, "n2": n2, "value": value}

    return components

