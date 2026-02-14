import re

def parse_time_source(expr):
    """
    Parse time-dependent source expressions:
    PULSE(V1 V2 TD TR TF PW PER)
    SIN(VOFF VAMP FREQ [PHASE])
    COS(VOFF VAMP FREQ [PHASE])
    """

    expr = expr.strip()
    match = re.match(r'(\w+)\((.*)\)', expr, re.IGNORECASE)
    if not match:
        return None

    source_type = match.group(1).upper()
    params = match.group(2).replace(',', ' ').split()

    params = [parse_value(p) for p in params]

    if source_type == "PULSE":
        if len(params) != 7:
            raise ValueError("PULSE requires 7 parameters")
        return {
            "type": "PULSE",
            "V1": params[0],
            "V2": params[1],
            "TD": params[2],
            "TR": params[3],
            "TF": params[4],
            "PW": params[5],
            "PER": params[6]
        }

    elif source_type in ["SIN", "COS"]:
        if len(params) < 3:
            raise ValueError(f"{source_type} requires at least 3 parameters")

        return {
            "type": source_type,
            "VOFF": params[0],
            "VAMP": params[1],
            "FREQ": params[2],
            "PHASE": params[3] if len(params) > 3 else 0.0
        }

    else:
        raise ValueError(f"Unsupported time source type: {source_type}")


def parse_value(value_str):
    """
    Convert a value string with units like 'k', 'm', 'u' to float.
    """
    multipliers = {'M': 1e6, 'k': 1e3, 'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12}
    value_str = value_str.lower().strip()

    if value_str[-1] in multipliers:
        return float(value_str[:-1]) * multipliers[value_str[-1]]
    else:
        return float(value_str)


def parse_netlist(file_path):
    """
    Reads a netlist and returns a dictionary of components.
    Now supports time-dependent voltage sources.
    """
    components = {}

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('*'):
                continue

            tokens = line.split()
            name = tokens[0]

            # 2-node elements (R, I, V, etc.)
            if len(tokens) >= 4 and tokens[0][0].upper() in ['R', 'I', 'V', 'L', 'C']:
                n1 = int(tokens[1])
                n2 = int(tokens[2])
                n3 = 0
                n4 = 0

                value_part = " ".join(tokens[3:])

                # Check if this is a time-dependent source
                time_source = parse_time_source(value_part)

                if time_source:
                    components[name] = {
                        "n1": n1,
                        "n2": n2,
                        "n3": n3,
                        "n4": n4,
                        "source": time_source
                    }
                else:
                    value = parse_value(tokens[3])
                    components[name] = {
                        "n1": n1,
                        "n2": n2,
                        "n3": n3,
                        "n4": n4,
                        "value": value
                    }

            # 4-node elements (VCCS etc.)
            elif len(tokens) == 6:
                name, n1, n2, n3, n4, value_str = tokens
                components[name] = {
                    "n1": int(n1),
                    "n2": int(n2),
                    "n3": int(n3),
                    "n4": int(n4),
                    "value": parse_value(value_str)
                }

            else:
                raise ValueError(f"Invalid netlist line: {line}")

    return components

if __name__ == '__main__':
    file = r"testfiles/transient_parser.txt"
    components = parse_netlist(file)
    print(components)
