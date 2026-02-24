import re

# =============================================================================
# SPICE VALUE PARSER
# =============================================================================
def parse_value(value_str):
    """
    Parses SPICE number formats (case-insensitive).
    Examples: 1k, 2.2u, 1e-3, 5MEG
    """
    if not value_str:
        return 0.0

    val_str = value_str.upper().strip()

    multipliers = {
        'T': 1e12, 'G': 1e9, 'MEG': 1e6, 'X': 1e6, 'K': 1e3,
        'MIL': 25.4e-6, 'M': 1e-3, 'U': 1e-6, 'N': 1e-9, 'P': 1e-12, 'F': 1e-15
    }

    match = re.match(r'^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)(\D*)$', val_str)
    if not match:
        return 0.0

    number_part = float(match.group(1))
    suffix_part = match.group(2)

    for suffix, mult in multipliers.items():
        if suffix_part.startswith(suffix):
            return number_part * mult

    return number_part


# =============================================================================
# PARAMETER EXTRACTOR
# =============================================================================
def parse_params(tokens):
    """
    Split tokens into key=value params and leftovers.
    Returns:
      params: dict of {KEY: float_value}
      leftovers: list[str] of tokens that weren't key=value
    """
    params = {}
    leftovers = []

    for token in tokens:
        if '=' in token:
            key, val_str = token.split('=', 1)
            params[key.upper()] = parse_value(val_str)
        else:
            leftovers.append(token)

    return params, leftovers


# =============================================================================
# SOURCE PARSERS
# =============================================================================
def parse_source_def(tokens):
    """
    Parse independent source definitions:
      V1 n+ n- DC 5 AC 1
      I1 n+ n- 1m
      V1 n+ n- PULSE(...)
    """
    source_def = {"dc": 0.0, "ac_mag": 0.0, "tran": None}
    it = iter(tokens)

    try:
        while True:
            token = next(it).upper()

            if token == "DC":
                source_def["dc"] = parse_value(next(it))

            elif token == "AC":
                source_def["ac_mag"] = parse_value(next(it))

            elif "(" in token:
                # transient source function like PULSE(...) or SIN(...)
                full_func = token
                if ")" not in token:
                    while True:
                        next_chunk = next(it)
                        full_func += " " + next_chunk
                        if ")" in next_chunk:
                            break
                source_def["tran"] = parse_tran_func(full_func)

            else:
                # Bare number means DC value if DC not explicitly provided
                val = parse_value(token)
                if source_def["dc"] == 0.0:
                    source_def["dc"] = val

    except StopIteration:
        pass

    return source_def


def parse_tran_func(func_str):
    match = re.match(r'(\w+)\((.*)\)', func_str, re.IGNORECASE)
    if not match:
        return None

    name = match.group(1).upper()
    args = [parse_value(x) for x in match.group(2).replace(',', ' ').split()]

    if name == "PULSE":
        keys = ["V1", "V2", "TD", "TR", "TF", "PW", "PER"]
        return {"type": "PULSE", **dict(zip(keys, args + [0] * (7 - len(args))))}

    if name == "SIN":
        keys = ["VOFF", "VAMP", "FREQ", "TD", "PHASE"]
        return {"type": "SIN", **dict(zip(keys, args + [0] * (5 - len(args))))}

    return None


# =============================================================================
# MAIN NETLIST PARSER
# =============================================================================
def parse_netlist(file_path):
    components = {}
    models = {}
    analyses = {}

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Handle line continuations with '+'
    full_lines = []
    current_line = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('*'):
            continue

        if line.startswith('+'):
            current_line += " " + line[1:].strip()
        else:
            if current_line:
                full_lines.append(current_line)
            current_line = line

    if current_line:
        full_lines.append(current_line)

    for line in full_lines:
        tokens = line.split()
        cmd = tokens[0].upper()

        # -------------------------
        # Dot commands
        # -------------------------
        if cmd.startswith('.'):
            if cmd == ".MODEL":
                mname = tokens[1].upper()
                # tokens[2] = model type, tokens[3:] = params in parentheses
                rest_of_line = " ".join(tokens[3:]).replace('(', ' ').replace(')', ' ')
                mparams, _ = parse_params(rest_of_line.split())
                models[mname] = mparams

            elif cmd == ".TRAN":
                analyses[cmd] = {"step": parse_value(tokens[1]), "stop": parse_value(tokens[2])}

            elif cmd == ".AC":
                analyses[cmd] = {
                    "type": tokens[1].upper(),
                    "num_points": int(tokens[2]),
                    "start": parse_value(tokens[3]),
                    "stop": parse_value(tokens[4]),
                }

            elif cmd == ".OP":
                analyses[cmd] = {}

            continue

        name = tokens[0].upper()
        type_char = name[0]

        # -------------------------
        # 1) PASSIVES: R, L, C
        # -------------------------
        if type_char in ['R', 'L', 'C']:
            n1, n2 = int(tokens[1]), int(tokens[2])
            val = parse_value(tokens[3])
            components[name] = {
                "name": name,
                "type": type_char,
                "n1": n1,
                "n2": n2,
                "value": val
            }

        # -------------------------
        # 2) DIODE: D
        # -------------------------
        elif type_char == 'D':
            n1, n2 = int(tokens[1]), int(tokens[2])
            token3 = tokens[3]

            # Heuristic: if looks numeric, treat as Is value, otherwise model name.
            try:
                val = float(token3)
                components[name] = {
                    "name": name,
                    "type": 'D',
                    "n1": n1,
                    "n2": n2,
                    "value": val
                }
            except ValueError:
                val = parse_value(token3)
                if val != 0.0 or token3.strip() == "0":
                    components[name] = {
                        "name": name,
                        "type": 'D',
                        "n1": n1,
                        "n2": n2,
                        "value": val
                    }
                else:
                    components[name] = {
                        "name": name,
                        "type": 'D',
                        "n1": n1,
                        "n2": n2,
                        "model": token3.upper()
                    }

        # -------------------------
        # 3) MOSFET: M
        # -------------------------
        elif type_char == 'M':
            n_d, n_g, n_s, n_b = [int(x) for x in tokens[1:5]]
            model_name = tokens[5].upper()
            params, _ = parse_params(tokens[6:])
            components[name] = {
                "name": name,
                "type": 'M',
                "n_d": n_d,
                "n_g": n_g,
                "n_s": n_s,
                "n_b": n_b,
                "model": model_name,
                "params": params
            }

        # -------------------------
        # 4) INDEPENDENT SOURCES: V, I
        # -------------------------
        elif type_char in ['V', 'I']:
            n1, n2 = int(tokens[1]), int(tokens[2])
            source_data = parse_source_def(tokens[3:])

            comp_data = {
                "name": name,
                "type": type_char,
                "n1": n1,
                "n2": n2,
                "value": source_data["dc"],
                "ac": source_data["ac_mag"],
            }
            if source_data["tran"] is not None:
                comp_data["source"] = source_data["tran"]

            components[name] = comp_data

        # -------------------------
        # 5) VCCS: G
        # -------------------------
        elif type_char == 'G':
            n1, n2, n3, n4 = [int(x) for x in tokens[1:5]]
            val = parse_value(tokens[5])
            components[name] = {
                "name": name,
                "type": 'G',
                "n1": n1,
                "n2": n2,
                "n3": n3,
                "n4": n4,
                "value": val
            }

        # -------------------------
        # 6) OP-AMP (custom): A
        # Format:
        #   A1 out vp vm Vsat=5 k=1e6
        # -------------------------
        elif type_char == 'A':
            out = int(tokens[1])
            vp = int(tokens[2])
            vm = int(tokens[3])

            params, leftovers = parse_params(tokens[4:])
            vsat = params.get("VSAT", 1.0)
            k_val = params.get("K", 1e3)

            components[name] = {
                "name": name,
                "type": "A",
                "out": out,
                "vp": vp,
                "vm": vm,
                "Vsat": vsat,
                "k": k_val,
                "params": params,
                "leftovers": leftovers
            }

        else:
            # Unknown component: ignore or warn
            # print(f"Warning: unsupported element line: {line}")
            pass

    # Attach .MODEL data
    for cname, comp in components.items():
        if "model" in comp:
            m_name = comp["model"]
            if m_name in models:
                comp["model_params"] = models[m_name]
            else:
                print(f"Warning: Model {m_name} not found for {cname}")

    return components, analyses