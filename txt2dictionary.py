import re

# =============================================================================
# SPICE VALUE PARSER
# =============================================================================
def parse_value(value_str):
    """
    Parses SPICE number formats (case-insensitive).
    """
    if not value_str:
        return 0.0

    val_str = value_str.upper()
    
    multipliers = {
        'T': 1e12, 'G': 1e9, 'MEG': 1e6, 'X': 1e6, 'K': 1e3,
        'MIL': 25.4e-6, 'M': 1e-3, 'U': 1e-6, 'N': 1e-9, 'P': 1e-12, 'F': 1e-15
    }

    match = re.match(r'^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)(\D*)$', val_str)
    
    if not match:
        return 0.0

    number_part = float(match.group(1))
    suffix_part = match.group(2)

    # if suffix_part.startswith("MEG"):
    #     return number_part * multipliers["MEG"]
    
    for suffix, mult in multipliers.items():
        if suffix_part.startswith(suffix):
            return number_part * mult
            
    return number_part

# =============================================================================
# PARAMETER EXTRACTOR
# =============================================================================
def parse_params(tokens):
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
                func_name = token.split('(')[0]
                full_func = token
                if ")" not in token:
                    while True:
                        next_chunk = next(it)
                        full_func += " " + next_chunk
                        if ")" in next_chunk:
                            break
                source_def["tran"] = parse_tran_func(full_func)
            else:
                try:
                    val = parse_value(token)
                    if source_def["dc"] == 0.0: source_def["dc"] = val
                except: pass
    except StopIteration: pass
    return source_def

def parse_tran_func(func_str):
    match = re.match(r'(\w+)\((.*)\)', func_str, re.IGNORECASE)
    if not match: return None
    
    name = match.group(1).upper()
    args = [parse_value(x) for x in match.group(2).replace(',', ' ').split()]
    
    if name == "PULSE":
        keys = ["V1", "V2", "TD", "TR", "TF", "PW", "PER"]
        return {"type": "PULSE", **dict(zip(keys, args + [0]*(7-len(args))))}
    elif name == "SIN":
        keys = ["VOFF", "VAMP", "FREQ", "TD", "PHASE"]
        return {"type": "SIN", **dict(zip(keys, args + [0]*(5-len(args))))}
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

    # print(lines)

    full_lines = []
    current_line = ""
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith('*'): continue 
        
        if line.startswith('+'):
            current_line += " " + line[1:].strip()
        else:
            # print(current_line)
            if current_line: full_lines.append(current_line)

            current_line = line

    if current_line: full_lines.append(current_line)

    for line in full_lines:
        print(line)

    # print(full_lines)

    for line in full_lines:
        tokens = line.split()
        cmd = tokens[0].upper()

        # print(cmd)
        
        if cmd.startswith('.'):
            if cmd == ".MODEL":
                mname = tokens[1].upper()
                rest_of_line = " ".join(tokens[3:]).replace('(', ' ').replace(')', ' ')
                mparams, _ = parse_params(rest_of_line.split())
                models[mname] = mparams
            elif cmd == ".TRAN":
                analyses[cmd] = {"step": parse_value(tokens[1]), "stop": parse_value(tokens[2])}
            elif cmd == ".AC":
                analyses[cmd] = {"type": tokens[1].upper(), "num_points": int(tokens[2]), "start": parse_value(tokens[3]), "stop": parse_value(tokens[4])}
            elif cmd == ".OP":
                analyses[cmd] = {}
            continue

        name = tokens[0].upper()
        type_char = name[0]
        
        # 1. PASSIVE (R, L, C)
        if type_char in ['R', 'L', 'C']:
            n1, n2 = int(tokens[1]), int(tokens[2])
            val = parse_value(tokens[3])
            components[name] = {"type": type_char, "n1": n1, "n2": n2, "value": val}
            
        # 2. DIODE (D) -> Handle both Value (1e-14) and Model Name
        elif type_char == 'D':
            n1, n2 = int(tokens[1]), int(tokens[2])
            token3 = tokens[3]
            
            # Heuristic: If it looks like a number, treat as Value (Is). Else, Model.
            try:
                val = float(token3)
                components[name] = {"type": 'D', "n1": n1, "n2": n2, "value": val}
            except ValueError:
                # If strict float fails, check if valid SPICE number (like 1u)
                val = parse_value(token3)
                if val != 0.0 or token3.strip() == "0":
                     components[name] = {"type": 'D', "n1": n1, "n2": n2, "value": val}
                else:
                     components[name] = {"type": 'D', "n1": n1, "n2": n2, "model": token3.upper()}

        # 3. MOSFET (M)
        elif type_char == 'M':
            n_d, n_g, n_s, n_b = [int(x) for x in tokens[1:5]]
            model_name = tokens[5].upper()
            params, _ = parse_params(tokens[6:])
            components[name] = {"type": 'M', "n_d": n_d, "n_g": n_g, "n_s": n_s, "n_b": n_b, "model": model_name, "params": params}
            
        # 4. SOURCES (V, I)
        elif type_char in ['V', 'I']:
            n1, n2 = int(tokens[1]), int(tokens[2])
            source_data = parse_source_def(tokens[3:])
            
            # FIX: Only add 'source' key if 'tran' is not None to prevent NoneType crash
            comp_data = {
                "type": type_char, "n1": n1, "n2": n2, 
                "value": source_data["dc"], "ac": source_data["ac_mag"]
            }
            if source_data["tran"] is not None:
                comp_data["source"] = source_data["tran"]
            
            components[name] = comp_data
            
        # 5. VCCS (G)
        elif type_char == 'G':
            n1, n2, n3, n4 = [int(x) for x in tokens[1:5]]
            val = parse_value(tokens[5])
            components[name] = {"type": 'G', "n1": n1, "n2": n2, "n3": n3, "n4": n4, "value": val}

    # Attach .MODEL data
    for name, comp in components.items():
        if "model" in comp:
            m_name = comp["model"]
            if m_name in models:
                comp["model_params"] = models[m_name]
            else:
                print(f"Warning: Model {m_name} not found for {name}")

    return components, analyses
