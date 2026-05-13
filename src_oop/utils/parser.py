"""
SPICE Netlist Parsing Module.

This module reads standard SPICE-formatted text files, handles line continuations,
extracts scaling suffixes (k, MEG, u, etc.), and compiles the text into structured 
data dictionaries that the `Circuit` factory can use to instantiate objects.
Includes support for HSPICE/Spectre statistical distributions.
"""

import re
import numpy as np

class NetlistParser:
    """Parses a SPICE netlist into component and analysis dictionaries."""

    def __init__(self):
        self.components = {}
        self.models = {}
        self.analyses = {}
        self.current_line = 0
        self.raw_line_text = ""
        
        self.dispatch_registry = {
            'R': self._parse_passive, 
            'L': self._parse_passive, 
            'C': self._parse_passive,
            'D': self._parse_diode,
            'M': self._parse_mosfet,
            'V': self._parse_source, 
            'I': self._parse_source,
            'G': self._parse_vccs,
            'E': self._parse_vcvs,
            'F': self._parse_cccs,
            'H': self._parse_ccvs
        }

    def parse(self, file_path):
        self.__init__() 
        with open(file_path, 'r') as f:
            lines = f.readlines()

        full_lines = self._preprocess_lines(lines)

        for line_num, line in full_lines:
            line = re.sub(
                r'(?i)(AGAUSS|GAUSS|AUNIF|UNIF)\s*\([^)]+\)', 
                lambda m: m.group(0).replace(' ', '').replace('\t', ''), 
                line
            )
            self.current_line = line_num
            self.raw_line_text = line
            
            tokens = line.split()
            cmd = tokens[0].upper()
            
            if cmd.startswith('.'):
                self._parse_command(tokens)
            else:
                self._parse_component(tokens)

        self._attach_models()
        
        if not self.analyses:
            self.analyses[".OP"] = {}

        return self.components, self.analyses

    def _parse_command(self, tokens):
        cmd = tokens[0].upper()

        if cmd == ".MODEL":
            if len(tokens) < 3: 
                self._throw_error("Malformed .MODEL command.")
                
            mname = tokens[1].upper()
            mtype = tokens[2].upper()
            
            # Rebuild the parameter string safely
            rest_of_line = " ".join(tokens[3:]).strip()
            
            # Only strip the OUTERMOST parentheses if they wrap the entire block
            if rest_of_line.startswith('(') and rest_of_line.endswith(')'):
                rest_of_line = rest_of_line[1:-1].strip()
            elif rest_of_line.startswith('('):
                rest_of_line = rest_of_line[1:].strip()
                
            # Now extract parameters. The internal GAUSS() parentheses are safe!
            mparams, stat_params, _ = self._extract_params(rest_of_line.split())
            self.models[mname] = {"type": mtype, "params": mparams, "stat_params": stat_params}

        elif cmd == ".TRAN":
            if len(tokens) < 3: 
                self._throw_error("Expected at least '.TRAN TSTEP TSTOP'.")
            tstep, tstop = self._parse_value(tokens[1]), self._parse_value(tokens[2])
            tmax = self._parse_value(tokens[4]) if len(tokens) > 4 else 0.0
            dt = tstep if tstep > 0 else (tmax if tmax > 0 else (tstop / 1000.0 if tstop > 0 else 1e-5))
            self.analyses[".TRAN"] = {"step": dt, "stop": tstop}
            
        elif cmd == ".AC":
            sweep_type = tokens[1].upper()
            if sweep_type == "LIST":
                freq_list = [self._parse_value(t) for t in tokens[2:]]
                self.analyses[".AC"] = {"sweep_type": "LIST", "num_points": freq_list, "start": freq_list[0], "stop": freq_list[-1]}
            else:
                if len(tokens) < 5: 
                    self._throw_error("Expected '.AC TYPE POINTS START STOP'.")
                self.analyses[".AC"] = {"sweep_type": sweep_type, "num_points": int(tokens[2]), "start": self._parse_value(tokens[3]), "stop": self._parse_value(tokens[4])}   

        elif cmd == ".OP":
            self.analyses[".OP"] = {}

        elif cmd == ".DC":
            if len(tokens) < 5: self._throw_error("Expected '.DC Source Start Stop Step'.")
            self.analyses[".DC"] = {"source": tokens[1].upper(), "start": self._parse_value(tokens[2]), "stop": self._parse_value(tokens[3]), "step": self._parse_value(tokens[4])}

        elif cmd == ".OPTIONS":
            for token in tokens[1:]:
                if '=' in token:
                    key, val = token.split('=', 1)
                    if key.upper() == "METHOD":
                        self.analyses["OPTIONS"] = self.analyses.get("OPTIONS", {})
                        self.analyses["OPTIONS"]["method"] = val.upper()

        elif cmd == ".IC":
            self.analyses[".IC"] = self.analyses.get(".IC", {})
            for token in tokens[1:]:
                match = re.match(r'V\((.*?)\)=(.*)', token, re.IGNORECASE)
                if match:
                    self.analyses[".IC"][self._parse_node(match.group(1))] = self._parse_value(match.group(2))

    def _parse_component(self, tokens):
        name = tokens[0].upper()
        type_char = name[0]
        parser_func = self.dispatch_registry.get(type_char)
        if parser_func:
            self.components[name] = parser_func(tokens, name, type_char)
        else:
            print(f"Warning: Unsupported component '{type_char}' on line {self.current_line}")

    # =========================================================================
    # COMPONENT MINI-PARSERS
    # =========================================================================
    def _parse_passive(self, tokens, name, type_char):
        if len(tokens) < 4: 
            self._throw_error(f"Missing nodes or value for '{name}'. Format: Name N1 N2 Value")
            
        nom_val, stat_data = self._parse_stat_string(tokens[3])
        comp = {
            "type": type_char, 
            "n1": self._parse_node(tokens[1]), "n2": self._parse_node(tokens[2]), 
            "value": nom_val,
            "stat_params": {}
        }
        if stat_data: 
            comp["stat_params"]["VALUE"] = stat_data
        return comp

    def _parse_diode(self, tokens, name, type_char):
        if len(tokens) < 3: self._throw_error(f"Missing nodes for diode '{name}'")
        comp = {"type": type_char, "n1": self._parse_node(tokens[1]), "n2": self._parse_node(tokens[2]), "stat_params": {}}
        if len(tokens) >= 4:
            nom_val, stat_data = self._parse_stat_string(tokens[3])
            if nom_val != 0.0 or tokens[3].strip() == "0":
                comp["value"] = nom_val
                if stat_data: comp["stat_params"]["VALUE"] = stat_data
            else:
                comp["model"] = tokens[3].upper()
        return comp

    def _parse_mosfet(self, tokens, name, type_char):
        if len(tokens) < 6: self._throw_error(f"Malformed MOSFET '{name}'")
        inst_params, stat_params, _ = self._extract_params(tokens[6:]) 
        return {
            "type": type_char, 
            "n_d": self._parse_node(tokens[1]), "n_g": self._parse_node(tokens[2]), 
            "n_s": self._parse_node(tokens[3]), "n_b": self._parse_node(tokens[4]), 
            "model": tokens[5].upper(), 
            "inst_params": inst_params,
            "stat_params": stat_params
        }

    def _parse_source(self, tokens, name, type_char):
        if len(tokens) < 4: self._throw_error(f"Malformed source '{name}'")
        # Ensure statistical variations on DC sources are captured
        nom_val, stat_data = self._parse_stat_string(tokens[3])
        source_data = self._parse_source_def(tokens[3:])
        
        comp_data = {
            "type": type_char, 
            "n1": self._parse_node(tokens[1]), "n2": self._parse_node(tokens[2]), 
            "value": nom_val if stat_data else source_data["dc"], 
            "ac_mag": source_data["ac_mag"], 
            "ac_phase": source_data["ac_phase"],
            "stat_params": {}
        }
        if stat_data:
            comp_data["stat_params"]["VALUE"] = stat_data
        if source_data["tran"] is not None:
            comp_data["source"] = source_data["tran"]
        return comp_data

    def _parse_vccs(self, tokens, name, type_char): return {"type": type_char, "n1": self._parse_node(tokens[1]), "n2": self._parse_node(tokens[2]), "n3": self._parse_node(tokens[3]), "n4": self._parse_node(tokens[4]), "value": self._parse_value(tokens[5])}
    def _parse_vcvs(self, tokens, name, type_char): return {"type": type_char, "n1": self._parse_node(tokens[1]), "n2": self._parse_node(tokens[2]), "n3": self._parse_node(tokens[3]), "n4": self._parse_node(tokens[4]), "value": self._parse_value(tokens[5])}
    def _parse_cccs(self, tokens, name, type_char): return {"type": type_char, "n1": self._parse_node(tokens[1]), "n2": self._parse_node(tokens[2]), "controlling_source": tokens[3].upper(), "value": self._parse_value(tokens[4])}
    def _parse_ccvs(self, tokens, name, type_char): return {"type": type_char, "n1": self._parse_node(tokens[1]), "n2": self._parse_node(tokens[2]), "controlling_source": tokens[3].upper(), "value": self._parse_value(tokens[4])}

    # =========================================================================
    # STRING & VALUE PARSING UTILITIES
    # =========================================================================
    @staticmethod
    def _parse_stat_string(val_str):
        """
        Parses HSPICE statistical distribution wrappers to support Yield Analysis.
        
        Supported Syntaxes:
        - GAUSS(nominal, rel_tol, sigma_cut)  -> e.g., GAUSS(1k, 0.05, 3) = 1k ±5% at 3-sigma
        - AGAUSS(nominal, abs_tol, sigma_cut) -> e.g., AGAUSS(5V, 0.1V, 3) = 5V ±0.1V at 3-sigma
        - UNIF(nominal, rel_tol)              -> e.g., UNIF(10u, 0.10) = 10u ±10% uniform
        - AUNIF(nominal, abs_tol)             -> e.g., AUNIF(0.7, 0.05) = 0.7 ±0.05 uniform
        
        Returns:
            tuple: (nominal_float_value, statistical_data_dictionary)
                   If no wrapper is found, returns (nominal, None).
        """
        match = re.match(r'^(AGAUSS|GAUSS|AUNIF|UNIF)\((.*)\)$', val_str, re.IGNORECASE)
        if not match:
            return NetlistParser._parse_value(val_str), None
            
        func_name = match.group(1).upper()
        args = [NetlistParser._parse_value(x.strip()) for x in match.group(2).split(',')]
        
        nom = args[0]
        stat_dict = {"dist_type": func_name}
        
        # Normalize all variations down to a relative tolerance percentage and a sigma-level
        # so the Woodbury Yield Engine can process them uniformly without complex logic.
        if func_name == "UNIF":
            stat_dict["tol"] = args[1]
            stat_dict["sigma"] = 3.0 # Uniform mapped to equivalent 3-sigma bounds
        elif func_name == "AUNIF":
            stat_dict["tol"] = abs(args[1] / nom) if nom != 0 else 0.0
            stat_dict["sigma"] = 3.0
        elif func_name == "GAUSS":
            stat_dict["tol"] = args[1]
            stat_dict["sigma"] = args[2] if len(args) > 2 else 3.0
        elif func_name == "AGAUSS":
            stat_dict["tol"] = abs(args[1] / nom) if nom != 0 else 0.0
            stat_dict["sigma"] = args[2] if len(args) > 2 else 3.0
            
        return nom, stat_dict

    @staticmethod
    def _parse_value(value_str):
        if not value_str: return 0.0
        val_str = value_str.upper()
        multipliers = {'T': 1e12, 'G': 1e9, 'MEG': 1e6, 'K': 1e3, 'MIL': 25.4e-6, 'M': 1e-3, 'U': 1e-6, 'N': 1e-9, 'P': 1e-12, 'F': 1e-15}
        match = re.match(r'^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)(\D*)$', val_str)
        if not match: return 0.0
        num, suffix = float(match.group(1)), match.group(2)
        for s, mult in multipliers.items():
            if suffix.startswith(s): return num * mult
        return num

    @staticmethod
    def _extract_params(tokens):
        params, stat_params, leftovers = {}, {}, []
        for token in tokens:
            if '=' in token:
                key, val_str = token.split('=', 1)
                
                # Check if the value contains a statistical wrapper
                nom_val, stat_data = NetlistParser._parse_stat_string(val_str)
                
                key_upper = key.upper()
                params[key_upper] = nom_val
                if stat_data:
                    stat_params[key_upper] = stat_data
            else:
                leftovers.append(token)
        return params, stat_params, leftovers

    @staticmethod
    def _parse_node(node_str):
        if node_str == "0" or node_str.upper() == "GND": return 0
        try: return int(node_str)
        except ValueError: return node_str

    def _parse_source_def(self, tokens):
        source_def = {"dc": 0.0, "ac_mag": 0.0, "ac_phase": 0.0, "tran": None}
        i = 0
        while i < len(tokens):
            token = tokens[i].upper()
            if token == "DC":
                if i + 1 < len(tokens):
                    source_def["dc"] = self._parse_value(tokens[i+1])
                    i += 1
            elif token == "AC":
                if i + 1 < len(tokens):
                    source_def["ac_mag"] = self._parse_value(tokens[i+1])
                    i += 1
                    if i + 1 < len(tokens) and re.match(r'^[+-]?\d', tokens[i+1]):
                        source_def["ac_phase"] = self._parse_value(tokens[i+1])
                        i += 1
            elif "(" in token and token.split('(')[0] not in ["AGAUSS", "GAUSS", "UNIF", "AUNIF"]: 
                full_func = token
                while ")" not in full_func and i + 1 < len(tokens):
                    i += 1
                    full_func += " " + tokens[i]
                source_def["tran"] = self._parse_tran_func(full_func)
            else:
                if re.match(r'^[+-]?\d', token) and source_def["dc"] == 0.0:
                    source_def["dc"] = self._parse_value(token)
            i += 1
        return source_def

    def _parse_tran_func(self, func_str):
        match = re.match(r'(\w+)\((.*)\)', func_str, re.IGNORECASE)
        if not match: return None
        name = match.group(1).upper()
        args = [self._parse_value(x) for x in match.group(2).replace(',', ' ').split()]
        if name == "PULSE": return {"type": "PULSE", **dict(zip(["V1", "V2", "TD", "TR", "TF", "PW", "PER"], args + [0]*(7-len(args))))}
        elif name in ["SINE","SIN", "COS"]: return {"type": name, **dict(zip(["VOFF", "VAMP", "FREQ", "TD", "PHASE"], args + [0]*(5-len(args))))}
        elif name == "PWL": return {"type": "PWL", "TIME_VOLTAGE_PAIRS": [(args[i], args[i+1]) for i in range(0, len(args)-1, 2)]}
        return None

    # =========================================================================
    # ERROR HANDLING & FILE PRE-PROCESSING
    # =========================================================================
    def _throw_error(self, msg):
        raise ValueError(f"Parse Error on Line {self.current_line}:\n{msg}\n-> {self.raw_line_text}")

    def _preprocess_lines(self, lines):
        full_lines, current_logical_line, start_line_num = [], "", 0
        for i, line in enumerate(lines):
            line_num, stripped = i + 1, line.strip()
            if not stripped or stripped.startswith('*'): continue 
            if stripped.startswith('+'):
                if current_logical_line: current_logical_line += " " + stripped[1:].strip()
                else: current_logical_line, start_line_num = stripped[1:].strip(), line_num
            else:
                if current_logical_line: full_lines.append((start_line_num, current_logical_line))
                current_logical_line, start_line_num = stripped, line_num
        if current_logical_line: full_lines.append((start_line_num, current_logical_line))
        return full_lines

    def _attach_models(self):
        for name, comp in self.components.items():
            if "model" in comp:
                m_name = comp["model"]
                if m_name in self.models:
                    comp["model_params"] = self.models[m_name]["params"]
                    comp["type"] = self.models[m_name]["type"]
                    # Inherit model-level statistical parameters if they exist
                    if "stat_params" in self.models[m_name]:
                        comp.setdefault("stat_params", {}).update(self.models[m_name]["stat_params"])
                else:
                    print(f"Warning: Model '{m_name}' not found for '{name}'")
