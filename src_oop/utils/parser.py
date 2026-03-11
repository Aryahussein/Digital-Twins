"""
SPICE Netlist Parsing Module.

This module reads standard SPICE-formatted text files, handles line continuations,
extracts scaling suffixes (k, MEG, u, etc.), and compiles the text into structured 
data dictionaries that the `Circuit` factory can use to instantiate objects.
"""

import re
import numpy as np

class NetlistParser:
    """Parses a SPICE netlist into component and analysis dictionaries.

    Attributes:
        components (dict): Parsed component data keyed by component name.
        models (dict): Parsed .MODEL definitions.
        analyses (dict): Parsed simulation commands (.TRAN, .AC, .DC, .OP).
        current_line (int): Tracks the line number for accurate error reporting.
        raw_line_text (str): Tracks the raw text of the current line.
        dispatch_registry (dict): Maps SPICE prefix letters to parsing methods.
    """

    def __init__(self):
        """Initializes the parser state and routing registry."""
        self.components = {}
        self.models = {}
        self.analyses = {}
        
        # Line tracking for precise error messages
        self.current_line = 0
        self.raw_line_text = ""
        
        # Dispatch Registry: Maps first letter to its specific mini-parser
        self.dispatch_registry = {
            'R': self._parse_passive, 
            'L': self._parse_passive, 
            'C': self._parse_passive,
            'D': self._parse_diode,
            'M': self._parse_mosfet,
            'V': self._parse_source, 
            'I': self._parse_source,
            'G': self._parse_vccs
        }

    # =========================================================================
    # PUBLIC API
    # =========================================================================
    def parse(self, file_path):
        """Main entry point to parse a netlist file.

        Args:
            file_path (str): The absolute or relative path to the .txt/.cir file.

        Returns:
            tuple: (components_dict, analyses_dict) containing the raw data 
            ready for the Circuit and Simulator objects.
        """
        self.__init__() # Reset state on new parse
        
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Step 1: Clean and group the lines
        full_lines = self._preprocess_lines(lines)

        # Step 2: Parse each logical line
        for line_num, line in full_lines:
            self.current_line = line_num
            self.raw_line_text = line
            
            tokens = line.split()
            cmd = tokens[0].upper()
            
            if cmd.startswith('.'):
                self._parse_command(tokens)
            else:
                self._parse_component(tokens)

        # Step 3: Post-processing
        self._attach_models()
        
        # Default to DC Operating Point if no analysis was requested
        if not self.analyses:
            self.analyses[".OP"] = {}

        return self.components, self.analyses

    # =========================================================================
    # CORE ROUTING METHODS
    # =========================================================================
    def _parse_command(self, tokens):
        """Routes dot-commands (.TRAN, .AC, .MODEL, .DC, .OP)."""
        cmd = tokens[0].upper()
        
        if cmd == ".MODEL":
            clean_line = " ".join(tokens).replace('(', ' ( ').replace(')', ' ) ')
            new_tokens = clean_line.split()

            if len(new_tokens) < 3: 
                self._throw_error("Malformed .MODEL command.")
                
            mname = new_tokens[1].upper()
            mtype = new_tokens[2].upper()
            
            rest_of_line = " ".join(new_tokens[3:]).replace('(', '').replace(')', '')
            mparams, _ = self._extract_params(rest_of_line.split())
            
            self.models[mname] = {"type": mtype, "params": mparams}            

        elif cmd == ".TRAN":
            if len(tokens) < 3: 
                self._throw_error("Expected at least '.TRAN TSTEP TSTOP'.")
                
            tstep, tstop = self._parse_value(tokens[1]), self._parse_value(tokens[2])
            tmax = self._parse_value(tokens[4]) if len(tokens) > 4 else 0.0
            dt = tstep if tstep > 0 else (tmax if tmax > 0 else (tstop / 1000.0 if tstop > 0 else 1e-5))
            self.analyses[".TRAN"] = {"step": dt, "stop": tstop}
            
        elif cmd == ".AC":
            if len(tokens) < 5: 
                self._throw_error("Expected '.AC TYPE POINTS START STOP'.")
                
            self.analyses[".AC"] = {
                "sweep_type": tokens[1].upper(), # Sync: Simulator expects 'sweep_type'
                "num_points": int(tokens[2]), 
                "start": self._parse_value(tokens[3]), 
                "stop": self._parse_value(tokens[4])
            }
            
        elif cmd in [".OP", ".DC"]:
            freq = 0.0 
            if len(tokens) > 1:
                try:
                    freq = self._parse_value(tokens[1])
                except Exception:
                    self._throw_error(f"Invalid frequency argument for .OP: '{tokens[1]}'")
                    
            self.analyses[".OP"] = {"freq": freq}

    def _parse_component(self, tokens):
        """Routes component parsing via the registry."""
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
        return {
            "type": type_char, 
            "n1": self._parse_node(tokens[1]), "n2": self._parse_node(tokens[2]), 
            "value": self._parse_value(tokens[3])
        }

    def _parse_diode(self, tokens, name, type_char):
        if len(tokens) < 4: 
            self._throw_error(f"Missing nodes or model for diode '{name}'. Format: Name N+ N- Model/Value")
        
        comp = {"type": type_char, "n1": self._parse_node(tokens[1]), "n2": self._parse_node(tokens[2])}
        token3 = tokens[3]
        
        try:
            comp["value"] = float(token3)
        except ValueError:
            val = self._parse_value(token3)
            if val != 0.0 or token3.strip() == "0":
                comp["value"] = val
            else:
                comp["model"] = token3.upper()
        return comp

    def _parse_mosfet(self, tokens, name, type_char):
        if len(tokens) < 6: 
            self._throw_error(f"Malformed MOSFET '{name}'. Format: Name Nd Ng Ns Nb Model [Params]")
            
        inst_params, _ = self._extract_params(tokens[6:]) 
        return {
            "type": type_char, 
            "n_d": self._parse_node(tokens[1]), "n_g": self._parse_node(tokens[2]), 
            "n_s": self._parse_node(tokens[3]), "n_b": self._parse_node(tokens[4]), 
            "model": tokens[5].upper(), 
            "inst_params": inst_params 
        }

    def _parse_source(self, tokens, name, type_char):
        if len(tokens) < 4: 
            self._throw_error(f"Malformed source '{name}'. Format: Name N+ N- [DC/AC/TRAN value]")
            
        source_data = self._parse_source_def(tokens[3:])
        comp_data = {
            "type": type_char, 
            "n1": self._parse_node(tokens[1]), "n2": self._parse_node(tokens[2]), 
            "value": source_data["dc"], 
            "ac_mag": source_data["ac_mag"], 
            "ac_phase": source_data["ac_phase"]
        }
        if source_data["tran"] is not None:
            comp_data["source"] = source_data["tran"]
        return comp_data

    def _parse_vccs(self, tokens, name, type_char):
        if len(tokens) < 6: 
            self._throw_error(f"Malformed VCCS '{name}'. Format: Name N+ N- NC+ NC- Gain")
        return {
            "type": type_char, 
            "n1": self._parse_node(tokens[1]), "n2": self._parse_node(tokens[2]), 
            "n3": self._parse_node(tokens[3]), "n4": self._parse_node(tokens[4]), 
            "value": self._parse_value(tokens[5])
        }

    # =========================================================================
    # STRING & VALUE PARSING UTILITIES
    # =========================================================================
    @staticmethod
    def _parse_node(node_str):
        """Allows nodes to be strings (e.g., 'vdd') or integers."""
        if node_str == "0" or node_str.upper() == "GND":
            return 0
        try:
            return int(node_str)
        except ValueError:
            return node_str

    def _parse_source_def(self, tokens):
        """Extracts DC, AC, and Transient source definitions from a line."""
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
            elif "(" in token: 
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
        """Parses SPICE transient source functions like PULSE, SIN, PWL."""
        match = re.match(r'(\w+)\((.*)\)', func_str, re.IGNORECASE)
        if not match: return None
        
        name = match.group(1).upper()
        args = [self._parse_value(x) for x in match.group(2).replace(',', ' ').split()]
        
        if name == "PULSE":
            keys = ["V1", "V2", "TD", "TR", "TF", "PW", "PER"]
            return {"type": "PULSE", **dict(zip(keys, args + [0]*(7-len(args))))}
            
        elif name in ["SIN", "COS"]:
            keys = ["VOFF", "VAMP", "FREQ", "TD", "PHASE"]
            return {"type": name, **dict(zip(keys, args + [0]*(5-len(args))))}
            
        # Extension: Translate flat PWL lists into coordinate tuples
        elif name == "PWL":
            pairs = [(args[i], args[i+1]) for i in range(0, len(args)-1, 2)]
            return {"type": "PWL", "TIME_VOLTAGE_PAIRS": pairs}
            
        return None

    @staticmethod
    def _parse_value(value_str):
        """Converts SPICE numbers with scaling suffixes (e.g., '10k', '5u') to floats."""
        if not value_str: return 0.0
        val_str = value_str.upper()
        multipliers = {
            'T': 1e12, 'G': 1e9, 'MEG': 1e6, 'X': 1e6, 'K': 1e3,
            'MIL': 25.4e-6, 'M': 1e-3, 'U': 1e-6, 'N': 1e-9, 'P': 1e-12, 'F': 1e-15
        }
        match = re.match(r'^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)(\D*)$', val_str)
        if not match: return 0.0

        num, suffix = float(match.group(1)), match.group(2)
        for s, mult in multipliers.items():
            if suffix.startswith(s):
                return num * mult
        return num

    @staticmethod
    def _extract_params(tokens):
        """Extracts 'KEY=VALUE' parameters into a dictionary."""
        params, leftovers = {}, []
        for token in tokens:
            if '=' in token:
                key, val_str = token.split('=', 1)
                params[key.upper()] = NetlistParser._parse_value(val_str)
            else:
                leftovers.append(token)
        return params, leftovers

    # =========================================================================
    # ERROR HANDLING & FILE PRE-PROCESSING
    # =========================================================================
    def _throw_error(self, msg):
        """Centralized error formatting with exact line numbers."""
        raise ValueError(f"Parse Error on Line {self.current_line}:\n{msg}\n-> {self.raw_line_text}")

    def _preprocess_lines(self, lines):
        """Handles '+' continuations and returns (line_number, full_string) tuples."""
        full_lines = []
        current_logical_line = ""
        start_line_num = 0

        for i, line in enumerate(lines):
            line_num = i + 1
            stripped = line.strip()
            
            if not stripped or stripped.startswith('*'):
                continue 
            
            if stripped.startswith('+'):
                if current_logical_line:
                    current_logical_line += " " + stripped[1:].strip()
                else:
                    current_logical_line = stripped[1:].strip()
                    start_line_num = line_num
            else:
                if current_logical_line:
                    full_lines.append((start_line_num, current_logical_line))
                current_logical_line = stripped
                start_line_num = line_num
                
        if current_logical_line:
            full_lines.append((start_line_num, current_logical_line))
            
        return full_lines

    def _attach_models(self):
        """Binds .MODEL parameters to their corresponding component instances."""
        for name, comp in self.components.items():
            if "model" in comp:
                m_name = comp["model"]
                if m_name in self.models:
                    comp["model_params"] = self.models[m_name]["params"]
                    comp["model_type"] = self.models[m_name]["type"]
                else:
                    print(f"Warning: Model '{m_name}' not found for '{name}'")
