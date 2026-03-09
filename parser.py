import re
import numpy as np
import logging

logger = logging.getLogger(__name__)


class NetlistParser:
    def __init__(self):
        self._reset()

        # Dispatch Registry: Maps first letter to its specific mini-parser
        self.dispatch_registry = {
            "R": self._parse_passive,
            "L": self._parse_passive,
            "C": self._parse_passive,
            "D": self._parse_diode,
            "M": self._parse_mosfet,
            "Q": self._parse_bjt,
            "V": self._parse_source,
            "I": self._parse_source,
            "G": self._parse_vccs,
            "E": self._parse_vcvs,
            "F": self._parse_cccs,
            "H": self._parse_ccvs,
            "O": self._parse_opamp,
        }

    def _reset(self):
        """Reset all mutable state for a fresh parse."""
        self.components = {}
        self.models = {}
        self.analyses = {}
        self.current_line = 0
        self.raw_line_text = ""

    # =========================================================================
    # COMPONENT MINI-PARSERS
    # =========================================================================
    def _parse_opamp(self, tokens, name, type_char):
        if len(tokens) < 5:
            self._throw_error(
                f"Malformed opamp '{name}'. "
                "Format: Oname nout nref nplus nminus [A or A=...]"
            )

        nout = int(tokens[1])
        nref = int(tokens[2])
        nplus = int(tokens[3])
        nminus = int(tokens[4])

        A = 1e5  # default gain
        if len(tokens) >= 6:
            t = tokens[5]
            if "=" in t:
                k, v = t.split("=", 1)
                if k.strip().upper() == "A":
                    A = float(v)
            else:
                A = float(t)

        return {
            "type": "O",
            "n1": nout,
            "n2": nref,
            "n3": nplus,
            "n4": nminus,
            "value": float(A),
        }

    # =========================================================================
    # PUBLIC API
    # =========================================================================
    def parse(self, file_path):
        """Main entry point to parse a netlist file."""
        self._reset()

        with open(file_path, "r") as f:
            lines = f.readlines()

        # Step 1: Clean and group the lines
        full_lines = self._preprocess_lines(lines)

        # Step 2: Parse each logical line
        for line_num, line in full_lines:
            self.current_line = line_num
            self.raw_line_text = line

            tokens = line.split()
            cmd = tokens[0].upper()

            if cmd == ".END":
                break
            elif cmd.startswith("."):
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
        """Routes dot-commands (.TRAN, .AC, .MODEL)"""
        cmd = tokens[0].upper()

        if cmd == ".MODEL":
            clean_line = " ".join(tokens).replace("(", " ( ").replace(")", " ) ")
            new_tokens = clean_line.split()

            if len(new_tokens) < 3:
                self._throw_error("Malformed .MODEL command.")

            mname = new_tokens[1].upper()

            # Known device types for the type field
            KNOWN_TYPES = {"NMOS", "PMOS", "NPN", "PNP", "D"}

            # Detect whether token[2] is a type keyword or already a parameter.
            # If token[2] contains '=' it's a parameter, meaning the type field
            # was omitted and the model name IS the type (common shorthand).
            candidate = new_tokens[2].upper()
            if candidate in KNOWN_TYPES:
                # Standard format: .MODEL name type (params...)
                mtype = candidate
                param_tokens = new_tokens[3:]
            elif "=" in candidate:
                # Shorthand: .MODEL NMOS VTO=0.7 KP=100u ...
                # Model name doubles as type, params start at token[2]
                mtype = mname
                param_tokens = new_tokens[2:]
            else:
                # Assume it's a type we don't recognize yet
                mtype = candidate
                param_tokens = new_tokens[3:]

            rest_of_line = " ".join(param_tokens).replace("(", "").replace(")", "")
            mparams, _ = self._extract_params(rest_of_line.split())

            self.models[mname] = {"type": mtype, "params": mparams}

        elif cmd == ".TRAN":
            # Standard SPICE: .TRAN TSTEP TSTOP [TSTART] [TMAX]
            if len(tokens) < 3:
                self._throw_error("Expected at least '.TRAN TSTEP TSTOP'.")

            tstep = self._parse_value(tokens[1])
            tstop = self._parse_value(tokens[2])
            tstart = self._parse_value(tokens[3]) if len(tokens) > 3 else 0.0
            tmax = self._parse_value(tokens[4]) if len(tokens) > 4 else 0.0

            dt = tstep if tstep > 0 else (tmax if tmax > 0 else tstop / 1000.0)
            self.analyses[".TRAN"] = {"step": dt, "stop": tstop, "start": tstart}

        elif cmd == ".AC":
            if len(tokens) < 5:
                self._throw_error("Expected '.AC TYPE POINTS START STOP'.")

            self.analyses[".AC"] = {
                "type": tokens[1].upper(),
                "num_points": int(tokens[2]),
                "start": self._parse_value(tokens[3]),
                "stop": self._parse_value(tokens[4]),
            }

        elif cmd in [".OP", ".DC"]:
            freq = 0.0
            if len(tokens) > 1:
                freq = self._parse_value(tokens[1])
            self.analyses[".OP"] = {"freq": freq}

        else:
            logger.warning("Ignoring unrecognized command '%s' on line %d", cmd, self.current_line)

    def _parse_component(self, tokens):
        """Routes component parsing via the registry."""
        name = tokens[0].upper()
        type_char = name[0]

        parser_func = self.dispatch_registry.get(type_char)
        if parser_func:
            self.components[name] = parser_func(tokens, name, type_char)
        else:
            logger.warning(
                "Unsupported component '%s' on line %d", type_char, self.current_line
            )

    # =========================================================================
    # COMPONENT MINI-PARSERS
    # =========================================================================
    def _parse_passive(self, tokens, name, type_char):
        if len(tokens) < 4:
            self._throw_error(
                f"Missing nodes or value for '{name}'. Format: Name N1 N2 Value"
            )
        return {
            "type": type_char,
            "n1": int(tokens[1]),
            "n2": int(tokens[2]),
            "value": self._parse_value(tokens[3]),
        }

    def _parse_diode(self, tokens, name, type_char):
        if len(tokens) < 4:
            self._throw_error(
                f"Missing nodes or model for diode '{name}'. "
                "Format: Name N+ N- Model/Value"
            )

        comp = {"type": type_char, "n1": int(tokens[1]), "n2": int(tokens[2])}
        token3 = tokens[3]

        # Try numeric parse first; if it fails, treat as model name
        try:
            comp["value"] = float(token3)
        except ValueError:
            try:
                val = self._parse_value(token3)
                comp["value"] = val
            except ValueError:
                comp["model"] = token3.upper()
        return comp

    def _parse_mosfet(self, tokens, name, type_char):
        if len(tokens) < 6:
            self._throw_error(
                f"Malformed MOSFET '{name}'. "
                "Format: Name Nd Ng Ns Nb Model [Params]"
            )

        inst_params, _ = self._extract_params(tokens[6:])
        return {
            "type": type_char,
            "n_d": int(tokens[1]),
            "n_g": int(tokens[2]),
            "n_s": int(tokens[3]),
            "n_b": int(tokens[4]),
            "model": tokens[5].upper(),
            "inst_params": inst_params,
        }

    def _parse_source(self, tokens, name, type_char):
        if len(tokens) < 4:
            self._throw_error(
                f"Malformed source '{name}'. "
                "Format: Name N+ N- [DC/AC/TRAN value]"
            )

        source_data = self._parse_source_def(tokens[3:])
        comp_data = {
            "type": type_char,
            "n1": int(tokens[1]),
            "n2": int(tokens[2]),
            "value": source_data["dc"],
            "ac_mag": source_data["ac_mag"],
            "ac_phase": source_data["ac_phase"],
        }
        if source_data["tran"] is not None:
            comp_data["source"] = source_data["tran"]
        return comp_data

    def _parse_vccs(self, tokens, name, type_char):
        if len(tokens) < 6:
            self._throw_error(
                f"Malformed VCCS '{name}'. Format: Name N+ N- NC+ NC- Gain"
            )
        return {
            "type": type_char,
            "n1": int(tokens[1]),
            "n2": int(tokens[2]),
            "n3": int(tokens[3]),
            "n4": int(tokens[4]),
            "value": self._parse_value(tokens[5]),
        }

    def _parse_vcvs(self, tokens, name, type_char):
        """Parse VCVS: Ename N+ N- NC+ NC- Gain"""
        if len(tokens) < 6:
            self._throw_error(
                f"Malformed VCVS '{name}'. Format: Ename N+ N- NC+ NC- Gain"
            )
        return {
            "type": type_char,
            "n1": int(tokens[1]),
            "n2": int(tokens[2]),
            "n3": int(tokens[3]),
            "n4": int(tokens[4]),
            "value": self._parse_value(tokens[5]),
        }

    def _parse_cccs(self, tokens, name, type_char):
        """Parse CCCS: Fname N+ N- Vcontrol Gain"""
        if len(tokens) < 5:
            self._throw_error(
                f"Malformed CCCS '{name}'. Format: Fname N+ N- Vcontrol Gain"
            )
        return {
            "type": type_char,
            "n1": int(tokens[1]),
            "n2": int(tokens[2]),
            "v_control": tokens[3].upper(),
            "value": self._parse_value(tokens[4]),
        }

    def _parse_ccvs(self, tokens, name, type_char):
        """Parse CCVS: Hname N+ N- Vcontrol Gain"""
        if len(tokens) < 5:
            self._throw_error(
                f"Malformed CCVS '{name}'. Format: Hname N+ N- Vcontrol Gain"
            )
        return {
            "type": type_char,
            "n1": int(tokens[1]),
            "n2": int(tokens[2]),
            "v_control": tokens[3].upper(),
            "value": self._parse_value(tokens[4]),
        }

    def _parse_bjt(self, tokens, name, type_char):
        """Parse BJT: Qname NC NB NE [NSubstrate] Model [params]"""
        if len(tokens) < 5:
            self._throw_error(
                f"Malformed BJT '{name}'. "
                "Format: Qname NC NB NE Model [params]"
            )

        # Standard 3-terminal: Q NC NB NE Model
        # Optional 4-terminal: Q NC NB NE NS Model
        # Heuristic: if token[4] is a number, it's a substrate node
        nc = int(tokens[1])
        nb = int(tokens[2])
        ne = int(tokens[3])

        try:
            ns = int(tokens[4])
            # tokens[4] was a number → it's substrate, model is tokens[5]
            model_token = tokens[5].upper() if len(tokens) > 5 else "NPN"
            extra_tokens = tokens[6:]
        except ValueError:
            # tokens[4] is the model name
            ns = 0
            model_token = tokens[4].upper()
            extra_tokens = tokens[5:]

        inst_params, _ = self._extract_params(extra_tokens)

        return {
            "type": type_char,
            "n_c": nc,
            "n_b": nb,
            "n_e": ne,
            "n_s": ns,
            "model": model_token,
            "inst_params": inst_params,
        }

    # =========================================================================
    # STRING & VALUE PARSING UTILITIES
    # =========================================================================
    def _parse_source_def(self, tokens):
        source_def = {"dc": 0.0, "ac_mag": 0.0, "ac_phase": 0.0, "tran": None}
        i = 0
        while i < len(tokens):
            token = tokens[i].upper()

            if token == "DC":
                if i + 1 < len(tokens):
                    source_def["dc"] = self._parse_value(tokens[i + 1])
                    i += 1
            elif token == "AC":
                if i + 1 < len(tokens):
                    source_def["ac_mag"] = self._parse_value(tokens[i + 1])
                    i += 1
                    if i + 1 < len(tokens) and re.match(r"^[+-]?\d", tokens[i + 1]):
                        source_def["ac_phase"] = self._parse_value(tokens[i + 1])
                        i += 1
            elif "(" in token:
                full_func = token
                while ")" not in full_func and i + 1 < len(tokens):
                    i += 1
                    full_func += " " + tokens[i]
                source_def["tran"] = self._parse_tran_func(full_func)
            else:
                if re.match(r"^[+-]?\d", token) and source_def["dc"] == 0.0:
                    source_def["dc"] = self._parse_value(token)
            i += 1
        return source_def

    def _parse_tran_func(self, func_str):
        match = re.match(r"(\w+)\((.*)\)", func_str, re.IGNORECASE)
        if not match:
            return None

        name = match.group(1).upper()
        args = [
            self._parse_value(x) for x in match.group(2).replace(",", " ").split()
        ]

        if name == "PULSE":
            keys = ["V1", "V2", "TD", "TR", "TF", "PW", "PER"]
            return {"type": "PULSE", **dict(zip(keys, args + [0] * (7 - len(args))))}
        elif name in ["SIN", "COS"]:
            keys = ["VOFF", "VAMP", "FREQ", "TD", "PHASE"]
            result = {"type": name, **dict(zip(keys, args + [0] * (5 - len(args))))}
            result["PHASE"] = np.radians(result["PHASE"])
            return result
        return None

    @staticmethod
    def _parse_value(value_str):
        """Parse a numeric string with optional SI suffix.

        Raises ValueError for completely unparseable strings instead of
        silently returning 0.0.
        """
        if not value_str:
            return 0.0

        val_str = value_str.strip().upper()
        multipliers = {
            "T": 1e12,
            "G": 1e9,
            "MEG": 1e6,
            "X": 1e6,
            "K": 1e3,
            "MIL": 25.4e-6,
            "M": 1e-3,
            "U": 1e-6,
            "N": 1e-9,
            "P": 1e-12,
            "F": 1e-15,
        }
        match = re.match(r"^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)(\D*)$", val_str)
        if not match:
            raise ValueError(f"Cannot parse value: '{value_str}'")

        num = float(match.group(1))
        suffix = match.group(2)

        if not suffix:
            return num

        for s, mult in multipliers.items():
            if suffix.startswith(s):
                return num * mult

        # Suffix present but unrecognized — warn but return number
        logger.warning("Unrecognized suffix '%s' in value '%s', using numeric part only", suffix, value_str)
        return num

    @staticmethod
    def _extract_params(tokens):
        params, leftovers = {}, []
        for token in tokens:
            if "=" in token:
                key, val_str = token.split("=", 1)
                params[key.upper()] = NetlistParser._parse_value(val_str)
            else:
                leftovers.append(token)
        return params, leftovers

    # =========================================================================
    # ERROR HANDLING & FILE PRE-PROCESSING
    # =========================================================================
    def _throw_error(self, msg):
        """Centralized error formatting with exact line numbers."""
        raise ValueError(
            f"Parse Error on Line {self.current_line}:\n{msg}\n-> {self.raw_line_text}"
        )

    def _preprocess_lines(self, lines):
        """
        Handles '+' continuations and returns (line_number, full_string) tuples.
        This ensures errors map perfectly to the user's actual text file.
        """
        full_lines = []
        current_logical_line = ""
        start_line_num = 0

        for i, line in enumerate(lines):
            line_num = i + 1
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or stripped.startswith("*"):
                continue

            # Handle line continuations
            if stripped.startswith("+"):
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

        # Push the very last line
        if current_logical_line:
            full_lines.append((start_line_num, current_logical_line))

        return full_lines

    def _attach_models(self):
        """Binds model parameters to component instances at the end."""
        for name, comp in self.components.items():
            if "model" in comp:
                m_name = comp["model"]
                if m_name in self.models:
                    comp["model_params"] = self.models[m_name]["params"]
                    comp["model_type"] = self.models[m_name]["type"]
                else:
                    logger.warning("Model '%s' not found for '%s'", m_name, name)
