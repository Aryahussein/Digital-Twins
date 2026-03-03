import re
import numpy as np
from typing import Dict, List, int, str, Match, Any, Callable


class NetlistParser:
    def __init__(self):
        # Internal state
        self.components: List[Dict] = []
        self.analyses: list[Dict[str, Any]] = []

        # Line tracking for precise error messages
        self.current_line: int = 0
        self.raw_line_text: str = ""

        # Dispatch Registry: Maps first letter to its specific mini-parser
        self.dispatch_registry: Dict[str, Callable] = {
            "R": self._parse_passive,
            "L": self._parse_passive,
            "C": self._parse_passive,
            "D": self._parse_diode,
            "M": self._parse_mosfet,
            "V": self._parse_source,
            "I": self._parse_source,
            "G": self._parse_vccs,
            "O": self._parse_opamp,
        }

    # =========================================================================
    # PUBLIC API
    # =========================================================================
    def parse(self, file_path: str) -> None:
        """Main entry point to parse a netlist file. Stores data in class attributes.
        Args:
            file_path (str): Path to the netlist file to be parsed.
        Returns:
            None
        """
        self.__init__()  # Reset state on new parse

        with open(file_path, "r") as f:
            lines: list[str] = f.readlines()

        # Step 1: Clean and group the lines
        full_lines: list[tuple[int, str]] = self._preprocess_lines(lines)

        # Step 2: Parse each logical line
        for line_num, line in full_lines:
            self.current_line: int = line_num
            self.raw_line_text: str = line

            tokens: list[str] = line.split()
            cmd: str = tokens[0].upper()

            if cmd.startswith("."):
                self._parse_command(tokens)
            else:
                self._parse_component(tokens)

        # Step 3: Post-processing
        # self._attach_models()

        # Default to DC Operating Point if no analysis was requested
        if not self.analyses:
            self.analyses.append({"analysis": ".OP", "params": {"freq": 0.0}})

    # =========================================================================
    # CORE ROUTING METHODS
    # =========================================================================
    def _parse_command(self, tokens: list[str]) -> None:
        """Routes dot-commands (.TRAN, .AC, .MODEL)
        Args:
            tokens (list[str]): The tokenized line, where tokens[0] is the command.
        Returns:
            None
        """
        cmd: str = tokens[0].upper()

        # al Unclear what how to handle .MODEL. Particularly leftovers.
        if cmd == ".MODEL":
            clean_line: str = " ".join(tokens).replace("(", " ( ").replace(")", " ) ")
            new_tokens: list[str] = clean_line.split()

            if len(new_tokens) < 3:
                self._throw_error("Malformed .MODEL command.")

            mname: str = new_tokens[1].upper()
            mtype: str = new_tokens[2].upper()

            rest_of_line: str = (
                " ".join(new_tokens[3:]).replace("(", "").replace(")", "")
            )
            mparams, _ = self._extract_params(rest_of_line.split())

            self.models[mname] = {"type": mtype, "params": mparams}

        elif cmd == ".TRAN":
            if len(tokens) < 3:
                self._throw_error("Expected at least '.TRAN TSTEP TSTOP'.")

            tstep: float = self._parse_value(tokens[1])
            tstop: float = self._parse_value(tokens[2])
            tmax: float = self._parse_value(tokens[4]) if len(tokens) > 4 else 0.0
            dt: float = (
                tstep
                if tstep > 0
                else (tmax if tmax > 0 else (tstop / 1000.0 if tstop > 0 else 1e-5))
            )

            transient_params: Dict = {"step": dt, "stop": tstop}
            self.analyses.append({"analysis": ".TRAN", "params": transient_params})

        elif cmd == ".AC":
            if len(tokens) < 5:
                self._throw_error("Expected '.AC TYPE POINTS START STOP'.")

            type: str = tokens[1].upper()
            num_points: int = int(tokens[2])
            start: int = self._parse_value(tokens[3])
            stop: int = self._parse_value(tokens[4])
            params: Dict = {
                "type": type,
                "num_points": num_points,
                "start": start,
                "stop": stop,
            }

            self.analyses.append({"analysis": ".AC", "params": params})

        elif cmd in [".OP", ".DC"]:
            # Default frequency is 0.0 (Standard DC OP)
            freq: float = 0.0

            # If the user typed something after .OP (e.g., .OP 1k), parse it!
            if len(tokens) > 1:
                try:
                    freq = self._parse_value(tokens[1])
                except Exception:
                    self._throw_error(
                        f"Invalid frequency argument for .OP: '{tokens[1]}'"
                    )
            # al : User should specify the type of analysis. No analysis specified should mean error!
            self.analyses.append({"analysis": cmd, "params": {"freq": freq}})

    def _parse_component(self, tokens: list[str]) -> None:
        """Routes component parsing via the registry.
        Args:
            tokens (list[str]): Tokenized line represent component in the netlist.
        Returns:
            None
        """
        name: str = tokens[0].upper()
        type_char: str = name[0]
        parser_func: Callable = self.dispatch_registry.get(type_char)
        if parser_func:
            params: Dict = parser_func(tokens, name, type_char)
            self.components.append({"name": name, "params": params})
        else:
            self._throw_error(
                f"Error: Unsupported component '{type_char}' on line {self.current_line}!"
            )

    # =========================================================================
    # COMPONENT MINI-PARSERS
    # =========================================================================
    def _parse_passive(self, tokens: list[str], name: str, type_char: str) -> Dict:
        """
        Parses Resistors, Capacitors, and Inductors. Format: Name N1 N2 Value
        Args:
            tokens (list[str]): Tokenized line representing the component.
            name (str): The name of the component (e.g., R1, C2).
            type_char (str): The type character ('R', 'C', or 'L').
        Returns:
            passive_element (Dict): A dictionary containing the parsed parameters for the component.
        """
        if len(tokens) < 4:
            self._throw_error(
                f"Missing nodes or value for '{name}'. Format: Name N1 N2 Value"
            )

        n1: int = int(tokens[1])
        n2: int = int(tokens[2])
        value: float = self._parse_value(tokens[3])
        passive_element: Dict = {
            "type": type_char,
            "n1": n1,
            "n2": n2,
            "value": value,
        }
        return passive_element

    def _parse_diode(self, tokens: list[str], name: str, type_char: str) -> Dict:
        """
        Parses a diode. Format: Name N+ N- Model
        Args:
            tokens (list[str]): Tokenized line representing the diode.
            name (str): The name of the diode (e.g., D1).
            type_char (str): The type character ('D').
        Returns:
            diode (Dict): A dictionary containing the parsed parameters for the diode.
        """
        if len(tokens) < 4:
            self._throw_error(
                f"Missing nodes or model for diode '{name}'. Format: Name N+ N- Model"
            )
        n1: int = int(tokens[1])
        n2: int = int(tokens[2])
        model: str = tokens[3].upper()
        diode: Dict = {"type": type_char, "n1": n1, "n2": n2, "model": model}
        return diode

    def _parse_mosfet(self, tokens: list[str], name: str, type_char: str) -> Dict:
        """
        Parses a MOSFET. Format: Name Nd Ng Ns Nb Model [Params]
        Args:
            tokens (list[str]): Tokenized line representing the MOSFET.
            name (str): The name of the MOSFET (e.g., M1).
            type_char (str): The type character ('M').
        Returns:
            mosfet (Dict): A dictionary containing the parsed parameters for the MOSFET.
        """
        if len(tokens) < 6:
            self._throw_error(
                f"Malformed MOSFET '{name}'. Format: Name Nd Ng Ns Nb Model [Params]"
            )

        n_d: int = int(tokens[1])
        n_g: int = int(tokens[2])
        n_s: int = int(tokens[3])
        n_b: int = int(tokens[4])
        model: str = tokens[5].upper()
        geometric_params: dict[str, Any]
        geometric_params, _ = self._extract_params(tokens[6:])

        mosfet: Dict = {
            "type": type_char,
            "n_d": n_d,
            "n_g": n_g,
            "n_s": n_s,
            "n_b": n_b,
            "model": model,
            "geometric_params": geometric_params,
        }
        return mosfet

    def _parse_source(self, tokens: list[str], name: str, type_char: str) -> Dict:
        # al source parser is confusing
        if len(tokens) < 4:
            self._throw_error(
                f"Malformed source '{name}'. Format: Name N+ N- [DC/AC/TRAN value]"
            )
        n1: int = int(tokens[1])
        n2: int = int(tokens[2])

        source_data = self._parse_source_def(tokens[3:])
        comp_data = {
            "type": type_char,
            "n1": n1,
            "n2": n2,
            "value": source_data["dc"],
            "ac_mag": source_data["ac_mag"],
            "ac_phase": source_data["ac_phase"],
        }
        if source_data["tran"] is not None:
            comp_data["source"] = source_data["tran"]
        return comp_data

    def _parse_vccs(self, tokens: list[str], name: str, type_char: str) -> Dict:
        """
        Parses a Voltage-Controlled Current Source (VCCS). Format: Name N+ N- NC+ NC- Gain
        Args:
            tokens (list[str]): Tokenized line representing the VCCS.
            name (str): The name of the VCCS (e.g., G1).
            type_char (str): The type character ('G').
        Returns:
            vccs (Dict): A dictionary containing the parsed parameters for the VCCS.
        """
        if len(tokens) < 6:
            self._throw_error(
                f"Malformed VCCS '{name}'. Format: Name N+ N- NC+ NC- Gain"
            )

        n1: int = int(tokens[1])
        n2: int = int(tokens[2])
        n3: int = int(tokens[3])
        n4: int = int(tokens[4])
        gain: float = self._parse_value(tokens[5])

        vccs: Dict = {
            "type": type_char,
            "n1": n1,
            "n2": n2,
            "n3": n3,
            "n4": n4,
            "value": gain,
        }
        return vccs

    def _parse_opamp(self, tokens, name, type_char) -> Dict:
        """
        Parses an ideal OpAmp. Format: Name Nout Nref N+ N- [A or A=Value]
        Args:
            tokens (list[str]): Tokenized line representing the OpAmp.
            name (str): The name of the OpAmp (e.g., O1).
            type_char (str): The type character ('O').
        Returns:
            opamp (Dict): A dictionary containing the parsed parameters for the OpAmp.
        """
        if len(tokens) < 5:
            self._throw_error(
                f"Malformed opamp '{name}'. Format: Oname nout nref nplus nminus [A or A=...]"
            )

        nout: int = int(tokens[1])
        nref: int = int(tokens[2])
        nplus: int = int(tokens[3])
        nminus: int = int(tokens[4])

        A: float = 1e5  # default
        if len(tokens) >= 6:
            t = tokens[5]
            if "=" in t:
                k, v = t.split("=", 1)
                if k.strip().upper() == "A":
                    A = float(v)
            else:
                A = float(t)
        opamp: Dict = {
            "type": "O",
            "n1": nout,
            "n2": nref,
            "n3": nplus,
            "n4": nminus,
            "value": A,
        }

        return opamp

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
        args = [self._parse_value(x) for x in match.group(2).replace(",", " ").split()]

        if name == "PULSE":
            keys = ["V1", "V2", "TD", "TR", "TF", "PW", "PER"]
            return {"type": "PULSE", **dict(zip(keys, args + [0] * (7 - len(args))))}
        elif name in ["SIN", "COS"]:
            keys = ["VOFF", "VAMP", "FREQ", "TD", "PHASE"]
            result = {"type": name, **dict(zip(keys, args + [0] * (5 - len(args))))}
            result["PHASE"] = np.radians(
                result["PHASE"]
            )  # Convert degrees to rads for Numpy
            return result
        return None

    @staticmethod
    def _parse_value(value_str: str) -> float:
        """
        Parses a string with optional SI suffixes into a float.
        Args:
            value_str (str): A string representing a number, optionally combined with a suffix denoting the order of magnitude.
        Returns:
            num (float): A float representing the parsed value, with the suffix applied (if present).
        """
        if not value_str:
            return 0.0
        val_str: str = value_str.upper()

        multipliers: Dict[str, float] = {
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

        match: Match[str] = re.match(
            r"^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)(\D*)$", val_str
        )
        if not match:
            return 0.0

        num: float = float(match.group(1))
        suffix: str = match.group(2)
        for s, mult in multipliers.items():
            if suffix.startswith(s):
                return num * mult
        return num

    @staticmethod
    def _extract_params(tokens: list[str]) -> dict[str, float]:
        """
        Extracts parameters associated with model. Discards any tokens left over.
        Args:
            tokens (list[str]): list of tokens representing the model parameters.
        Returns:
            params (dict(str, float)): dictionary representing the parameters."""
        params: dict[str, float]
        leftovers: Any
        params, leftovers = {}, []
        for token in tokens:
            if "=" in token:
                key: str
                val_str: float
                key, val_str = token.split("=", 1)
                params[key.upper()] = NetlistParser._parse_value(val_str)
            else:
                leftovers.append(token)
        return params

    # =========================================================================
    # ERROR HANDLING & FILE PRE-PROCESSING
    # =========================================================================
    def _throw_error(self, msg: str):
        """Centralized error formatting with exact line numbers."""
        raise ValueError(
            f"Parse Error on Line {self.current_line}:\n{msg}\n-> {self.raw_line_text}"
        )

    def _preprocess_lines(self, lines) -> list[tuple[int, str]]:
        """
        Handles '+' continuations and returns (line_number, full_string) tuples.
        This ensures errors map perfectly to the user's actual text file.
        Args:
            lines
        """
        full_lines: list[tuple[int, str]] = []
        current_logical_line: str = ""
        start_line_num: int = 0

        for i, line in enumerate(lines):
            line_num: int = i + 1
            stripped: str = line.strip()

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

    # def _attach_models(self):
    #     """Binds model parameters to component instances at the end."""
    #     for name, comp in self.components:
    #         if "model" in comp:
    #             m_name = comp["model"]
    #             if m_name in self.models:
    #                 comp["model_params"] = self.models[m_name]["params"]
    #                 comp["model_type"] = self.models[m_name]["type"]
    #             else:
    #                 print(f"Warning: Model '{m_name}' not found for '{name}'")
