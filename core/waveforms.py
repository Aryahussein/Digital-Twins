"""
Time-Domain Waveform Module.

This module defines the mathematical evaluation of time-varying independent
voltage and current sources (e.g., PULSE, SIN, PWL) used during Transient Analysis.
"""

import numpy as np

class Waveform:
    """Evaluates time-domain source functions for transient simulations.

    This class parses standard SPICE transient source parameters and calculates
    the instantaneous voltage or current at any given simulation time `t`.

    Attributes:
        type (str): The waveform type identifier (e.g., 'PULSE', 'SIN', 'COS', 'PWL').
        params (dict): The dictionary of parameters defining the waveform shape.
    """

    def __init__(self, source_dict):
        """Initializes the waveform generator.

        Args:
            source_dict (dict): Parsed dictionary containing the 'type' key and 
                all required parameters for that specific waveform type.
        """
        self.type = str(source_dict.get("type", "")).upper()
        self.params = source_dict

    def get_value(self, t):
        """Calculates the instantaneous value of the waveform at time t.

        Args:
            t (float): The current simulation time in seconds.

        Returns:
            float: The instantaneous voltage (V) or current (A).
        """
        if self.type == "PULSE":
            V1 = self.params.get("V1", 0.0)
            V2 = self.params.get("V2", 1.0)
            TD = self.params.get("TD", 0.0)
            TR = self.params.get("TR", 0.0)
            TF = self.params.get("TF", 0.0)
            PW = self.params.get("PW", 1.0)
            PER = self.params.get("PER", 2.0)

            if t < TD: 
                return V1
            
            t_rel = (t - TD) % PER if PER > 0 else (t - TD)
            rise_done = TR if TR > 0 else 0.0

            if TR > 0 and t_rel < TR:
                return V1 + (V2 - V1) * (t_rel / TR)
            
            if t_rel < rise_done + PW:
                return V2
            
            if TF > 0 and t_rel < rise_done + PW + TF:
                return V2 - (V2 - V1) * ((t_rel - rise_done - PW) / TF)
            
            return V1

        elif self.type == "SIN" or self.type == "COS":
            voff = self.params.get("VOFF", 0.0)
            vamp = self.params.get("VAMP", 1.0)
            freq = self.params.get("FREQ", 1.0)
            phase_deg = self.params.get("PHASE", 0.0)
            phase_rad = np.radians(phase_deg)
            
            if self.type == "SIN":
                return voff + vamp * np.sin(2 * np.pi * freq * t + phase_rad)
            else:
                return voff + vamp * np.cos(2 * np.pi * freq * t + phase_rad)

        elif self.type == "PWL":
            pairs = self.params.get("TIME_VOLTAGE_PAIRS", [(0.0, 0.0)])
            times = [p[0] for p in pairs]
            volts = [p[1] for p in pairs]
            return np.interp(t, times, volts)

        return self.params.get("value", 0.0)
