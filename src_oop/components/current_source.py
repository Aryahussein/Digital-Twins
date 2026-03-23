from .base import Component
from core.waveforms import Waveform
import numpy as np

class CurrentSource(Component):
    """An independent current source (Type 'I')."""
    def __init__(self, name, data_dict):
        super().__init__(name, data_dict)
        
        # 1. Setup Transient Waveform (PULSE, SIN, etc.)
        self.waveform = None
        if "source" in data_dict:
            self.waveform = Waveform(data_dict["source"])
            
        # 2. Setup AC Phasor for Frequency Sweeps
        self.ac_mag = data_dict.get("ac_mag", 0.0)
        self.ac_phase = data_dict.get("ac_phase", 0.0)
        self.phasor = self.ac_mag * np.exp(1j * np.radians(self.ac_phase))

    def bind_nodes(self, node_map):
        """Maps terminals to matrix indices."""
        self.idx_1 = node_map.get(self.data.get("n1", 0))
        self.idx_2 = node_map.get(self.data.get("n2", 0))

    def stamp_dc(self, Y, sources):
        """Stamps the steady-state DC value (at t=0)."""
        val = self.waveform.get_value(0.0) if self.waveform else self.value
        self._apply_rhs_stamp(sources, val)

    def stamp_transient(self, Y, sources, t, dt, v_prev):
        """Stamps the time-varying value based on the waveform function."""
        val = self.waveform.get_value(t) if self.waveform else self.value
        self._apply_rhs_stamp(sources, val)

    def stamp_ac(self, Y, sources, w):
        """Stamps the complex AC phasor."""
        if self.ac_mag != 0.0:
            self._apply_rhs_stamp(sources, self.phasor)

    def _apply_rhs_stamp(self, sources, current_val):
        """Internal helper to apply the nodal current flow."""
        if self.idx_1 is not None:
            sources[self.idx_1] -= current_val
        if self.idx_2 is not None:
            sources[self.idx_2] += current_val

    def get_sensitivities(self, VI, PsiPhi, w=0.0, dt=None, V_prev=None):
        """Returns the sensitivity of the output w.r.t the source value."""
        p1 = PsiPhi[self.idx_1] if self.idx_1 is not None else 0.0
        p2 = PsiPhi[self.idx_2] if self.idx_2 is not None else 0.0
        # For a current source, sensitivity is -(psi_i - psi_j)
        return {self.name: -(p1 - p2)}
