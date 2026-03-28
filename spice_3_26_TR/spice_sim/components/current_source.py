"""Current Source Component Module."""

from .base import Component
from core.waveforms import Waveform
import numpy as np


class CurrentSource(Component):
    """An independent current source (Type 'I').
    
    Current flows from n1 (positive) to n2 (negative) through the external
    circuit. In MNA, this stamps directly into the RHS source vector.
    """

    def __init__(self, name, data_dict):
        super().__init__(name, data_dict)
        
        self.waveform = None
        if "source" in data_dict:
            self.waveform = Waveform(data_dict["source"])
            
        self.ac_mag = data_dict.get("ac_mag", 0.0)
        self.ac_phase = data_dict.get("ac_phase", 0.0)
        self.phasor = self.ac_mag * np.exp(1j * np.radians(self.ac_phase))

    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0))
        self.idx_2 = node_map.get(self.data.get("n2", 0))

    def stamp_dc(self, Y, sources):
        """Stamps the DC current value into the RHS.
        
        Standard SPICE convention: DC OP uses the explicit DC value only.
        Transient waveforms are ignored during operating point calculation.
        """
        self._apply_rhs_stamp(sources, self.value)

    def stamp_transient(self, Y, sources, t, dt, v_prev, method='BE'):
        """Stamps the time-varying current based on the waveform function."""
        val = self.waveform.get_value(t) if self.waveform else self.value
        self._apply_rhs_stamp(sources, val)

    def stamp_ac(self, Y, sources, w):
        """Stamps the complex AC phasor into the RHS."""
        if self.ac_mag != 0.0:
            self._apply_rhs_stamp(sources, self.phasor)

    def _apply_rhs_stamp(self, sources, current_val):
        """Internal helper to apply the nodal current flow convention."""
        if self.idx_1 is not None:
            sources[self.idx_1] -= current_val
        if self.idx_2 is not None:
            sources[self.idx_2] += current_val

    def get_sensitivities(self, VI, PsiPhi, w=0.0, dt=None, V_prev=None, method='BE'):
        """Sensitivity of output w.r.t. the source current value."""
        p1 = PsiPhi[self.idx_1] if self.idx_1 is not None else 0.0
        p2 = PsiPhi[self.idx_2] if self.idx_2 is not None else 0.0
        return {self.name: -(p1 - p2)}
