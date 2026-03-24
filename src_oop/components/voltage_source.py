from .base import Component
from core.waveforms import Waveform
import numpy as np

class VoltageSource(Component):
    """An independent voltage source requiring an MNA branch equation."""
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
        self.branch_idx = node_map.get(self.name)

    def stamp_mna_connection(self, Y):
        self._stamp_branch_equation(Y)

    def stamp_dc(self, Y, sources):
        if self.waveform:
            sources[self.branch_idx] = self.waveform.get_value(0.0)
        else:
            sources[self.branch_idx] = self.value

    def stamp_transient(self, Y, sources, t, dt, v_prev, method = 'TR'):
        if self.waveform:
            current_volts = self.waveform.get_value(t)
        else:
            current_volts = self.value
            
        sources[self.branch_idx] = current_volts

    def stamp_ac(self, Y, sources, w):
        if self.ac_mag != 0.0:
            sources[self.branch_idx] += self.phasor
