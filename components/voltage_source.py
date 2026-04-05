"""Voltage Source Component Module."""

from .base import Component
from core.waveforms import Waveform
import numpy as np


class VoltageSource(Component):
    """An independent voltage source (Type 'V'). Requires an MNA branch equation.
    
    The branch equation enforces V(n1) - V(n2) = V_source, introducing
    an additional unknown (the branch current) into the MNA system.
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
        self.branch_idx = node_map.get(self.name)

    def stamp_mna_connection(self, Y):
        """Stamps +1/-1 branch topology into the MNA matrix."""
        self._stamp_branch_equation(Y)

    def stamp_dc(self, Y, sources):
        """Stamps the DC voltage value into the RHS.
        
        Standard SPICE convention: the DC operating point always uses the 
        explicit DC value, ignoring any transient waveform. The waveform 
        only takes effect during transient analysis (.TRAN).
        """
        sources[self.branch_idx] = self.value

    def stamp_transient(self, Y, sources, t, dt, v_prev, method='BE'):
        """Stamps the time-varying voltage into the branch equation RHS."""
        if self.waveform:
            current_volts = self.waveform.get_value(t)
        else:
            current_volts = self.value
            
        sources[self.branch_idx] = current_volts

    def stamp_ac(self, Y, sources, w):
        """Stamps the AC phasor into the branch equation RHS.
        
        In AC analysis, DC sources are killed (RHS = 0), and only the AC
        phasor is applied.
        """
        if self.ac_mag != 0.0:
            sources[self.branch_idx] += self.phasor

    def get_sensitivities(self, VI, PsiPhi, w=0.0, dt=None, V_prev=None, method='BE'):
        """Sensitivity of the output w.r.t. the DC source voltage value.
        
        The branch equation is: V(n1) - V(n2) = V_source
        Derivative w.r.t. V_source appears as +1 in the RHS.
        Adjoint formula: sens = Psi_branch * 1.0 = Psi_branch
        """
        if self.branch_idx is None:
            return {}
        
        psi_branch = PsiPhi[self.branch_idx]
        return {self.name: psi_branch}
