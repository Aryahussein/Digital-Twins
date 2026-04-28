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

    # ==========================================
    # SOLVER ENGINES (Stamping)
    # ==========================================
    def stamp_mna_connection(self, Y):
        """Stamps the +1/-1 topology for the branch current."""
        self._stamp_branch_equation(Y)

    def stamp_dc(self, Y, sources):
        if self.waveform:
            sources[self.branch_idx] = self.waveform.get_value(0.0)
        else:
            sources[self.branch_idx] = self.value

    def stamp_transient(self, Y, sources, t, dt, v_prev, method='TR'):
        if self.waveform:
            current_volts = self.waveform.get_value(t)
        else:
            current_volts = self.value
            
        sources[self.branch_idx] = current_volts

    def stamp_ac(self, Y, sources, w):
        if self.ac_mag != 0.0:
            sources[self.branch_idx] += self.phasor

    def get_sensitivities(self, VI, PsiPhi, **kwargs):
        """Returns the sensitivity of the output w.r.t the source value.
        
        For an independent voltage source, the Adjoint sensitivity is strictly
        driven by the Adjoint current flowing through its branch equation.
        """
        if self.branch_idx is None: return {}
        
        # Sensitivity is exactly the adjoint branch variable
        psi_branch = PsiPhi[self.branch_idx]
        
        return {self.name: psi_branch}

    def stamp_PQ(self, P, Q, col_idx):
        """Independent sources do not change the Admittance matrix (Y).
        
        While the Voltage Source has a topological presence in Y (+1/-1), its 
        actual parameter value only exists in the RHS vector (J). Therefore, 
        there is no P/Q injection topology required for Woodbury.
        """
        pass

    def get_delta_y(self, param_name, dp, **kwargs):
        """Transforms a physical parameter change into a scalar Admittance change.
        
        Because independent sources do not alter the values inside the 
        Admittance matrix, their Delta Y shift is always mathematically zero.
        """
        return 0.0
