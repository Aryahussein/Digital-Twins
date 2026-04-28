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

    # ==========================================
    # SOLVER ENGINES (Stamping)
    # ==========================================
    def stamp_dc(self, Y, sources):
        """Stamps the steady-state DC value (at t=0)."""
        val = self.waveform.get_value(0.0) if self.waveform else self.value
        self._apply_rhs_stamp(sources, val)

    def stamp_transient(self, Y, sources, t, dt, v_prev, method='TR'):
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

    # ==========================================
    # SENSITIVITY ENGINES (Adjoint & Woodbury)
    # ==========================================
    def get_sensitivities(self, VI, PsiPhi, **kwargs):
        """Returns the sensitivity of the output w.r.t the source value.
        
        For an independent current source, the Adjoint sensitivity is strictly
        driven by the Adjoint potential difference across its terminals.
        """
        p1 = PsiPhi[self.idx_1] if self.idx_1 is not None else 0.0
        p2 = PsiPhi[self.idx_2] if self.idx_2 is not None else 0.0
        
        # Sensitivity = -(Psi_positive - Psi_negative)
        return {self.name: -(p1 - p2)}

    def stamp_PQ(self, P, Q, col_idx):
        """Independent sources do not populate the Admittance matrix (Y).
        
        Because they only exist in the RHS vector (J), they have no P or Q 
        topology to inject for Woodbury inversions.
        """
        pass

    def get_delta_y(self, param_name, dp, **kwargs):
        """Transforms a physical parameter change into a scalar Admittance change.
        
        Because independent sources do not populate the Admittance matrix, 
        their Delta Y shift is always mathematically zero.
        """
        return 0.0
