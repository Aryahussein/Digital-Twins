from .base import Component
from core.waveforms import Waveform
import numpy as np

class CurrentSource(Component):
    """An independent current source (Type 'I')."""

    IS_INDEPENDENT_SOURCE = True
    
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
    # 1. STATIC STAMPING (Independent Sources)
    # ==========================================
    def stamp_sources(self, J, domain, t=0.0):
        """Phase 2 (Sources): Stamps steady-state, time-varying, or AC phasors.
        
        Bypasses the evaluate_physics pipeline because independent sources 
        do not depend on the dynamic voltage state (V_k), saving NR iterations.
        """
        if domain == "frequency":
            # Small-signal AC suppresses independent DC/Transient sources
            if self.ac_mag != 0.0:
                self._apply_rhs_stamp(J, self.phasor)
            return 
            
        # Determine the current value based on the domain and waveform
        if self.waveform is not None:
            current_val = self.waveform.get_value(t if domain == "time" else 0.0)
        else:
            current_val = self.value
            
        self._apply_rhs_stamp(J, current_val)

    def _apply_rhs_stamp(self, sources, current_val):
        """Internal helper to apply the nodal current flow.
        
        Perfectly matches the unified MNA convention: 
        Subtract from positive, add to negative.
        """
        if self.idx_1 is not None:
            sources[self.idx_1] -= current_val
        if self.idx_2 is not None:
            sources[self.idx_2] += current_val

    # ==========================================
    # 2. SENSITIVITY ENGINES (Adjoint & Woodbury)
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

    def stamp_PQ(self, P, Q, start_col_idx):
        """Independent sources do not populate the Admittance matrix (Y).
        
        This override MUST exist to prevent the Base Class from accidentally
        injecting 2-terminal admittance topology into the Woodbury engine!
        """
        pass

