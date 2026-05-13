from .base import Component
from core.waveforms import Waveform
import numpy as np

class VoltageSource(Component):
    """An independent voltage source (Type 'V') requiring an MNA branch equation."""

    IS_INDEPENDENT_SOURCE = True
    REQUIRES_BRANCH_EQ = True
    
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
        """Maps terminals to matrix indices and assigns a branch index."""
        self.idx_1 = node_map.get(self.data.get("n1", 0))
        self.idx_2 = node_map.get(self.data.get("n2", 0))
        self.branch_idx = node_map.get(self.name)

    # ==========================================
    # 1. STATIC STAMPING (Skeleton Matrix)
    # ==========================================
    def stamp_base_matrix(self, Y):
        """Phase 1: The +1/-1 topology belongs in the Skeleton. Never loop this."""
        self._stamp_branch_equation(Y)

    # ==========================================
    # 2. SOURCE STAMPING (Independent Sources)
    # ==========================================
    def stamp_sources(self, J, domain, t=0.0):
        """Phase 2 (Sources): Stamps steady-state, time-varying, or AC phasors.
        
        Bypasses the evaluate_physics pipeline because independent sources 
        do not depend on the dynamic voltage state (V_k).
        """
        if self.branch_idx is None: return

        if domain == "frequency":
            # Small-signal AC suppresses independent DC/Transient sources
            if self.ac_mag != 0.0:
                J[self.branch_idx] += self.phasor
            return 
            
        # Determine the voltage value based on the domain and waveform
        if domain == "time" and self.waveform is not None:
            v_val = self.waveform.get_value(t)
        else:
            v_val = self.value
            
        J[self.branch_idx] += v_val

    # ==========================================
    # 3. SENSITIVITY & WOODBURY ENGINES
    # ==========================================
    def get_sensitivities(self, VI, PsiPhi, **kwargs):
        """Returns the sensitivity of the output w.r.t the source value.
        
        For an independent voltage source, the Adjoint sensitivity is strictly
        driven by the Adjoint current flowing through its branch equation.
        """
        if self.branch_idx is None: return {}
        
        # Sensitivity is exactly the adjoint branch variable
        psi_branch = PsiPhi[self.branch_idx]
        
        return {self.name: psi_branch}

    def stamp_PQ(self, P, Q, start_col_idx):
        """Independent sources do not change the Admittance matrix (Y).
        
        This override MUST exist to prevent the Base Class from accidentally
        injecting 2-terminal admittance topology into the Woodbury engine!
        """
        pass
