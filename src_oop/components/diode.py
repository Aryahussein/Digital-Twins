from .base import Component
import core.models as models

class Diode(Component):
    """Nonlinear Diode (Type 'D') using hard-coded MNA stamping."""

    IS_NONLINEAR = True

    def bind_nodes(self, node_map):
        """Maps Anode (n1) and Cathode (n2) to matrix indices."""
        self.idx_a = node_map.get(self.data.get("n1", 0))
        self.idx_k = node_map.get(self.data.get("n2", 0))
        
        params = self.data.get("model_params", {})
        self.IS = params.get("IS", 1e-14)
        self.VT = params.get("VT", 0.02585)

    def stamp_nonlinear(self, Y, sources, p_V_guess, V_guess):
        """Stamps linearized gd and Ieq into the MNA system."""
        # 1. Extract voltages
        va = V_guess[self.idx_a] if self.idx_a is not None else 0.0
        vk = V_guess[self.idx_k] if self.idx_k is not None else 0.0
        vd = va - vk

        # 2. Evaluate Physics from models.py
        res = models.evaluate_diode(vd, self.IS, self.VT)
        id_val, gd = res["I_D"], res["gd"]

        # 3. Newton-Raphson linearized current source
        # Ieq = Id - gd * Vd
        ieq = id_val - gd * vd

        # 4. Hard-Coded "Slot" Stamping
        # --- Anode Row ---
        if self.idx_a is not None:
            Y[self.idx_a, self.idx_a] += gd
            sources[self.idx_a] -= ieq
            if self.idx_k is not None:
                Y[self.idx_a, self.idx_k] -= gd

        # --- Cathode Row ---
        if self.idx_k is not None:
            Y[self.idx_k, self.idx_k] += gd
            sources[self.idx_k] += ieq
            if self.idx_a is not None:
                Y[self.idx_k, self.idx_a] -= gd

    def get_sensitivities(self, VI, PsiPhi, w=0.0, dt=None, V_prev=None):
        """Calculates sensitivity w.r.t Saturation Current (IS)."""
        va = VI[self.idx_a] if self.idx_a is not None else 0.0
        vk = VI[self.idx_k] if self.idx_k is not None else 0.0
        
        pa = PsiPhi[self.idx_a] if self.idx_a is not None else 0.0
        pk = PsiPhi[self.idx_k] if self.idx_k is not None else 0.0
        
        res = models.evaluate_diode(va - vk, self.IS, self.VT)
        
        # d(Output)/d(IS) = (Psi_anode - Psi_cathode) * d(Id)/d(IS)
        return {f"{self.name}_IS": (pa - pk) * res["dId_dIs"]}
    