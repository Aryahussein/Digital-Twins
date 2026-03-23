from .base import Component
import core.models as models

class Mosfet(Component):
    """Simple Level 1 NMOS model with Hard-Coded Stamping."""

    def bind_nodes(self, node_map):
        self.idx_d = node_map.get(self.data.get("n_d", 0))
        self.idx_g = node_map.get(self.data.get("n_g", 0))
        self.idx_s = node_map.get(self.data.get("n_s", 0))
        
        params = self.data.get("model_params", {})
        inst = self.data.get("inst_params", {})
        
        self.VTO = params.get("VTO", 0.7)
        self.W = inst.get("W", 1e-6)
        self.L = inst.get("L", 1e-6)
        
        # Process parameters
        mu = params.get("MU", 600e-4) # default mobility
        cox = params.get("C_OX", 3.45e-3) # default oxide capacitance
        self.KP = params.get("KP", mu * cox)
        
        # Bn = (W/L) * KP
        self.Bn = (self.W / self.L) * self.KP

    def stamp_nonlinear(self, Y, sources, p_V_guess, V_guess):
        # 1. Get current iteration voltages
        vd = V_guess[self.idx_d] if self.idx_d is not None else 0.0
        vg = V_guess[self.idx_g] if self.idx_g is not None else 0.0
        vs = V_guess[self.idx_s] if self.idx_s is not None else 0.0
        
        # 2. Evaluate Physics
        res = models.evaluate_nmos(vg - vs, vd - vs, self.VTO, self.Bn)
        Id, gm, gds = res["I_D"], res["gm"], res["gds"]

        # 3. Hard-Coded Stamping (The "Slot" Method)
        # We define Ieq to handle the Newton-Raphson linearization offset
        ieq = Id - gm * (vg - vs) - gds * (vd - vs)

        # Drain terminal slots
        if self.idx_d is not None:
            if self.idx_g is not None: Y[self.idx_d, self.idx_g] += gm
            if self.idx_s is not None: Y[self.idx_d, self.idx_s] -= (gm + gds)
            Y[self.idx_d, self.idx_d] += gds
            sources[self.idx_d] -= ieq

        # Source terminal slots
        if self.idx_s is not None:
            if self.idx_g is not None: Y[self.idx_s, self.idx_g] -= gm
            if self.idx_d is not None: Y[self.idx_s, self.idx_d] -= gds
            Y[self.idx_s, self.idx_s] += (gm + gds)
            sources[self.idx_s] += ieq

    def get_sensitivities(self, VI, PsiPhi, w=0.0, dt=None, V_prev=None):
        """Calculates sensitivities for W, L, and VTO using the Adjoint method."""
        vd = VI[self.idx_d] if self.idx_d is not None else 0.0
        vg = VI[self.idx_g] if self.idx_g is not None else 0.0
        vs = VI[self.idx_s] if self.idx_s is not None else 0.0
        
        pd = PsiPhi[self.idx_d] if self.idx_d is not None else 0.0
        ps = PsiPhi[self.idx_s] if self.idx_s is not None else 0.0
        
        # Adjoint term: potential difference across the 'current' of the device
        adj_factor = (pd - ps)
        
        res = models.evaluate_nmos(vg - vs, vd - vs, self.VTO, self.Bn)
        dId_dBn = res["dId_dBn"]
        dId_dVTO = res["dId_dVTO"]

        # Apply Chain Rule
        sens_W = adj_factor * dId_dBn * (self.KP / self.L)
        sens_L = adj_factor * dId_dBn * (-self.W * self.KP / (self.L**2))
        sens_VTO = adj_factor * dId_dVTO
        
        return {
            f"{self.name}_W": sens_W,
            f"{self.name}_L": sens_L,
            f"{self.name}_VTO": sens_VTO
        }
