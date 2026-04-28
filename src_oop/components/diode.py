from .base import Component
import core.models as models
import core.constants as c

class Diode(Component):
    """Nonlinear Diode (Type 'D') using hard-coded MNA stamping."""

    IS_NONLINEAR = True

    def __init__(self, name, data_dict):
        super().__init__(name, data_dict)
        # Single Source of Truth for differentiable parameters
        self._diff_attrs = ["IS"]

    @property
    def differentiable_params(self):
        """Dynamically advertises parameters based on the internal registry."""
        return [f"{self.name}_{attr}" for attr in self._diff_attrs]

    def get_nominal_value(self, param_name):
        """Dynamically routes parameter requests using the registry."""
        prefix = f"{self.name}_"
        if param_name.startswith(prefix):
            attr_name = param_name[len(prefix):]
            if attr_name in self._diff_attrs:
                return getattr(self, attr_name)
                
        return super().get_nominal_value(param_name)

    def bind_nodes(self, node_map):
        """Maps Anode (n1) and Cathode (n2) to matrix indices."""
        self.idx_a = node_map.get(self.data.get("n1", 0))
        self.idx_k = node_map.get(self.data.get("n2", 0))
        
        params = self.data.get("model_params", {})
        self.IS = params.get("IS", 1e-14)
        self.VT = params.get("VT", c.Vt)

    # ==========================================
    # THE PHYSICS HELPER
    # ==========================================
    def _evaluate_physics(self, V_array, param_name=None, dp=0.0):
        """Centralized helper for voltage extraction and device physics."""
        va = V_array[self.idx_a] if self.idx_a is not None else 0.0
        vk = V_array[self.idx_k] if self.idx_k is not None else 0.0
        vd = va - vk

        is_val = self.IS
        
        # Apply physical shifts if requested by the Large Change Engine
        if param_name and dp != 0.0:
            if param_name.endswith("_IS"): 
                is_val += dp

        res = models.evaluate_diode(vd, is_val, self.VT)
        return res, vd

    # ==========================================
    # SOLVER ENGINES
    # ==========================================
    def stamp_nonlinear(self, Y, sources, p_V_guess, V_guess):
        """Stamps linearized gd and Ieq into the MNA system."""
        # Ask the helper for the exact physics evaluation
        res, vd = self._evaluate_physics(V_guess)
        id_val, gd = res["I_D"], res["gd"]

        # Newton-Raphson linearized current source (Ieq = Id - gd * Vd)
        ieq = id_val - gd * vd

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

    def get_sensitivities(self, VI, PsiPhi, **kwargs):
        """Calculates sensitivity w.r.t Saturation Current (IS)."""
        res, _ = self._evaluate_physics(VI)
        
        pa = PsiPhi[self.idx_a] if self.idx_a is not None else 0.0
        pk = PsiPhi[self.idx_k] if self.idx_k is not None else 0.0
        
        adj_factor = (pa - pk)
        
        sens_map = {
            "IS": adj_factor * res["dId_dIs"]
        }

        return {
            f"{self.name}_{attr}": sens_map[attr]
            for attr in self._diff_attrs
            if attr in sens_map
        }
    
    def stamp_PQ(self, P, Q, col_idx):
        """Stamps the Injection (P) and Extraction (Q) topology vectors in-place."""
        # BUGFIX: Changed idx_1/idx_2 to idx_a/idx_k
        a, k = self.idx_a, self.idx_k
        
        if a is not None: 
            P[a, col_idx] = 1.0
            Q[a, col_idx] = 1.0
        if k is not None: 
            P[k, col_idx] = -1.0
            Q[k, col_idx] = -1.0

    def get_delta_y(self, param_name, dp, **kwargs):
        """Calculates Woodbury Admittance shifts (Delta gd)."""
        VI = kwargs.get("VI")
        if VI is None: return 0.0 # Safety fallback
        
        # Nominal dynamic conductance
        res_nom, _ = self._evaluate_physics(VI)
        gd_nom = res_nom["gd"]
        
        # Shifted dynamic conductance
        res_new, _ = self._evaluate_physics(VI, param_name=param_name, dp=dp)
        gd_new = res_new["gd"]
        
        return gd_new - gd_nom
