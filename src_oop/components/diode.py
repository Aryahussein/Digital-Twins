from .base import Component
import core.models as models
import core.constants as c

class Diode(Component):
    """Nonlinear Diode (Type 'D') using generic MNA stamping."""

    IS_NONLINEAR = True
    
    # Tells the Base Class time-travel function and matrix stamper which key to extract!
    SHIFT_KEY = "gd"

    def __init__(self, name, data_dict):
        super().__init__(name, data_dict)
        # Single Source of Truth for differentiable parameters
        self._diff_attrs = ["IS"]

        # 2. Route user parameters safely through the setter
        params = self.data.get("model_params", {})
        for key, val in params.items():
            self.set_nominal_value(f"{self.name}_{key}", val)

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
                return getattr(self, attr_name, 0.0)
                
        return super().get_nominal_value(param_name)

    def set_nominal_value(self, param_name, new_val):
        """Safely routes multi-parameter updates and updates dependent physics."""
        prefix = f"{self.name}_"
        if param_name.startswith(prefix):
            attr_name = param_name[len(prefix):] 
            if hasattr(self, attr_name):
                # 1. Update the physical value
                setattr(self, attr_name, new_val)
                return
                
        super().set_nominal_value(param_name, new_val)

    def bind_nodes(self, node_map):
        """Maps Anode (n1) and Cathode (n2) to matrix indices."""
        self.idx_a = node_map.get(self.data.get("n1", 0))
        self.idx_k = node_map.get(self.data.get("n2", 0))
        
        params = self.data.get("model_params", {})
        self.IS = params.get("IS", 1e-14)
        self.VT = params.get("VT", c.Vt)

    # ==========================================
    # 1. THE PHYSICS EVALUATOR
    # ==========================================
    def evaluate_physics(self, v_k=None, overrides=None, **kwargs):
        """Pure mathematical evaluation. Safely applies absolute parameter overrides."""
        res = {}
        
        # Diode only evaluates if we have a valid non-linear voltage guess
        if v_k is None:
            return res
            
        overrides = overrides or {}
            
        va = v_k[self.idx_a] if self.idx_a is not None else 0.0
        vk = v_k[self.idx_k] if self.idx_k is not None else 0.0
        vd = va - vk

        # PURE PHYSICS: Use the override if it exists, otherwise use nominal
        eff_is = overrides.get(f"{self.name}_IS", self.IS)

        # 1. Core Physics Evaluation
        phys_res = models.evaluate_diode(vd, eff_is, self.VT)
        id_val, gd = phys_res["I_D"], phys_res["gd"]
        
        # 2. Pack data for the generic Stampers
        res["gd"] = gd
        # Standard Newton-Raphson Equivalent Current (I_eq = I_D - G_D * V_D)
        # Because it is positive here, the Base Class will correctly subtract it 
        # from the positive node's RHS vector (J[pos] -= I_eq).
        res["I_eq"] = id_val - gd * vd 
        
        # 3. Pack data for the Sensitivity/Adjoint engines
        res["dId_dIs"] = phys_res["dId_dIs"]
        
        return res

    # ==========================================
    # 2. SENSITIVITY ENGINES
    # ==========================================
    def get_sensitivities(self, VI, PsiPhi, **kwargs):
        """Calculates sensitivity w.r.t Saturation Current (IS)."""
        res = self.evaluate_physics(v_k=VI)
        if "dId_dIs" not in res: return {}
        
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
        
