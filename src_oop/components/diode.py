from .base import Component
import core.models as models
import core.constants as c

class Diode(Component):
    """Nonlinear Diode (Type 'D') using generic MNA stamping."""

    IS_NONLINEAR = True

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
                return getattr(self, attr_name)
                
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
    def evaluate_physics(self, domain="static", t=0.0, dt=0.0, w=0.0, v_prev=None, v_k=None, method="TR", param_name=None, dp=0.0, **kwargs):
        """Pure mathematical evaluation. Packs all data into a single generic dictionary."""
        res = {}
        
        # Diode only evaluates if we have a valid non-linear voltage guess
        if v_k is None:
            return res
            
        va = v_k[self.idx_a] if self.idx_a is not None else 0.0
        vk = v_k[self.idx_k] if self.idx_k is not None else 0.0
        vd = va - vk

        is_val = self.IS
        
        # Apply physical shifts if requested by the Large Change Engine
        if param_name and dp != 0.0:
            if param_name.endswith("_IS"): 
                is_val += dp

        # 1. Core Physics Evaluation
        phys_res = models.evaluate_diode(vd, is_val, self.VT)
        id_val, gd = phys_res["I_D"], phys_res["gd"]
        
        # 2. Pack data for the generic Stampers
        res["gd"] = gd
        res["ieq"] = id_val - gd * vd # Equivalent NR Current
        
        # 3. Pack data for the Sensitivity/Adjoint engines
        res["dId_dIs"] = phys_res["dId_dIs"]
        
        return res

    # ==========================================
    # 2. THE STAMPERS (Generic Interface)
    # ==========================================
    def stamp_matrix(self, Y, res, *args):
        """Stamps the dynamic conductance (gd) into the Jacobian."""
        if "gd" not in res: return
        
        gd = res["gd"]
        a, k = self.idx_a, self.idx_k

        if a is not None:
            Y[a, a] += gd
            if k is not None: Y[a, k] -= gd

        if k is not None:
            Y[k, k] += gd
            if a is not None: Y[k, a] -= gd

    def stamp_rhs(self, J, res, *args):
        """Stamps the equivalent NR current into the Residual vector."""
        if "ieq" not in res: return
        
        ieq = res["ieq"]
        a, k = self.idx_a, self.idx_k

        if a is not None: J[a] -= ieq
        if k is not None: J[k] += ieq

    # ==========================================
    # SENSITIVITY ENGINES
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
    
    def stamp_PQ(self, P, Q, col_idx):
        """Stamps the Injection (P) and Extraction (Q) topology vectors in-place."""
        a, k = self.idx_a, self.idx_k
        
        if a is not None: 
            P[a, col_idx] = 1.0
            Q[a, col_idx] = 1.0
        if k is not None: 
            P[k, col_idx] = -1.0
            Q[k, col_idx] = -1.0

    def get_delta_y(self, param_name, dp, V_nom=None, V_k=None, **kwargs):
        """Calculates Woodbury Admittance shifts exactly per the slides."""
        # 1. Evaluate baseline dynamic conductance
        res_nom = self.evaluate_physics(v_k=V_nom)
        gd_nom = res_nom.get("gd", 0.0)
        
        # 2. Evaluate shifted dynamic conductance at current NR guess
        res_new = self.evaluate_physics(v_k=V_k, param_name=param_name, dp=dp)
        gd_new = res_new.get("gd", 0.0)
        
        return gd_new - gd_nom
