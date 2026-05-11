from .base import Component
import core.models as models
import numpy as np


class Mosfet(Component):
    """Abstract Base Class for MOSFETs using generic MNA stamping."""

    POLARITY: float
    IS_NONLINEAR = True

    def __init__(self, name, data_dict):
        super().__init__(name, data_dict)
        
        if not hasattr(self, "POLARITY"):
            raise NotImplementedError(
                f"Cannot instantiate '{self.__class__.__name__}'. "
                "Subclasses of MOS must define a 'POLARITY' constant."
            )

        self._diff_attrs = ["W", "L", "VTO", "KP", "MU", "COX"]

        params = self.data.get("model_params", {})
        inst = self.data.get("inst_params", {})

        for key, val in {**params, **inst}.items():
            if key in self._diff_attrs:
                setattr(self, key, val)

        if not hasattr(self, "KP"):
            if hasattr(self, "MU") and hasattr(self, "COX"):
                self.KP = self.MU * self.COX


        missing_params = []
        for req_param in ["W", "L", "VTO", "KP"]:
            if not hasattr(self, req_param):
                missing_params.append(req_param)

        if missing_params:
            raise ValueError(
                f"FATAL: Transistor '{self.name}' cannot be initialized. "
                f"Missing required parameters: {missing_params}. "
                "Verify your netlist defines W and L, and points to a valid .MODEL card "
                "containing VTO and either KP or (MU and COX)."
            )

        for key, val in {**params, **inst}.items():
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
        """Safely routes multi-parameter updates for Transistors."""
        prefix = f"{self.name}_"
        
        if param_name.startswith(prefix):
            attr_name = param_name[len(prefix):] 
            
            if attr_name in self._diff_attrs:
                setattr(self, attr_name, new_val)
                
                # NEW: Dynamically recalculate KP if MU or COX was swept!
                if attr_name in ["MU", "COX"] and hasattr(self, "MU") and hasattr(self, "COX"):
                    self.KP = self.MU * self.COX
                
                self.Bn = (self.W / self.L) * self.KP
                return
                
        super().set_nominal_value(param_name, new_val)

    def bind_nodes(self, node_map):
        """Translates string terminal names into matrix indices."""
        self.idx_d = node_map.get(self.data.get("n_d", 0))
        self.idx_g = node_map.get(self.data.get("n_g", 0))
        self.idx_s = node_map.get(self.data.get("n_s", 0))

    # ==========================================
    # 1. THE PHYSICS EVALUATOR
    # ==========================================
    def evaluate_physics(self, domain="static", t=0.0, dt=0.0, w=0.0, v_prev=None, v_k=None, method="TR", param_name=None, dp=0.0, **kwargs):
        """Pure mathematical evaluation. Packs all data into a single generic dictionary."""
        res = {}
        if v_k is None: return res
            
        # 1. Extract Voltages
        vd = v_k[self.idx_d] if self.idx_d is not None else 0.0
        vg = v_k[self.idx_g] if self.idx_g is not None else 0.0
        vs = v_k[self.idx_s] if self.idx_s is not None else 0.0

        vgs = self.POLARITY * (vg - vs)
        vds = self.POLARITY * (vd - vs)

        # 2. Dynamic Parameter Setup
        vto, w_val, l_val, kp_val = self.VTO, self.W, self.L, self.KP
        mu_val = getattr(self, "MU", 0.0)
        cox_val = getattr(self, "COX", 0.0)
        
        # Apply the Woodbury perturbation dynamically
        if param_name and dp != 0.0:
            prefix = f"{self.name}_"
            if param_name.startswith(prefix):
                attr = param_name[len(prefix):]
                if attr == "VTO": vto += dp
                elif attr == "W": w_val += dp
                elif attr == "L": l_val += dp
                elif attr == "KP": kp_val += dp
                elif attr == "MU": 
                    mu_val += dp
                    kp_val = mu_val * cox_val # Re-derive KP
                elif attr == "COX":
                    cox_val += dp
                    kp_val = mu_val * cox_val # Re-derive KP
            
        bn = (w_val / l_val) * kp_val

        # 3. Core Physics Evaluation
        phys_res = models.evaluate_nmos(vgs, vds, vto, bn)
        Id, gm, gds = phys_res["I_D"], phys_res["gm"], phys_res["gds"]
        
        # 4. Pack data for the generic Stampers
        res["gm"] = gm
        res["gds"] = gds
        res["ieq"] = (Id - gm * vgs - gds * vds) * self.POLARITY
        
        # 5. Pack data for Sensitivity/Adjoint engines
        res["dId_dBn"] = phys_res["dId_dBn"]
        res["dId_dVTO"] = phys_res["dId_dVTO"]
        
        return res

    # ==========================================
    # 2. THE STAMPERS (Generic Interface)
    # ==========================================
    def stamp_matrix(self, Y, res, *args):
        """Stamps Jacobians (gm, gds) into the matrix."""
        if "gm" not in res or "gds" not in res: return
        
        gm, gds = res["gm"], res["gds"]
        d, g, s = self.idx_d, self.idx_g, self.idx_s

        if d is not None:
            Y[d, d] += gds
            if g is not None: Y[d, g] += gm
            if s is not None: Y[d, s] -= (gm + gds)

        if s is not None:
            Y[s, s] += (gm + gds)
            if g is not None: Y[s, g] -= gm
            if d is not None: Y[s, d] -= gds

    def stamp_rhs(self, J, res, *args):
        """Stamps the Equivalent Current into the RHS vector."""
        if "ieq" not in res: return
        
        ieq = res["ieq"]
        d, s = self.idx_d, self.idx_s

        if d is not None: J[d] -= ieq
        if s is not None: J[s] += ieq

    # ==========================================
    # SENSITIVITY ENGINES (Adjoint & Woodbury)
    # ==========================================
    def get_sensitivities(self, VI, PsiPhi, **kwargs):
        """Calculates Adjoint derivatives dynamically using the chain rule."""
        res = self.evaluate_physics(v_k=VI)
        if "dId_dBn" not in res: return {}
        
        pd = PsiPhi[self.idx_d] if self.idx_d is not None else 0.0
        ps = PsiPhi[self.idx_s] if self.idx_s is not None else 0.0
        adj_factor = self.POLARITY * (pd - ps)

        dId_dBn = res["dId_dBn"]
        dId_dVTO = res["dId_dVTO"]

        # The Chain Rule Map: How does changing a parameter affect Bn?
        # partial Bn / partial X
        chain_rules = {
            "W": self.KP / self.L,
            "L": -self.W * self.KP / (self.L**2),
            "KP": self.W / self.L,
            "MU": (self.W / self.L) * getattr(self, "COX", 0.0) if hasattr(self, "COX") else 0.0,
            "COX": (self.W / self.L) * getattr(self, "MU", 0.0) if hasattr(self, "MU") else 0.0
        }

        sens_dict = {}
        
        # Dynamically loop through whatever parameters the object has loaded
        for attr in self._diff_attrs:
            if not hasattr(self, attr):
                continue # Skip if the user didn't define this in the netlist
                
            if attr == "VTO":
                sens_val = adj_factor * dId_dVTO
            else:
                # Chain rule: (dId/dBn) * (dBn/dAttr)
                dBn_dAttr = chain_rules.get(attr, 0.0)
                sens_val = adj_factor * dId_dBn * dBn_dAttr
                
            sens_dict[f"{self.name}_{attr}"] = sens_val

        return sens_dict

    def stamp_PQ(self, P, Q, col_idx):
        """Stamps the MOSFET linearized topology (VCCS equivalent).
        
        Current Injection (P): Drain to Source.
        Voltage Extraction (Q): Gate to Source.
        """
        d, g, s = self.idx_d, self.idx_g, self.idx_s
        
        # P: Current injection (Drain +, Source -) adjusted by POLARITY
        if d is not None: P[d, col_idx] = 1.0 * self.POLARITY
        if s is not None: P[s, col_idx] = -1.0 * self.POLARITY
        
        # Q: Control voltage measurement (Gate +, Source -) adjusted by POLARITY
        if g is not None: Q[g, col_idx] = 1.0 * self.POLARITY
        if s is not None: Q[s, col_idx] = -1.0 * self.POLARITY

    def get_delta_y(self, param_name, dp, V_nom=None, V_k=None, **kwargs):
        """Calculates Woodbury Admittance shifts exactly per the slides."""
        # 1. Nominal gm (evaluated at baseline V_nom)
        res_nom = self.evaluate_physics(v_k=V_nom)
        gm_nom = res_nom.get("gm", 0.0)
        
        # 2. Shifted gm (evaluated at current Newton-Raphson guess V_k)
        res_new = self.evaluate_physics(v_k=V_k, param_name=param_name, dp=dp)
        gm_new = res_new.get("gm", 0.0)
        
        return gm_new - gm_nom


# --- The Sibling Subclasses ---

class NMOS(Mosfet):
    """N-Channel MOSFET."""
    POLARITY = 1.0

class PMOS(Mosfet):
    """P-Channel MOSFET."""
    POLARITY = -1.0
