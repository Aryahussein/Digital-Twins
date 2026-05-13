from .base import Component
import core.models as models
import numpy as np


class Mosfet(Component):
    """Abstract Base Class for MOSFETs using generic MNA stamping."""

    POLARITY: float
    IS_NONLINEAR = True
    rank = 2

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
                return getattr(self, attr_name, 0.0)
                
        return super().get_nominal_value(param_name)


    def set_nominal_value(self, param_name, new_val):
        """Safely routes multi-parameter updates for Transistors."""
        prefix = f"{self.name}_"
        
        if param_name.startswith(prefix):
            attr_name = param_name[len(prefix):] 
            
            if attr_name in self._diff_attrs:
                setattr(self, attr_name, new_val)
                
                # Dynamically recalculate KP if MU or COX was swept
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
    def evaluate_physics(self, v_k=None, overrides=None, **kwargs):
        """Pure mathematical evaluation using absolute overrides."""
        res = {}
        if v_k is None: return res
        overrides = overrides or {}
            
        # 1. Extract Voltages
        vd = v_k[self.idx_d] if self.idx_d is not None else 0.0
        vg = v_k[self.idx_g] if self.idx_g is not None else 0.0
        vs = v_k[self.idx_s] if self.idx_s is not None else 0.0

        vgs = self.POLARITY * (vg - vs)
        vds = self.POLARITY * (vd - vs)

        # 2. PURE PHYSICS: Use overrides if they exist, otherwise use nominal
        eff_vto = overrides.get(f"{self.name}_VTO", self.VTO)
        eff_w = overrides.get(f"{self.name}_W", self.W)
        eff_l = overrides.get(f"{self.name}_L", self.L)
        
        eff_mu = overrides.get(f"{self.name}_MU", getattr(self, "MU", 0.0))
        eff_cox = overrides.get(f"{self.name}_COX", getattr(self, "COX", 0.0))

        # Clean override logic for the KP vs (MU*COX) relationship
        if f"{self.name}_KP" in overrides:
            eff_kp = overrides[f"{self.name}_KP"]
        elif f"{self.name}_MU" in overrides or f"{self.name}_COX" in overrides:
            eff_kp = eff_mu * eff_cox
        else:
            eff_kp = self.KP
            
        bn = (eff_w / eff_l) * eff_kp

        # 3. Core Physics Evaluation
        phys_res = models.evaluate_nmos(vgs, vds, eff_vto, bn)
        Id, gm, gds = phys_res["I_D"], phys_res["gm"], phys_res["gds"]
        
        # 4. Pack data for the generic Stampers
        res["gm"] = gm
        res["gds"] = gds
        res["I_eq"] = (Id - gm * vgs - gds * vds) * self.POLARITY
        
        # 5. Pack data for Sensitivity/Adjoint engines
        res["dId_dBn"] = phys_res["dId_dBn"]
        res["dId_dVTO"] = phys_res["dId_dVTO"]
        
        return res

    # ==========================================
    # 2. THE STAMPERS (Custom 3-Terminal Interface)
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
        """Stamps the Equivalent Current into the RHS vector.
        
        Perfectly matches the unified MNA convention:
        Current leaving the Drain (-) and entering the Source (+).
        """
        I_eq = res.get("I_eq", 0.0)
        if I_eq == 0.0: return
        
        d, s = self.idx_d, self.idx_s

        if d is not None: J[d] -= I_eq
        if s is not None: J[s] += I_eq

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
        chain_rules = {
            "W": self.KP / self.L,
            "L": -self.W * self.KP / (self.L**2),
            "KP": self.W / self.L,
            "MU": (self.W / self.L) * getattr(self, "COX", 0.0) if hasattr(self, "COX") else 0.0,
            "COX": (self.W / self.L) * getattr(self, "MU", 0.0) if hasattr(self, "MU") else 0.0
        }

        sens_dict = {}
        
        for attr in self._diff_attrs:
            if not hasattr(self, attr):
                continue 
                
            if attr == "VTO":
                sens_val = adj_factor * dId_dVTO
            else:
                dBn_dAttr = chain_rules.get(attr, 0.0)
                sens_val = adj_factor * dId_dBn * dBn_dAttr
                
            sens_dict[f"{self.name}_{attr}"] = sens_val

        return sens_dict


    def stamp_PQ(self, P, Q, start_col_idx):
        """Stamps the Rank-2 MOSFET linearized topology.
        Column 1: gm (Measures VGS, Injects IDS)
        Column 2: gds (Measures VDS, Injects IDS)
        """
        d, g, s = self.idx_d, self.idx_g, self.idx_s
        col_gm = start_col_idx
        col_gds = start_col_idx + 1
        
        # --- Column 1: Transconductance (gm) ---
        if d is not None: P[d, col_gm] = 1.0 * self.POLARITY
        if s is not None: P[s, col_gm] = -1.0 * self.POLARITY
        if g is not None: Q[g, col_gm] = 1.0 * self.POLARITY
        if s is not None: Q[s, col_gm] = -1.0 * self.POLARITY

        # --- Column 2: Output Conductance (gds) ---
        if d is not None: P[d, col_gds] = 1.0 * self.POLARITY
        if s is not None: P[s, col_gds] = -1.0 * self.POLARITY
        if d is not None: Q[d, col_gds] = 1.0 * self.POLARITY
        if s is not None: Q[s, col_gds] = -1.0 * self.POLARITY


    def get_delta_y(self, shifts=None, V_nom=None, V_k=None, **kwargs):
        """Returns a 2x2 block matrix containing both gm and gds shifts."""
        import numpy as np
        shifts = shifts or {}
        
        if not shifts and V_nom is not None and np.allclose(V_nom, V_k):
            return np.zeros((2, 2))

        # TRANSLATION LAYER: Convert shifts (deltas) into absolute overrides
        new_overrides = {}
        old_overrides = {}
        for param_name, delta in shifts.items():
            nom_val = self.get_nominal_value(param_name)
            new_overrides[param_name] = nom_val
            old_overrides[param_name] = nom_val - delta

        # 1. New State (Absolute Parameters)
        res_new = self.evaluate_physics(v_k=V_k, overrides=new_overrides, **kwargs)
        
        # 2. Old State (Absolute Parameters)
        res_old = self.evaluate_physics(v_k=V_nom, overrides=old_overrides, **kwargs)
        
        # 3. Extract the dual mathematical differences
        d_gm = res_new.get("gm", 0.0) - res_old.get("gm", 0.0)
        d_gds = res_new.get("gds", 0.0) - res_old.get("gds", 0.0)
        
        # Return as a 2x2 diagonal block
        return np.array([
            [d_gm, 0.0],
            [0.0, d_gds]
        ])

# --- The Sibling Subclasses ---

class NMOS(Mosfet):
    """N-Channel MOSFET."""
    POLARITY = 1.0

class PMOS(Mosfet):
    """P-Channel MOSFET."""
    POLARITY = -1.0
