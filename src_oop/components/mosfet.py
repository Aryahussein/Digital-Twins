from .base import Component
import core.models as models
import numpy as np


class Mosfet(Component):
    """Abstract Base Class for MOSFETs. Do not instantiate directly."""

    POLARITY: float
    IS_NONLINEAR = True

    def __init__(self, name, data_dict):
        super().__init__(name, data_dict)
        
        if not hasattr(self, "POLARITY"):
            raise NotImplementedError(
                f"Cannot instantiate '{self.__class__.__name__}'. "
                "Subclasses of MOS must define a 'POLARITY' constant."
            )

        params = self.data.get("model_params", {})
        inst = self.data.get("inst_params", {})

        self.VTO = abs(params.get("VTO", 0.7))
        self.W = inst.get("W", 1e-6)
        self.L = inst.get("L", 1e-6)
        
        mu = params.get("MU", 600e-4)
        cox = params.get("C_OX", 3.45e-3)
        self.KP = params.get("KP", mu * cox)
        self.Bn = (self.W / self.L) * self.KP

        self._diff_attrs = ["W", "L", "VTO"]

    @property
    def differentiable_params(self):
        """Dynamically advertises parameters based on the internal registry."""
        # Automatically builds ["M1_W", "M1_L", "M1_VTO"]
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
        """Translates string terminal names into matrix indices."""
        self.idx_d = node_map.get(self.data.get("n_d", 0))
        self.idx_g = node_map.get(self.data.get("n_g", 0))
        self.idx_s = node_map.get(self.data.get("n_s", 0))

    # ==========================================
    # THE PHYSICS HELPER
    # ==========================================
    def _evaluate_physics(self, V_array, param_name=None, dp=0.0):
        """Centralized helper for voltage extraction and device physics."""
        # 1. Extract Voltages
        vd = V_array[self.idx_d] if self.idx_d is not None else 0.0
        vg = V_array[self.idx_g] if self.idx_g is not None else 0.0
        vs = V_array[self.idx_s] if self.idx_s is not None else 0.0

        vgs = self.POLARITY * (vg - vs)
        vds = self.POLARITY * (vd - vs)

        # 2. Setup Parameters (apply shifts if requested)
        vto, w, l = self.VTO, self.W, self.L
        
        if param_name and dp != 0.0:
            if param_name.endswith("_VTO"): vto += dp
            elif param_name.endswith("_W"): w += dp
            elif param_name.endswith("_L"): l += dp
            
        bn = (w / l) * self.KP

        # 3. Evaluate Model
        res = models.evaluate_nmos(vgs, vds, vto, bn)
        return res, vgs, vds

    # ==========================================
    # SOLVER ENGINES
    # ==========================================
    def stamp_nonlinear(self, Y, sources, p_V_guess, V_guess):
        """Stamps absolute values into the matrix."""
        res, vgs, vds = self._evaluate_physics(V_guess)
        
        Id, gm, gds = res["I_D"], res["gm"], res["gds"]
        ieq = (Id - gm * vgs - gds * vds) * self.POLARITY

        # Matrix stamping
        if self.idx_d is not None:
            if self.idx_g is not None: Y[self.idx_d, self.idx_g] += gm
            if self.idx_s is not None: Y[self.idx_d, self.idx_s] -= gm + gds
            Y[self.idx_d, self.idx_d] += gds
            sources[self.idx_d] -= ieq

        if self.idx_s is not None:
            if self.idx_g is not None: Y[self.idx_s, self.idx_g] -= gm
            if self.idx_d is not None: Y[self.idx_s, self.idx_d] -= gds
            Y[self.idx_s, self.idx_s] += gm + gds
            sources[self.idx_s] += ieq

    # ==========================================
    # SENSITIVITY ENGINES (Adjoint & Woodbury)
    # ==========================================
    def get_sensitivities(self, VI, PsiPhi, **kwargs):
        """Calculates Adjoint derivatives."""
        res, _, _ = self._evaluate_physics(VI)
        
        pd = PsiPhi[self.idx_d] if self.idx_d is not None else 0.0
        ps = PsiPhi[self.idx_s] if self.idx_s is not None else 0.0
        adj_factor = self.POLARITY * (pd - ps)

        dId_dBn, dId_dVTO = res["dId_dBn"], res["dId_dVTO"]

        sens_map = {
            "W": adj_factor * dId_dBn * (self.KP / self.L),
            "L": adj_factor * dId_dBn * (-self.W * self.KP / (self.L**2)),
            "VTO": adj_factor * dId_dVTO,
            "KP": adj_factor * dId_dBn * (self.W / self.L)
        }

        return {f"{self.name}_{attr}": sens_map[attr] for attr in self._diff_attrs if attr in sens_map}

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

    def get_delta_y(self, param_name, dp, **kwargs):
        """Calculates Woodbury Admittance shifts."""
        # Using a default numpy array length is tricky here because total_dim isn't natively known by the component.
        # However, the engine guarantees 'VI' is passed in kwargs, so a direct fetch is safe.
        VI = kwargs.get("VI")
        if VI is None: return 0.0
        
        # Nominal gm
        res_nom, _, _ = self._evaluate_physics(VI)
        gm_nom = res_nom["gm"]
        
        # Shifted gm
        res_new, _, _ = self._evaluate_physics(VI, param_name=param_name, dp=dp)
        gm_new = res_new["gm"]
        
        return gm_new - gm_nom


# --- The Sibling Subclasses ---

class NMOS(Mosfet):
    """N-Channel MOSFET."""
    POLARITY = 1.0

class PMOS(Mosfet):
    """P-Channel MOSFET."""
    POLARITY = -1.0
