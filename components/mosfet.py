"""MOSFET Component Module.

Implements Level 1 (Shichman-Hodges) NMOS and PMOS transistors using a 
shared base class with a POLARITY constant to handle the symmetric equations.
"""

from .base import Component
import core.models as models
import warnings
import numpy as np


class Mosfet(Component):
    """Abstract Base Class for Level 1 MOSFETs. Do not instantiate directly.
    
    Subclasses must define POLARITY:
        NMOS: POLARITY = +1.0 (VGS = VG - VS, VDS = VD - VS)
        PMOS: POLARITY = -1.0 (VSG = VS - VG, VSD = VS - VD)
    """

    POLARITY: float 
    IS_NONLINEAR = True
    NODE_KEYS = ("n_d", "n_g", "n_s", "n_b")

    def __init__(self, name, data_dict):
        super().__init__(name, data_dict)
        
        if not hasattr(self, 'POLARITY'):
            raise NotImplementedError(
                f"Cannot instantiate '{self.__class__.__name__}'. "
                "Subclasses of Mosfet must define a 'POLARITY' constant."
            )
    
    def bind_nodes(self, node_map):
        self.idx_d = node_map.get(self.data.get("n_d", 0))
        self.idx_g = node_map.get(self.data.get("n_g", 0))
        self.idx_s = node_map.get(self.data.get("n_s", 0))
        self.idx_b = node_map.get(self.data.get("n_b", 0))

        # Warn if bulk terminal is connected to a non-ground node,
        # since body effect is not yet modeled
        n_b_val = self.data.get("n_b", 0)
        if n_b_val != 0 and n_b_val != "0" and str(n_b_val).upper() != "GND":
            if self.idx_b != self.idx_s:
                warnings.warn(
                    f"MOSFET '{self.name}': Bulk terminal (node '{n_b_val}') is not "
                    f"connected to source. Body effect is not modeled in Level 1 — "
                    f"the bulk connection will be ignored."
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

    def stamp_nonlinear(self, Y, sources, p_V_guess, V_guess):
        """Stamps NR-linearized drain current into the MNA system."""
        vd = V_guess[self.idx_d] if self.idx_d is not None else 0.0
        vg = V_guess[self.idx_g] if self.idx_g is not None else 0.0
        vs = V_guess[self.idx_s] if self.idx_s is not None else 0.0
        
        vgs = self.POLARITY * (vg - vs)
        vds = self.POLARITY * (vd - vs)
        
        res = models.evaluate_mosfet_level1(vgs, vds, self.VTO, self.Bn)
        Id, gm, gds = res["I_D"], res["gm"], res["gds"]

        ieq = (Id - gm * vgs - gds * vds) * self.POLARITY

        if self.idx_d is not None:
            if self.idx_g is not None: Y[self.idx_d, self.idx_g] += gm
            if self.idx_s is not None: Y[self.idx_d, self.idx_s] -= (gm + gds)
            Y[self.idx_d, self.idx_d] += gds
            sources[self.idx_d] -= ieq

        if self.idx_s is not None:
            if self.idx_g is not None: Y[self.idx_s, self.idx_g] -= gm
            if self.idx_d is not None: Y[self.idx_s, self.idx_d] -= gds
            Y[self.idx_s, self.idx_s] += (gm + gds)
            sources[self.idx_s] += ieq

    def get_sensitivities(self, VI, PsiPhi, w=0.0, dt=None, V_prev=None, method='BE'):
        """Sensitivity w.r.t. W, L, and VTO parameters.
        
        The MOSFET stamps drain current as:
            KCL at drain:  -I_D * POLARITY  (current leaves for NMOS)
            KCL at source: +I_D * POLARITY  (current enters for NMOS)
            
        So df_drain/dp = POLARITY * dI_D/dp, and the adjoint formula is:
            sens = -[psi_d * POLARITY * dI_D/dp + psi_s * (-POLARITY * dI_D/dp)]
                 = -POLARITY * (psi_d - psi_s) * dI_D/dp
        """
        vd, vg, vs = (VI[idx] if idx is not None else 0.0 for idx in (self.idx_d, self.idx_g, self.idx_s))
        pd, ps = (PsiPhi[idx] if idx is not None else 0.0 for idx in (self.idx_d, self.idx_s))
        
        vgs = self.POLARITY * (vg - vs)
        vds = self.POLARITY * (vd - vs)
        adj_factor = -self.POLARITY * (pd - ps)
        
        res = models.evaluate_mosfet_level1(vgs, vds, self.VTO, self.Bn)
        dId_dBn = res["dId_dBn"]
        dId_dVTO = res["dId_dVTO"]

        return {
            f"{self.name}_W": adj_factor * dId_dBn * (self.KP / self.L),
            f"{self.name}_L": adj_factor * dId_dBn * (-self.W * self.KP / (self.L**2)),
            f"{self.name}_VTO": adj_factor * dId_dVTO
        }

    def get_noise_sources(self, VI, w):
        """MOSFET noise sources.

        Thermal channel noise:  S_id = 4kT * (2/3) * gm   A²/Hz
        Flicker (1/f) noise:    S_id = KF * Id / (Cox*L^2 * f)  A²/Hz
          where KF defaults to 1e-24 A²·s/F if not specified in model.

        Both are current sources from drain to source.
        """
        import core.constants as const, core.models as models

        vd = float(np.real(VI[self.idx_d])) if self.idx_d is not None else 0.0
        vg = float(np.real(VI[self.idx_g])) if self.idx_g is not None else 0.0
        vs = float(np.real(VI[self.idx_s])) if self.idx_s is not None else 0.0

        vgs = self.POLARITY * (vg - vs)
        vds = self.POLARITY * (vd - vs)
        res = models.evaluate_mosfet_level1(vgs, vds, self.VTO, self.Bn)
        gm = float(res['gm'])
        Id = abs(float(res['I_D']))

        sources = []

        # Thermal channel noise (van der Ziel model, gamma=2/3)
        S_thermal = 4.0 * const.kb * const.T * (2.0/3.0) * gm
        sources.append({
            'nodes': (self.idx_d, self.idx_s),
            'S': float(S_thermal),
            'label': f'{self.name}_thermal'
        })

        # Flicker noise (1/f): S = KF*Id^AF/(Cox*L^2*f)
        params = self.data.get('model_params', {})
        KF  = float(params.get('KF', 1e-24))
        AF  = float(params.get('AF', 1.0))
        cox = float(params.get('C_OX', 3.45e-3))
        f   = w / (2.0 * 3.14159265) if w > 0 else 1.0
        S_flicker = KF * (Id ** AF) / (cox * self.L**2 * f)
        sources.append({
            'nodes': (self.idx_d, self.idx_s),
            'S': float(S_flicker),
            'label': f'{self.name}_flicker'
        })

        return sources


class NMOS(Mosfet):
    """N-Channel MOSFET."""
    POLARITY = 1.0


class PMOS(Mosfet):
    """P-Channel MOSFET."""
    POLARITY = -1.0

