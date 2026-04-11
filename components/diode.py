"""Diode Component Module."""

from .base import Component
import core.models as models


class Diode(Component):
    """Nonlinear Diode (Type 'D') using the Shockley equation.
    
    The diode is stamped using Newton-Raphson linearization:
        I_NR = gd * V_d + I_eq
    where gd is the small-signal conductance and I_eq is the equivalent
    current source for the linearized model.
    
    IS can be specified in three ways:
      1. Inline value:  D1 2 0 1e-14
      2. Model params:  D1 2 0 DMOD / .MODEL DMOD D (IS=1e-14)
      3. Default:       1e-14 if neither is specified
    """

    IS_NONLINEAR = True

    def bind_nodes(self, node_map):
        """Maps Anode (n1) and Cathode (n2) to matrix indices."""
        self.idx_a = node_map.get(self.data.get("n1", 0))
        self.idx_k = node_map.get(self.data.get("n2", 0))
        
        # IS priority: model_params > inline value > default
        params = self.data.get("model_params", {})
        if "IS" in params:
            self.IS = params["IS"]
        elif self.value is not None and self.value > 0:
            self.IS = self.value
        else:
            self.IS = 1e-14
            
        self.VT = params.get("VT", 0.02585)

    def stamp_nonlinear(self, Y, sources, p_V_guess, V_guess):
        """Stamps linearized gd and Ieq into the MNA system."""
        va = V_guess[self.idx_a] if self.idx_a is not None else 0.0
        vk = V_guess[self.idx_k] if self.idx_k is not None else 0.0
        vd = va - vk

        res = models.evaluate_diode(vd, self.IS, self.VT)
        id_val, gd = res["I_D"], res["gd"]

        ieq = id_val - gd * vd

        if self.idx_a is not None:
            Y[self.idx_a, self.idx_a] += gd
            sources[self.idx_a] -= ieq
            if self.idx_k is not None:
                Y[self.idx_a, self.idx_k] -= gd

        if self.idx_k is not None:
            Y[self.idx_k, self.idx_k] += gd
            sources[self.idx_k] += ieq
            if self.idx_a is not None:
                Y[self.idx_k, self.idx_a] -= gd

    def get_sensitivities(self, VI, PsiPhi, w=0.0, dt=None, V_prev=None, method='BE'):
        """Sensitivity w.r.t. Saturation Current (IS).
        
        The diode stamps current as:
            KCL at anode:  -I_D  (current leaves)
            KCL at cathode: +I_D  (current enters)
            
        So df_anode/dIS = -dI_D/dIS and df_cathode/dIS = +dI_D/dIS.
        
        Adjoint: sens = psi_a*(-dI_D/dIS) + psi_k*(+dI_D/dIS) = -(psi_a - psi_k)*dI_D/dIS
        """
        va = VI[self.idx_a] if self.idx_a is not None else 0.0
        vk = VI[self.idx_k] if self.idx_k is not None else 0.0
        
        pa = PsiPhi[self.idx_a] if self.idx_a is not None else 0.0
        pk = PsiPhi[self.idx_k] if self.idx_k is not None else 0.0
        
        res = models.evaluate_diode(va - vk, self.IS, self.VT)
        
        return {f"{self.name}_IS": -(pa - pk) * res["dId_dIs"]}

    def get_noise_sources(self, VI, w):
        """Diode shot noise: S_id = 2*q*|Id|  A²/Hz."""
        import core.constants as const
        va = VI[self.idx_a] if self.idx_a is not None else 0.0
        vk = VI[self.idx_k] if self.idx_k is not None else 0.0
        res = models.evaluate_diode(va - vk, self.IS, self.VT)
        Id = abs(res['I_D'])
        S = 2.0 * const.e * Id
        return [{'nodes': (self.idx_a, self.idx_k),
                 'S': S,
                 'label': f'{self.name}_shot'}]
