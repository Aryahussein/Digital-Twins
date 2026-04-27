from .base import Component
import core.models as models


class Mosfet(Component):
    """Abstract Base Class for MOSFETs. Do not instantiate directly."""

    POLARITY: float
    IS_NONLINEAR = True

    def __init__(self, name, data_dict):
        super().__init__(name, data_dict)

        # 2. Runtime Safety Check (Prevents Developer Errors)
        # If someone forgets to define POLARITY in a subclass, or tries
        # to instantiate the base MOS class directly, crash immediately.
        if not hasattr(self, "POLARITY"):
            raise NotImplementedError(
                f"Cannot instantiate '{self.__class__.__name__}'. "
                "Subclasses of MOS must define a 'POLARITY' constant."
            )

    @property
    def differentiable_params(self):
        """Overrides the base Component property to advertise MOS-specific parameters."""
        return [f"{self.name}_W", f"{self.name}_L", f"{self.name}_VTO"]

    def bind_nodes(self, node_map):
        self.idx_d = node_map.get(self.data.get("n_d", 0))
        self.idx_g = node_map.get(self.data.get("n_g", 0))
        self.idx_s = node_map.get(self.data.get("n_s", 0))

        params = self.data.get("model_params", {})
        inst = self.data.get("inst_params", {})

        # Absolute value ensures the physics equations always see a positive threshold
        self.VTO = abs(params.get("VTO", 0.7))
        self.W = inst.get("W", 1e-6)
        self.L = inst.get("L", 1e-6)

        mu = params.get("MU", 600e-4)
        cox = params.get("C_OX", 3.45e-3)
        self.KP = params.get("KP", mu * cox)
        self.Bn = (self.W / self.L) * self.KP

    def stamp_nonlinear(self, Y, sources, p_V_guess, V_guess):
        vd = V_guess[self.idx_d] if self.idx_d is not None else 0.0
        vg = V_guess[self.idx_g] if self.idx_g is not None else 0.0
        vs = V_guess[self.idx_s] if self.idx_s is not None else 0.0

        # POLARITY automatically maps terminal voltages (VGS vs VSG)
        vgs = self.POLARITY * (vg - vs)
        vds = self.POLARITY * (vd - vs)

        res = models.evaluate_nmos(vgs, vds, self.VTO, self.Bn)
        Id, gm, gds = res["I_D"], res["gm"], res["gds"]
        self._last_gds = gds

        ieq = (Id - gm * vgs - gds * vds) * self.POLARITY

        # Matrix stamping
        if self.idx_d is not None:
            if self.idx_g is not None:
                Y[self.idx_d, self.idx_g] += gm
            if self.idx_s is not None:
                Y[self.idx_d, self.idx_s] -= gm + gds
            Y[self.idx_d, self.idx_d] += gds
            sources[self.idx_d] -= ieq

        if self.idx_s is not None:
            if self.idx_g is not None:
                Y[self.idx_s, self.idx_g] -= gm
            if self.idx_d is not None:
                Y[self.idx_s, self.idx_d] -= gds
            Y[self.idx_s, self.idx_s] += gm + gds
            sources[self.idx_s] += ieq

    def get_sensitivities(self, VI, PsiPhi, **kwargs):
        vd, vg, vs = (
            VI[idx] if idx is not None else 0.0
            for idx in (self.idx_d, self.idx_g, self.idx_s)
        )
        pd, ps = (
            PsiPhi[idx] if idx is not None else 0.0 for idx in (self.idx_d, self.idx_s)
        )

        vgs = self.POLARITY * (vg - vs)
        vds = self.POLARITY * (vd - vs)
        adj_factor = self.POLARITY * (pd - ps)

        res = models.evaluate_nmos(vgs, vds, self.VTO, self.Bn)
        dId_dBn = res["dId_dBn"]
        dId_dVTO = res["dId_dVTO"]

        return {
            f"{self.name}_W": adj_factor * dId_dBn * (self.KP / self.L),
            f"{self.name}_L": adj_factor * dId_dBn * (-self.W * self.KP / (self.L**2)),
            f"{self.name}_VTO": adj_factor * dId_dVTO,
        }


# --- The Sibling Subclasses ---


class NMOS(Mosfet):
    """N-Channel MOSFET."""

    POLARITY = 1.0


class PMOS(Mosfet):
    """P-Channel MOSFET."""

    POLARITY = -1.0
