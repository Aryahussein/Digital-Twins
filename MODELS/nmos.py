import numpy as np
from MODELS.base import Component


# Voltage limiting constants (SPICE-style)
_VGS_MAX  =  1.5    # clamp Vgs to avoid gm explosion
_VGS_MIN  = -0.5
_VDS_MAX  =  2.0    # clamp Vds to keep gds finite
_VDS_MIN  = -0.5


class NMOS(Component):

    def __init__(self, name, d, g, s, b="0",
                 Vth=0.4, k=200e-6, lam=0.05):

        self.name = name
        self.d = d
        self.g = g
        self.s = s
        self.b = b

        self.Vth = Vth
        self.k   = k
        self.lam = lam

    def stamp(self, G, b, ctx):

        node_index = ctx["node_index"]
        x = ctx["x"]

        def v(n):
            return x[node_index[n]] if n != "0" else 0.0

        vd = v(self.d)
        vg = v(self.g)
        vs = v(self.s)

        # ---- Voltage limiting (prevents gm/gds blow-up during Newton) ----
        Vgs = np.clip(vg - vs, _VGS_MIN, _VGS_MAX)
        Vds = np.clip(vd - vs, _VDS_MIN, _VDS_MAX)

        # ---- Region detection ----
        Vov = Vgs - self.Vth

        if Vov <= 0:
            # Cut-off
            Id  = 0.0
            gm  = 0.0
            gds = 0.0

        elif Vds < Vov:
            # Linear (triode)
            Id  = self.k * (Vov * Vds - 0.5 * Vds**2)
            gm  = self.k * Vds
            gds = self.k * (Vov - Vds)

        else:
            # Saturation
            Id  = 0.5 * self.k * Vov**2 * (1.0 + self.lam * Vds)
            gm  = self.k * Vov * (1.0 + self.lam * Vds)
            gds = 0.5 * self.k * Vov**2 * self.lam

        # ---- Node indices ----
        d, s, g = self.d, self.s, self.g

        di = node_index[d] if d != "0" else None
        si = node_index[s] if s != "0" else None
        gi = node_index[g] if g != "0" else None

        # ---- MNA stamp (symmetric KCL form) ----
        #
        # The linearised drain current is:
        #   Id = gm*(Vg - Vs) + gds*(Vd - Vs)
        #      = gm*Vg - gm*Vs + gds*Vd - gds*Vs
        #
        # KCL at drain  (+Id flows into D):
        #   +gds*Vd - gds*Vs + gm*Vg - gm*Vs
        # KCL at source (-Id flows into S):
        #   -gds*Vd + gds*Vs - gm*Vg + gm*Vs
        #   = +(gds+gm)*Vs - gds*Vd - gm*Vg
        #
        # G-matrix contributions:
        #   G[di,di] += gds
        #   G[di,si] -= gds           (not gds+gm — gm appears via gate)
        #   G[si,di] -= gds           (symmetric to G[di,si])
        #   G[si,si] += gds + gm
        #   G[di,gi] += gm
        #   G[si,gi] -= gm

        if di is not None:
            G[di, di] += gds
        if si is not None:
            G[si, si] += gds + gm
        if di is not None and si is not None:
            G[di, si] -= gds          # BUG WAS: -(gds+gm) here — wrong, asymmetric
            G[si, di] -= gds          # BUG WAS: -gds — correct but paired wrong above
        if gi is not None and di is not None:
            G[di, gi] += gm
        if gi is not None and si is not None:
            G[si, gi] -= gm

        # ---- RHS (Norton equivalent current source) ----
        # Ieq = Id_op - gm*Vgs_op - gds*Vds_op  (linearisation residual)
        Ieq = Id - gm * Vgs - gds * Vds

        if di is not None:
            b[di] -= Ieq
        if si is not None:
            b[si] += Ieq