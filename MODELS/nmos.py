import numpy as np
from MODELS.base import Component


_VGS_MAX =  1.5
_VGS_MIN = -0.5
_VDS_MAX =  2.0
_VDS_MIN = -0.5


class NMOS(Component):

    def __init__(self, name, d, g, s, b="0",
                 Vth=0.4, k=200e-6, lam=0.05):
        self.name = name
        self.d    = d
        self.g    = g
        self.s    = s
        self.b    = b
        self.Vth  = Vth
        self.k    = k
        self.lam  = lam

    def stamp(self, G, b, ctx):

        node_index = ctx["node_index"]
        x          = ctx["x"]

        def v(n):
            return x[node_index[n]] if n != "0" else 0.0

        vd = v(self.d)
        vg = v(self.g)
        vs = v(self.s)

        Vgs = np.clip(vg - vs, _VGS_MIN, _VGS_MAX)
        Vds = np.clip(vd - vs, _VDS_MIN, _VDS_MAX)
        Vov = Vgs - self.Vth

        if Vov <= 0:
            # ── Cut-off ───────────────────────────────────────────────────────
            Id  = 0.0
            gm  = 0.0
            gds = 0.0

        elif Vds < Vov:
            # ── Linear (triode) ───────────────────────────────────────────────
            #
            # Id = k * (Vov*Vds - 0.5*Vds²) * (1 + λ*Vds)
            #
            # Including (1+λ*Vds) here is essential — without it Id jumps
            # discontinuously at the sat/linear boundary.
            #
            # Derivatives (exact, product rule):
            #   gm  = dId/dVgs = k * Vds * (1 + λ*Vds)
            #   gds = dId/dVds = k * [(Vov-Vds)*(1+λ*Vds) + (Vov*Vds - 0.5*Vds²)*λ]
            #
            # At Vds = Vov these equal the saturation derivatives exactly,
            # so Id, gm, and gds are all continuous across the boundary.

            B   = 1.0 + self.lam * Vds
            A   = Vov * Vds - 0.5 * Vds**2

            Id  = self.k * A * B
            gm  = self.k * Vds * B
            gds = self.k * ((Vov - Vds) * B + A * self.lam)

        else:
            # ── Saturation ────────────────────────────────────────────────────
            #
            # Id = 0.5 * k * Vov² * (1 + λ*Vds)
            #
            # gm  = dId/dVgs = k * Vov * (1 + λ*Vds)
            # gds = dId/dVds = 0.5 * k * Vov² * λ

            Id  = 0.5 * self.k * Vov**2 * (1.0 + self.lam * Vds)
            gm  = self.k * Vov * (1.0 + self.lam * Vds)
            gds = 0.5 * self.k * Vov**2 * self.lam

        # ── Node indices ──────────────────────────────────────────────────────
        di = node_index[self.d] if self.d != "0" else None
        si = node_index[self.s] if self.s != "0" else None
        gi = node_index[self.g] if self.g != "0" else None

        # ── MNA stamp ─────────────────────────────────────────────────────────
        #
        # Linearised drain current:
        #   Id_lin = gm*(Vg-Vs) + gds*(Vd-Vs) + Ieq
        #   Ieq    = Id - gm*Vgs - gds*Vds
        #
        # KCL: +Id_lin into drain, -Id_lin into source:
        #
        #   G[d,d] += gds       G[d,s] -= gds      G[d,g] += gm
        #   G[s,s] += gds+gm    G[s,d] -= gds      G[s,g] -= gm
        #   b[d]   -= Ieq
        #   b[s]   += Ieq

        Ieq = Id - gm * Vgs - gds * Vds

        if di is not None:
            G[di, di] += gds
            b[di]     -= Ieq
        if si is not None:
            G[si, si] += gds + gm
            b[si]     += Ieq
        if di is not None and si is not None:
            G[di, si] -= gds
            G[si, di] -= gds
        if gi is not None and di is not None:
            G[di, gi] += gm
        if gi is not None and si is not None:
            G[si, gi] -= gm