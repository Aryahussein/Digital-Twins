import numpy as np
from MODELS.base import Component


_VSG_MAX =  1.5
_VSG_MIN = -0.5
_VSD_MAX =  2.0
_VSD_MIN = -0.5


class PMOS(Component):

    def __init__(self, name, d, g, s, b="0",
                 Vth=-0.4, k=100e-6, lam=0.05):
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

        Vsg = np.clip(vs - vg, _VSG_MIN, _VSG_MAX)
        Vsd = np.clip(vs - vd, _VSD_MIN, _VSD_MAX)
        Vtp = abs(self.Vth)
        Vov = Vsg - Vtp

        if Vov <= 0:
            # ── Cut-off ───────────────────────────────────────────────────────
            Id_mag = 0.0
            gm     = 0.0
            gds    = 0.0

        elif Vsd < Vov:
            # ── Linear (triode) ───────────────────────────────────────────────
            #
            # Symmetric to NMOS with Vsd/Vsg in place of Vds/Vgs:
            #   Id = k * (Vov*Vsd - 0.5*Vsd²) * (1 + λ*Vsd)
            #   gm  = k * Vsd * (1 + λ*Vsd)
            #   gds = k * [(Vov-Vsd)*(1+λ*Vsd) + (Vov*Vsd - 0.5*Vsd²)*λ]

            B      = 1.0 + self.lam * Vsd
            A      = Vov * Vsd - 0.5 * Vsd**2

            Id_mag = self.k * A * B
            gm     = self.k * Vsd * B
            gds    = self.k * ((Vov - Vsd) * B + A * self.lam)

        else:
            # ── Saturation ────────────────────────────────────────────────────
            Id_mag = 0.5 * self.k * Vov**2 * (1.0 + self.lam * Vsd)
            gm     = self.k * Vov * (1.0 + self.lam * Vsd)
            gds    = 0.5 * self.k * Vov**2 * self.lam

        # ── Node indices ──────────────────────────────────────────────────────
        di = node_index[self.d] if self.d != "0" else None
        si = node_index[self.s] if self.s != "0" else None
        gi = node_index[self.g] if self.g != "0" else None

        # ── MNA stamp ─────────────────────────────────────────────────────────
        #
        # PMOS: Id_mag flows S→D. Partial derivatives w.r.t. node voltages:
        #   ∂Id/∂Vg = -gm,  ∂Id/∂Vd = -gds,  ∂Id/∂Vs = +(gm+gds)
        #
        # Linearised: Id_lin = Ieq - gm*Vg - gds*Vd + (gm+gds)*Vs
        # Ieq = Id_mag - gm*Vsg - gds*Vsd
        #
        # KCL: +Id_lin into drain, -Id_lin out of source:
        #
        #   Drain row:
        #     G[d,d] += gds          G[d,s] -= (gm+gds)    G[d,g] += gm
        #     b[d]   += Ieq
        #
        #   Source row:
        #     G[s,s] += (gm+gds)     G[s,d] -= gds          G[s,g] -= gm
        #     b[s]   -= Ieq

        Ieq = Id_mag - gm * Vsg - gds * Vsd

        if di is not None:
            G[di, di] += gds
            b[di]     += Ieq
        if si is not None:
            G[si, si] += (gm + gds)
            b[si]     -= Ieq
        if di is not None and si is not None:
            G[di, si] -= (gm + gds)
            G[si, di] -= gds
        if di is not None and gi is not None:
            G[di, gi] += gm
        if si is not None and gi is not None:
            G[si, gi] -= gm