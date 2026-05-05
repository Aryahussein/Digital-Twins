import numpy as np
from MODELS.base import Component


_VSG_MAX =  1.5
_VSG_MIN = -0.5
_VSD_MAX =  2.0
_VSD_MIN = -0.5


class PMOS(Component):

    def __init__(self, name, d, g, s, b="0",
                 Vth=-0.4,
                 muCox=100e-6,
                 W=1e-6,
                 L=1e-6,
                 lam=0.05,
                 Cgs_per_W=1e-12,
                 Cgd_per_W=0.2e-12):

        self.name = name
        self.d    = d
        self.g    = g
        self.s    = s
        self.b    = b

        self.Vth  = Vth
        self.muCox = muCox
        self.W     = W
        self.L     = L
        self.lam   = lam

        self.k = muCox * (W / L)

        self.Cgs = Cgs_per_W * (W/1e-6)  # Convert to F
        self.Cgd = Cgd_per_W * (W/1e-6)  # Convert to F

    def stamp(self, G, b, ctx):

        node_index = ctx["node_index"]
        x          = ctx["x"]
        analysis   = ctx.get("analysis", "dc")

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
            Id_mag = 0.0
            gm     = 0.0
            gds    = 0.0

        elif Vsd < Vov:
            B      = 1.0 + self.lam * Vsd
            A      = Vov * Vsd - 0.5 * Vsd**2

            Id_mag = self.k * A * B
            gm     = self.k * Vsd * B
            gds    = self.k * ((Vov - Vsd) * B + A * self.lam)

        else:
            Id_mag = 0.5 * self.k * Vov**2 * (1.0 + self.lam * Vsd)
            gm     = self.k * Vov * (1.0 + self.lam * Vsd)
            gds    = 0.5 * self.k * Vov**2 * self.lam

        di = node_index[self.d] if self.d != "0" else None
        si = node_index[self.s] if self.s != "0" else None
        gi = node_index[self.g] if self.g != "0" else None

        if analysis == "dc":

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

        elif analysis == "ac":

            jw = ctx["jw"]

            if di is not None:
                G[di, di] += gds
            if si is not None:
                G[si, si] += (gm + gds)
            if di is not None and si is not None:
                G[di, si] -= (gm + gds)
                G[si, di] -= gds
            if di is not None and gi is not None:
                G[di, gi] += gm
            if si is not None and gi is not None:
                G[si, gi] -= gm

            Ygs = jw * self.Cgs
            Ygd = jw * self.Cgd

            if gi is not None:
                G[gi, gi] += Ygs
            if si is not None:
                G[si, si] += Ygs
            if gi is not None and si is not None:
                G[gi, si] -= Ygs
                G[si, gi] -= Ygs

            if gi is not None:
                G[gi, gi] += Ygd
            if di is not None:
                G[di, di] += Ygd
            if gi is not None and di is not None:
                G[gi, di] -= Ygd
                G[di, gi] -= Ygd