import numpy as np
from MODELS.base import Component


# Voltage limiting constants (SPICE-style, in PMOS sign convention)
_VSG_MAX  =  1.5    # clamp Vsg to avoid gm explosion
_VSG_MIN  = -0.5
_VSD_MAX  =  2.0
_VSD_MIN  = -0.5


class PMOS(Component):

    def __init__(self, name, d, g, s, b="0",
                 Vth=-0.4, k=100e-6, lam=0.05):

        self.name = name
        self.d = d
        self.g = g
        self.s = s
        self.b = b

        self.Vth = Vth  # negative for PMOS
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

        # ---- Voltage limiting ----
        # PMOS operates in Vsg / Vsd convention
        Vsg = np.clip(vs - vg, _VSG_MIN, _VSG_MAX)
        Vsd = np.clip(vs - vd, _VSD_MIN, _VSD_MAX)

        Vtp = abs(self.Vth)   # threshold magnitude (positive)
        Vov = Vsg - Vtp

        # ---- Region detection (all quantities positive when active) ----
        if Vov <= 0:
            # Cut-off
            Id_mag = 0.0
            gm     = 0.0
            gds    = 0.0

        elif Vsd < Vov:
            # Linear (triode)
            Id_mag = self.k * (Vov * Vsd - 0.5 * Vsd**2)
            gm     = self.k * Vsd
            gds    = self.k * (Vov - Vsd)

        else:
            # Saturation
            Id_mag = 0.5 * self.k * Vov**2 * (1.0 + self.lam * Vsd)
            gm     = self.k * Vov * (1.0 + self.lam * Vsd)
            gds    = 0.5 * self.k * Vov**2 * self.lam

        # ---- Node indices ----
        d, s, g = self.d, self.s, self.g

        di = node_index[d] if d != "0" else None
        si = node_index[s] if s != "0" else None
        gi = node_index[g] if g != "0" else None

        # ---- MNA stamp ----
        #
        # PMOS convention: conventional current flows S → D (drain is the low side).
        # In terms of node voltages:
        #   Id (S→D) = gm*(Vs - Vg) + gds*(Vs - Vd)
        #            = gm*Vsg + gds*Vsd        (using clamped values)
        #
        # KCL at drain  (current flows IN to drain from source, so +Id at D):
        #   b[di] contribution: +gds*(Vs-Vd) + gm*(Vs-Vg)
        #   G[di,di] += gds
        #   G[di,si] -= gds
        #   G[di,gi] -= gm   ← BUG WAS: +gm (wrong sign, didn't account for PMOS direction)
        #
        # KCL at source (current leaves source, so -Id at S):
        #   G[si,si] += gds + gm
        #   G[si,di] -= gds
        #   G[si,gi] += gm   ← BUG WAS: -gm (wrong sign)

        if di is not None:
            G[di, di] += gds
        if si is not None:
            G[si, si] += gds + gm
        if di is not None and si is not None:
            G[di, si] -= gds
            G[si, di] -= gds
        if gi is not None and di is not None:
            G[di, gi] -= gm   # fixed sign
        if gi is not None and si is not None:
            G[si, gi] += gm   # fixed sign

        # ---- RHS ----
        # Ieq = Id_op - gm*Vsg_op - gds*Vsd_op
        Ieq = Id_mag - gm * Vsg - gds * Vsd

        # Current flows S→D: enters drain (+), leaves source (-)
        if di is not None:
            b[di] += Ieq   # BUG WAS: -Ieq (wrong sign convention)
        if si is not None:
            b[si] -= Ieq   # BUG WAS: +Ieq