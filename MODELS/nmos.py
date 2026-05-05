import numpy as np
from MODELS.base import Component


_VGS_MAX =  1.5
_VGS_MIN = -0.5
_VDS_MAX =  2.0
_VDS_MIN = -0.5


class NMOS(Component):

    def __init__(self, name, d, g, s, b="0",
                 Vth=0.4,
                 muCox=200e-6,   # μn*Cox
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

        # Derived parameters
        self.k = muCox * (W / L)

        # Capacitances scale with width
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

        Vgs = np.clip(vg - vs, _VGS_MIN, _VGS_MAX)
        Vds = np.clip(vd - vs, _VDS_MIN, _VDS_MAX)
        Vov = Vgs - self.Vth

        # ── Region equations ─────────────────────────
        if Vov <= 0:
            Id  = 0.0
            gm  = 0.0
            gds = 0.0

        elif Vds < Vov:
            B   = 1.0 + self.lam * Vds
            A   = Vov * Vds - 0.5 * Vds**2

            Id  = self.k * A * B
            gm  = self.k * Vds * B
            gds = self.k * ((Vov - Vds) * B + A * self.lam)

        else:
            Id  = 0.5 * self.k * Vov**2 * (1.0 + self.lam * Vds)
            gm  = self.k * Vov * (1.0 + self.lam * Vds)
            gds = 0.5 * self.k * Vov**2 * self.lam

        di = node_index[self.d] if self.d != "0" else None
        si = node_index[self.s] if self.s != "0" else None
        gi = node_index[self.g] if self.g != "0" else None

        # ── DC and TRAN stamping (Newton-linearised) ─────────────────────────
        if analysis in ("dc", "tran"):

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

            # ── Capacitor companion models (tran only) ────────────────────
            # Backward Euler: C stamps as conductance G=C/dt with history current
            # I_hist = (C/dt) * V_prev
            if analysis == "tran":
                dt     = ctx["dt"]
                x_prev = ctx["x_prev"]

                def v_prev(n):
                    return x_prev[node_index[n]] if n != "0" else 0.0

                # Cgs companion: between g and s
                Ggs = self.Cgs / dt
                Vgs_prev = v_prev(self.g) - v_prev(self.s)
                Igs_hist = Ggs * Vgs_prev

                if gi is not None:
                    G[gi, gi] += Ggs
                    b[gi]     += Igs_hist
                if si is not None:
                    G[si, si] += Ggs
                    b[si]     -= Igs_hist
                if gi is not None and si is not None:
                    G[gi, si] -= Ggs
                    G[si, gi] -= Ggs

                # Cgd companion: between g and d
                Ggd = self.Cgd / dt
                Vgd_prev = v_prev(self.g) - v_prev(self.d)
                Igd_hist = Ggd * Vgd_prev

                if gi is not None:
                    G[gi, gi] += Ggd
                    b[gi]     += Igd_hist
                if di is not None:
                    G[di, di] += Ggd
                    b[di]     -= Igd_hist
                if gi is not None and di is not None:
                    G[gi, di] -= Ggd
                    G[di, gi] -= Ggd

        # ── AC stamping ─────────────────────────
        elif analysis == "ac":

            jw = ctx["jw"]

            # gm/gds
            if di is not None:
                G[di, di] += gds
            if si is not None:
                G[si, si] += gds + gm
            if di is not None and si is not None:
                G[di, si] -= gds
                G[si, di] -= gds
            if gi is not None and di is not None:
                G[di, gi] += gm
            if gi is not None and si is not None:
                G[si, gi] -= gm

            # Capacitors
            Ygs = jw * self.Cgs
            Ygd = jw * self.Cgd

            # Cgs
            if gi is not None:
                G[gi, gi] += Ygs
            if si is not None:
                G[si, si] += Ygs
            if gi is not None and si is not None:
                G[gi, si] -= Ygs
                G[si, gi] -= Ygs

            # Cgd
            if gi is not None:
                G[gi, gi] += Ygd
            if di is not None:
                G[di, di] += Ygd
            if gi is not None and di is not None:
                G[gi, di] -= Ygd
                G[di, gi] -= Ygd