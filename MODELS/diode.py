from MODELS.base import Component
import numpy as np


class Diode(Component):
    def __init__(self, name, n1, n2, Is=1e-12, Vt=0.02585):
        super().__init__(name, (n1, n2))
        self.Is = Is
        self.Vt = Vt

    # ==========================
    # Diode current
    # ==========================
    def current(self, Vd):
        return self.Is * (np.exp(Vd / self.Vt) - 1)

    # ==========================
    # Conductance dI/dV
    # ==========================
    def conductance(self, Vd):
        return (self.Is / self.Vt) * np.exp(Vd / self.Vt)

    # ==========================
    # Stamp (linearized)
    # ==========================
    def stamp(self, G, b, ctx):

        node_index = ctx["node_index"]
        x = ctx.get("x", None)   # Newton state

        n1, n2 = self.nodes

        # Voltages
        v1 = x[node_index[n1]] if (x is not None and n1 != "0") else 0
        v2 = x[node_index[n2]] if (x is not None and n2 != "0") else 0

        Vd = v1 - v2

        # Voltage limiting (SPICE-style clamp)
        Vd = np.clip(Vd, -1, 0.8)

        Gd = self.conductance(Vd)
        Id = self.current(Vd)

        Ieq = Id - Gd * Vd

        # Stamp like resistor + current source

        if n1 != "0":
            i = node_index[n1]
            G[i, i] += Gd
            b[i] -= Ieq

        if n2 != "0":
            j = node_index[n2]
            G[j, j] += Gd
            b[j] += Ieq

        if n1 != "0" and n2 != "0":
            i = node_index[n1]
            j = node_index[n2]
            G[i, j] -= Gd
            G[j, i] -= Gd

    # ==========================
    # Sensitivity (optional later)
    # ==========================
    def sens_contribution(self, X, lam, ctx):
        return None