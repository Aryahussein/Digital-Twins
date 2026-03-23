from MODELS.base import Component

class Resistor(Component):
    def __init__(self, name, n1, n2, value):
        super().__init__(name, (n1, n2))
        self.R = value

    def stamp(self, G, b, ctx):

        g = 1 / self.R
        n1, n2 = self.nodes
        node_index = ctx["node_index"]

        if n1 != "0":
            i = node_index[n1]
            G[i, i] += g

        if n2 != "0":
            j = node_index[n2]
            G[j, j] += g

        if n1 != "0" and n2 != "0":
            i = node_index[n1]
            j = node_index[n2]
            G[i, j] -= g
            G[j, i] -= g

    # -------- Sensitivity --------
    def sens_contribution(self, X, lam, ctx):

        n1, n2 = self.nodes
        node_index = ctx["node_index"]

        V1 = X[node_index[n1]] if n1 != "0" else 0
        V2 = X[node_index[n2]] if n2 != "0" else 0

        L1 = lam[node_index[n1]] if n1 != "0" else 0
        L2 = lam[node_index[n2]] if n2 != "0" else 0

        return (1 / self.R**2) * (V1 - V2) * (L1 - L2)