from MODELS.base import Component

class Inductor(Component):
    def __init__(self, name, n1, n2, value):
        super().__init__(name, (n1, n2))
        self.L = value

    def stamp(self, G, b, ctx):

        n1, n2 = self.nodes
        node_index = ctx["node_index"]
        analysis = ctx["analysis"]

        if analysis == "ac":
            jw = ctx["jw"]
            g = 1 / (jw * self.L)
        else:
            return

        if n1 != "0":
            G[node_index[n1], node_index[n1]] += g

        if n2 != "0":
            G[node_index[n2], node_index[n2]] += g

        if n1 != "0" and n2 != "0":
            i = node_index[n1]
            j = node_index[n2]
            G[i, j] -= g
            G[j, i] -= g