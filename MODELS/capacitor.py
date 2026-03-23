from MODELS.base import Component

class Capacitor(Component):
    def __init__(self, name, n1, n2, value):
        super().__init__(name, (n1, n2))
        self.C = value

    def stamp(self, G, b, ctx):

        n1, n2 = self.nodes
        node_index = ctx["node_index"]
        analysis = ctx["analysis"]

        if analysis == "dc":
            return

        elif analysis == "ac":
            jw = ctx["jw"]
            g = jw * self.C

        elif analysis == "tran":
            dt = ctx["dt"]
            x_prev = ctx["x_prev"]

            g = self.C / dt

            v_prev = 0
            if n1 != "0":
                v_prev += x_prev[node_index[n1]]
            if n2 != "0":
                v_prev -= x_prev[node_index[n2]]

            Ieq = g * v_prev

            if n1 != "0":
                b[node_index[n1]] += Ieq
            if n2 != "0":
                b[node_index[n2]] -= Ieq

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