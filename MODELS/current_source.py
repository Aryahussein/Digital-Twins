from MODELS.base import Component

class CurrentSource(Component):
    def __init__(self, name, n1, n2, value):
        super().__init__(name, (n1, n2))
        self.value = value

    def stamp(self, G, b, ctx):

        n1, n2 = self.nodes
        node_index = ctx["node_index"]

        if n1 != "0":
            b[node_index[n1]] -= self.value

        if n2 != "0":
            b[node_index[n2]] += self.value