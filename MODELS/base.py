class Component:
    def __init__(self, name, nodes):
        self.name = name
        self.nodes = nodes

    def stamp(self, G, b, ctx):
        raise NotImplementedError

    def sens_contribution(self, X, lam, ctx):
        return None