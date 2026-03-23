from MODELS.base import Component

class OpAmp(Component):
    def __init__(self, name, nplus, nminus, nout, gain, index):
        super().__init__(name, (nplus, nminus, nout))
        self.gain = gain
        self.index = index

    def stamp(self, G, b, ctx):

        nplus, nminus, nout = self.nodes
        node_index = ctx["node_index"]
        N = ctx["N"]
        Mv = ctx["Mv"]

        row = N + Mv + self.index

        if nout != "0":
            G[row, node_index[nout]] = 1
            G[node_index[nout], row] = 1

        if nplus != "0":
            G[row, node_index[nplus]] -= self.gain

        if nminus != "0":
            G[row, node_index[nminus]] += self.gain