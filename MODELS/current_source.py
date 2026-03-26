from MODELS.base import Component


class CurrentSource(Component):
    def __init__(self, name, n1, n2, value):
        super().__init__(name, (n1, n2))
        self.value = value  # current flows n1 → n2 through source (SPICE convention)

    def stamp(self, G, b, ctx):

        n1, n2 = self.nodes
        node_index = ctx["node_index"]

        # SPICE convention: Iname n1 n2 value
        # Current flows FROM n1 TO n2 *inside* the source,
        # so conventional current ENTERS the circuit at n1 and LEAVES at n2.
        # KCL: +I injected at n1, -I injected at n2.
        #
        # BUG WAS: signs were swapped (b[n1] -= I, b[n2] += I),
        # which pulled current OUT of n1 and INTO n2 — backwards.

        # Apply source stepping scale if present (matches dc_solver ramp)
        scale = ctx.get("source_scale", 1.0)
        I = self.value * scale

        if n1 != "0":
            b[node_index[n1]] += I   # current leaves circuit at n1 (+I injected)

        if n2 != "0":
            b[node_index[n2]] -= I   # current enters circuit at n2 (-I injected)