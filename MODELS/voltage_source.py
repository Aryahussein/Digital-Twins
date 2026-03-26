from MODELS.base import Component
import math


def evaluate_source(source_type, params, t):

    if source_type == "DC":
        return params[0]

    if source_type == "STEP":
        v1, v2, tdelay = params
        return v1 if t < tdelay else v2

    if source_type == "RAMP":
        v1, v2, tstart, trise = params
        if t < tstart:
            return v1
        if t > tstart + trise:
            return v2
        slope = (v2 - v1) / trise
        return v1 + slope * (t - tstart)

    if source_type == "SINE":
        voff = params[0]
        vamp = params[1]
        freq = params[2]
        phase_deg = params[3] if len(params) > 3 else 0.0
        phase_rad = math.radians(phase_deg)
        return voff + vamp * math.sin(2 * math.pi * freq * t + phase_rad)

    if source_type == "PULSE":
        vlow, vhigh, tdelay, trise, tfall, ton, period = params
        if t < tdelay:
            return vlow
        local = (t - tdelay) % period
        return vhigh if local < ton else vlow

    return 0


class VoltageSource(Component):
    def __init__(self, name, n1, n2, source_type, params, index):
        super().__init__(name, (n1, n2))
        self.source_type = source_type
        self.params = params
        self.index = index

    def stamp(self, G, b, ctx):

        n1, n2 = self.nodes
        node_index = ctx["node_index"]
        N = ctx["N"]

        row = N + self.index

        t = ctx.get("t", 0)
        analysis = ctx["analysis"]

        if analysis == "tran":
            value = evaluate_source(self.source_type, self.params, t)
        else:
            # DC/AC → use DC value
            value = self.params[0]

        if n1 != "0":
            G[row, node_index[n1]] = 1
            G[node_index[n1], row] = 1

        if n2 != "0":
            G[row, node_index[n2]] = -1
            G[node_index[n2], row] = -1

        b[row] = value