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
        if t < tstart:    return v1
        if t > tstart + trise: return v2
        return v1 + (v2 - v1) / trise * (t - tstart)

    if source_type == "SINE":
        voff  = params[0]
        vamp  = params[1]
        freq  = params[2]
        phase = math.radians(params[3]) if len(params) > 3 else 0.0
        return voff + vamp * math.sin(2 * math.pi * freq * t + phase)

    if source_type == "PULSE":
        vlow, vhigh, tdelay, trise, tfall, ton, period = params
        if t < tdelay: return vlow
        local = (t - tdelay) % period
        if local < trise:              return vlow  + (vhigh - vlow) * local / trise
        if local < trise + ton:        return vhigh
        if local < trise + ton + tfall:return vhigh - (vhigh - vlow) * (local - trise - ton) / tfall
        return vlow

    return 0.0


class VoltageSource(Component):
    def __init__(self, name, n1, n2, source_type, params, index):
        super().__init__(name, (n1, n2))
        self.source_type = source_type
        self.params      = params
        self.index       = index

    def stamp(self, G, b, ctx):

        n1, n2     = self.nodes
        node_index = ctx["node_index"]
        N          = ctx["N"]
        row        = N + self.index
        analysis   = ctx["analysis"]

        # DC sweep: this source may be overridden at this sweep point
        sweep_overrides = ctx.get("sweep_overrides", {})
        if self.name.lower() in sweep_overrides:
            value = sweep_overrides[self.name.lower()]
        elif analysis == "tran":
            value = evaluate_source(self.source_type, self.params, ctx.get("t", 0))
        else:
            value = self.params[0]

        # NOTE: source_scale is NOT applied here.
        # dc_solver._stamp_all owns all source-stepping scaling
        # by directly multiplying b[row] after all stamps are done.
        # Applying scale here would cause double-scaling.

        if n1 != "0":
            G[row, node_index[n1]] = 1
            G[node_index[n1], row] = 1

        if n2 != "0":
            G[row, node_index[n2]] = -1
            G[node_index[n2], row] = -1

        b[row] = value