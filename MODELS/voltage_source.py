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
        if t < tstart: return v1
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
        if local < trise:               return vlow  + (vhigh - vlow) * local / trise
        if local < trise + ton:         return vhigh
        if local < trise + ton + tfall: return vhigh - (vhigh - vlow) * (local - trise - ton) / tfall
        return vlow

    return 0.0


class VoltageSource(Component):
    def __init__(self, name, n1, n2, source_type, params, index,
                 ac_mag=None, ac_phase=0.0):
        """
        Parameters
        ----------
        source_type : str
            Primary waveform for DC/tran: "DC", "SINE", "PULSE", etc.
            Also "AC" for pure AC-only sources (no DC bias).
        params : list
            Parameters for the primary waveform.
        index : int
            MNA branch-current index.
        ac_mag : float or None
            Small-signal AC magnitude from the "AC <mag>" token.
            None means the source has no AC excitation (e.g. Vdd supply).
        ac_phase : float
            Small-signal AC phase in degrees.  Defaults to 0.
        """
        super().__init__(name, (n1, n2))
        self.source_type = source_type
        self.params      = params
        self.index       = index

        # For a pure "AC" source type (no DC keyword), read mag/phase from
        # params if the caller didn't pass them explicitly via kwargs.
        if ac_mag is None and source_type == "AC":
            self.ac_mag   = float(params[0])
            self.ac_phase = float(params[1]) if len(params) > 1 else 0.0
        else:
            self.ac_mag   = float(ac_mag) if ac_mag is not None else None
            self.ac_phase = float(ac_phase)

    def stamp(self, G, b, ctx):

        n1, n2     = self.nodes
        node_index = ctx["node_index"]
        N          = ctx["N"]
        row        = N + self.index
        analysis   = ctx["analysis"]

        sweep_overrides = ctx.get("sweep_overrides", {})

        # ── AC analysis ───────────────────────────────────────────────
        # Always driven by ac_mag / ac_phase, regardless of source_type.
        # This correctly handles mixed declarations such as:
        #   Vinp 1 0 DC 0.44 AC 0.5        →  0.5 ∠  0°
        #   Vinn 6 0 DC 0.44 AC 0.5 180    →  0.5 ∠180°
        #   Vdd  3 0 1.2                    →  0 V (no AC spec → ac_mag=None)
        if analysis == "ac":
            if self.ac_mag is None:
                value = 0.0          # DC supply: zero small-signal contribution
            else:
                phase_rad = math.radians(self.ac_phase)
                value = self.ac_mag * (math.cos(phase_rad) + 1j * math.sin(phase_rad))

        # ── DC sweep override ─────────────────────────────────────────
        elif self.name.lower() in sweep_overrides:
            value = sweep_overrides[self.name.lower()]

        # ── Transient ─────────────────────────────────────────────────
        elif analysis == "tran":
            value = evaluate_source(self.source_type, self.params, ctx.get("t", 0))

        # ── DC operating point ────────────────────────────────────────
        else:
            value = self.params[0]

        # ── MNA stamping ──────────────────────────────────────────────
        if n1 != "0":
            G[row, node_index[n1]] = 1
            G[node_index[n1], row] = 1

        if n2 != "0":
            G[row, node_index[n2]] = -1
            G[node_index[n2], row] = -1

        b[row] = value