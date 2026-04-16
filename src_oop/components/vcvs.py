from .base import Component

class VCVS(Component):
    """
    GVoltage-Controlled Voltage Source (Type 'E').
    V(n1, n2) = Gain * V(n3, n4)
    """

    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0))   # Output +
        self.idx_2 = node_map.get(self.data.get("n2", 0))   # Output -
        self.idx_3 = node_map.get(self.data.get("n3", 0))   # Control +
        self.idx_4 = node_map.get(self.data.get("n4", 0))   # Control -
        self.branch_idx = node_map.get(self.name)
        self.gain = self.value

    def stamp_mna_connection(self, Y):
        if self.branch_idx is None: return
        
        b = self.branch_idx

        # Output topology: branch current enters n1, leaves n2
        if self.idx_1 is not None:
            Y[self.idx_1, b] += 1.0
            Y[b, self.idx_1] += 1.0
        if self.idx_2 is not None:
            Y[self.idx_2, b] -= 1.0
            Y[b, self.idx_2] -= 1.0

        # Control voltage: KVL row gets -Gain at control nodes
        if self.idx_3 is not None:
            Y[b, self.idx_3] -= self.gain
        if self.idx_4 is not None:
            Y[b, self.idx_4] += self.gain

    def get_sensitivities(self, VI, PsiPhi, **kwargs):
        """Calculates sensitivity w.r.t Gain."""
        if self.branch_idx is None: return {}

        v3 = VI[self.idx_3] if self.idx_3 is not None else 0.0
        v4 = VI[self.idx_4] if self.idx_4 is not None else 0.0
        v_control = v3 - v4

        psi_branch = PsiPhi[self.branch_idx]

        return {self.name: psi_branch * v_control}
