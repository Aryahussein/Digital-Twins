from .base import Component

class VCVS(Component):
    """
    Voltage-Controlled Voltage Source (Type 'E').
    V(n1, n2) = Gain * V(n3, n4)
    """

    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0))   # Output +
        self.idx_2 = node_map.get(self.data.get("n2", 0))   # Output -
        self.idx_3 = node_map.get(self.data.get("n3", 0))   # Control +
        self.idx_4 = node_map.get(self.data.get("n4", 0))   # Control -
        self.branch_idx = node_map.get(self.name)
        self.gain = self.value

    # ==========================================
    # SOLVER ENGINES (Stamping)
    # ==========================================
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

    # ==========================================
    # SENSITIVITY ENGINES (Adjoint & Woodbury)
    # ==========================================
    def get_sensitivities(self, VI, PsiPhi, **kwargs):
        """Calculates sensitivity w.r.t Gain."""
        if self.branch_idx is None: return {}

        v3 = VI[self.idx_3] if self.idx_3 is not None else 0.0
        v4 = VI[self.idx_4] if self.idx_4 is not None else 0.0
        v_control = v3 - v4

        psi_branch = PsiPhi[self.branch_idx]

        return {self.name: psi_branch * v_control}

    def stamp_PQ(self, P, Q, col_idx):
        """Stamps the VCVS topology for Woodbury updates. 
        
        Injection (P): The auxiliary branch equation row.
        Extraction (Q): The differential control voltage (V3 - V4).
        """
        b = self.branch_idx
        c1, c2 = self.idx_3, self.idx_4 
        
        # P: The gain parameter is stamped into the KVL branch equation row
        if b is not None: P[b, col_idx] = 1.0
        
        # Q: The state variable multiplying the gain is (V_ctrl+ - V_ctrl-)
        if c1 is not None: Q[c1, col_idx] = 1.0
        if c2 is not None: Q[c2, col_idx] = -1.0

    def get_delta_y(self, param_name, dp, **kwargs):
        """Transforms a physical parameter change into a scalar Admittance change.
        
        Because the KVL equation is V_out - Gain * (V_ctrl) = 0, the Gain 
        is stamped as a negative multiplier. Therefore, if the gain shifts 
        by dp, the matrix shifts by -dp.
        """
        return -dp
