from .base import Component

class VCCS(Component):
    """Voltage-Controlled Current Source (Type 'G'). I = G * (V_pos - V_neg)."""

    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0)) # Output +
        self.idx_2 = node_map.get(self.data.get("n2", 0)) # Output -
        self.idx_3 = node_map.get(self.data.get("n3", 0)) # Control +
        self.idx_4 = node_map.get(self.data.get("n4", 0)) # Control -

    # ==========================================
    # 1. STATIC STAMPING (Skeleton Matrix)
    # ==========================================
    def stamp_base_matrix(self, Y):
        """Phase 1: Stamps time/voltage-invariant transconductance into the skeleton matrix."""
        g = self.value # The transconductance
        i, j, k, l = self.idx_1, self.idx_2, self.idx_3, self.idx_4
        
        # Current out of node i depends on (Vk - Vl)
        if i is not None:
            if k is not None: Y[i, k] += g
            if l is not None: Y[i, l] -= g
        
        # Current into node j depends on (Vk - Vl)
        if j is not None:
            if k is not None: Y[j, k] -= g
            if l is not None: Y[j, l] += g

    # ==========================================
    # 2. SENSITIVITY & WOODBURY ENGINES
    # ==========================================
    def get_sensitivities(self, VI, PsiPhi, **kwargs):
        """
        Calculates sensitivity w.r.t the Transconductance (G).
        Formula: -(Psi_out+ - Psi_out-) * (V_ctrl+ - V_ctrl-)
        """
        # 1. Get Forward (Primal) Control Voltage
        v3 = VI[self.idx_3] if self.idx_3 is not None else 0.0
        v4 = VI[self.idx_4] if self.idx_4 is not None else 0.0
        v_control = v3 - v4

        # 2. Get Backward (Adjoint) Output Voltage
        p1 = PsiPhi[self.idx_1] if self.idx_1 is not None else 0.0
        p2 = PsiPhi[self.idx_2] if self.idx_2 is not None else 0.0
        psi_output = p1 - p2

        # 3. Sensitivity is the product of the control voltage and the adjoint output
        # Note the negative sign: it comes from the MNA matrix derivative
        sens_g = -(psi_output * v_control)

        return {self.name: sens_g}

    def stamp_PQ(self, P, Q, col_idx):
        """Stamps the VCCS topology for Woodbury updates. 
        
        Injection (P): Output Nodes. 
        Extraction (Q): Control Nodes.
        """
        out1, out2 = self.idx_1, self.idx_2
        c1, c2 = self.idx_3, self.idx_4 
        
        if out1 is not None: P[out1, col_idx] = 1.0
        if out2 is not None: P[out2, col_idx] = -1.0
        if c1 is not None: Q[c1, col_idx] = 1.0
        if c2 is not None: Q[c2, col_idx] = -1.0

    def get_delta_y(self, param_name, dp, V_nom=None, V_k=None, **kwargs):
        """Transforms a physical parameter change into a scalar Admittance change.
        
        Because transconductance (G) is stamped directly into the Y-matrix as a 
        linear multiplier, Delta Y is exactly equal to the parameter shift dp.
        """
        return dp
