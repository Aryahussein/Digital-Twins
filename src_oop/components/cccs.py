from .base import Component

class CCCS(Component):
    """
    Current-Controlled Current Source (Type 'F').
    I(n1, n2) = Gain * I(Vcontrol)
    """

    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0))   # Output +
        self.idx_2 = node_map.get(self.data.get("n2", 0))   # Output -
        self.gain = self.value
        
        # Find the branch index of the controlling voltage source
        ctrl_name = self.data.get("controlling_source")
        self.ctrl_branch_idx = node_map.get(ctrl_name)
        
        if self.ctrl_branch_idx is None:
            raise ValueError(
                f"CCCS '{self.name}': controlling source '{ctrl_name}' "
                f"not found in circuit. It must be an independent/dependent voltage source."
            )

    # ==========================================
    # 1. STATIC STAMPING (Skeleton Matrix)
    # ==========================================
    def stamp_base_matrix(self, Y):
        """Phase 1: Stamps time/voltage-invariant gain into the skeleton matrix."""
        b = self.ctrl_branch_idx
        g = self.gain

        # Stamp gain at output node rows, controlling branch column
        if self.idx_1 is not None:
            Y[self.idx_1, b] += g
        if self.idx_2 is not None:
            Y[self.idx_2, b] -= g

    # ==========================================
    # 2. SENSITIVITY & WOODBURY ENGINES
    # ==========================================
    def get_sensitivities(self, VI, PsiPhi, **kwargs):
        """Calculates exact sensitivity w.r.t Gain (alpha)."""
        if self.ctrl_branch_idx is None: return {}

        # Forward: controlling branch current
        i_ctrl = VI[self.ctrl_branch_idx]

        # Adjoint: output node difference
        p1 = PsiPhi[self.idx_1] if self.idx_1 is not None else 0.0
        p2 = PsiPhi[self.idx_2] if self.idx_2 is not None else 0.0
        psi_output = p1 - p2

        return {self.name: -(psi_output * i_ctrl)}

    def stamp_PQ(self, P, Q, col_idx):
        """Stamps the CCCS Woodbury topology. 
        
        Current Injection (P): Output nodes.
        State Extraction (Q): Controlling branch current.
        """
        out1, out2 = self.idx_1, self.idx_2
        b_ctrl = self.ctrl_branch_idx 
        
        # P: Where does the current get injected?
        if out1 is not None: P[out1, col_idx] = 1.0
        if out2 is not None: P[out2, col_idx] = -1.0
        
        # Q: What is the state variable controlling the source?
        # For a CCCS, it's the absolute current flowing through the controlling branch!
        if b_ctrl is not None: Q[b_ctrl, col_idx] = 1.0

    def get_delta_y(self, param_name, dp, **kwargs):
        """Transforms a physical parameter change into a scalar Admittance change.
        
        For a CCCS, the parameter is the Gain. Because the gain is stamped 
        directly as a linear multiplier in the Y-matrix, Delta Y is exactly dp.
        """
        return dp
