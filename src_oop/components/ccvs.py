from .base import Component

class CCVS(Component):
    """
    Current-Controlled Voltage Source (Type 'H').
    V(n1, n2) = Transresistance * I(Vcontrol)
    """
    REQUIRES_BRANCH_EQ = True
    SHIFT_KEY = "rm"
    SHIFT_MULTIPLIER = -1.0

    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0))   # Output +
        self.idx_2 = node_map.get(self.data.get("n2", 0))   # Output -
        self.branch_idx = node_map.get(self.name)
        
        # Find the branch index of the controlling voltage source
        ctrl_name = self.data.get("controlling_source")
        self.ctrl_branch_idx = node_map.get(ctrl_name)
        
        if self.ctrl_branch_idx is None:
            raise ValueError(
                f"CCVS '{self.name}': controlling source '{ctrl_name}' "
                f"not found in circuit. It must be a voltage source."
            )

    # ==========================================
    # 1. THE PHYSICS EVALUATOR
    # ==========================================
    def evaluate_physics(self, v_k=None, overrides=None, **kwargs):
        """Calculates the effective transresistance (H) using absolute overrides."""
        overrides = overrides or {}
        
        # PURE PHYSICS: Use override if it exists, otherwise use nominal value
        eff_rm = overrides.get(self.name, self.value)
        
        return {"rm": eff_rm}

    # ==========================================
    # 2. STATIC STAMPING (Skeleton Matrix)
    # ==========================================
    def stamp_base_matrix(self, Y):
        """Phase 1: Stamps time/voltage-invariant transresistance into the skeleton matrix."""
        if self.branch_idx is None: return
        
        b = self.branch_idx
        self._stamp_branch_equation(Y)

        # Ask evaluate_physics for the nominal transresistance
        res = self.evaluate_physics()
        rm = res.get("rm", 0.0)

        # KVL row: -H at controlling branch column
        if self.ctrl_branch_idx is not None:
            Y[b, self.ctrl_branch_idx] -= rm

    # ==========================================
    # 3. SENSITIVITY & WOODBURY ENGINES
    # ==========================================
    def get_sensitivities(self, VI, PsiPhi, **kwargs):
        """Calculates exact sensitivity w.r.t Transresistance (H)."""
        if self.branch_idx is None or self.ctrl_branch_idx is None: 
            return {}

        # Forward: controlling branch current
        i_ctrl = VI[self.ctrl_branch_idx]

        # Adjoint: branch variable for this voltage source
        psi_branch = PsiPhi[self.branch_idx]

        # Note: Because dY/dH is -1.0, the adjoint math resolves to positive (psi_branch * i_ctrl)
        return {self.name: psi_branch * i_ctrl}

    def stamp_PQ(self, P, Q, start_col_idx):
        """Stamps the CCVS topology for Woodbury updates. 
        
        Injection (P): The auxiliary branch equation row (KVL).
        Extraction (Q): The controlling branch current.
        """
        b = self.branch_idx
        b_ctrl = self.ctrl_branch_idx 
        
        # P: Where does the parameter change occur in the rows? 
        # Inside the KVL branch equation row!
        if b is not None: P[b, start_col_idx] = 1.0
        
        # Q: What state variable multiplies against this parameter?
        # The current of the controlling branch!
        if b_ctrl is not None: Q[b_ctrl, start_col_idx] = 1.0

