from .base import Component

class VCVS(Component):
    """
    Voltage-Controlled Voltage Source (Type 'E').
    V(n1, n2) = Gain * V(n3, n4)
    """
    REQUIRES_BRANCH_EQ = True
    
    # Base class Woodbury targets:
    SHIFT_KEY = "gain"
    SHIFT_MULTIPLIER = -1.0  # Gain acts inversely in the KVL branch equation

    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0))   # Output +
        self.idx_2 = node_map.get(self.data.get("n2", 0))   # Output -
        self.idx_3 = node_map.get(self.data.get("n3", 0))   # Control +
        self.idx_4 = node_map.get(self.data.get("n4", 0))   # Control -
        self.branch_idx = node_map.get(self.name)
        self.gain = self.value

    # ==========================================
    # 1. THE PHYSICS EVALUATOR
    # ==========================================
    def evaluate_physics(self, v_k=None, overrides=None, **kwargs):
        """Calculates the effective gain using absolute overrides."""
        overrides = overrides or {}
        
        # PURE PHYSICS: Use override if it exists, otherwise use nominal
        eff_gain = overrides.get(self.name, self.gain)
        
        return {"gain": eff_gain}

    # ==========================================
    # 2. STATIC STAMPING (Skeleton Matrix)
    # ==========================================
    def stamp_base_matrix(self, Y):
        """Phase 1: Stamps time/voltage-invariant MNA equations."""
        if self.branch_idx is None: return
        
        b = self.branch_idx
        self._stamp_branch_equation(Y)

        # Ask evaluate_physics for the nominal gain
        res = self.evaluate_physics()
        gain = res.get("gain", 0.0)

        # Control voltage: KVL row gets -Gain at control nodes
        if self.idx_3 is not None:
            Y[b, self.idx_3] -= gain
        if self.idx_4 is not None:
            Y[b, self.idx_4] += gain

    # ==========================================
    # 3. SENSITIVITY & WOODBURY ENGINES
    # ==========================================
    def get_sensitivities(self, VI, PsiPhi, **kwargs):
        """Calculates sensitivity w.r.t Gain."""
        if self.branch_idx is None: return {}

        v3 = VI[self.idx_3] if self.idx_3 is not None else 0.0
        v4 = VI[self.idx_4] if self.idx_4 is not None else 0.0
        v_control = v3 - v4

        psi_branch = PsiPhi[self.branch_idx]

        # Note: Because dY/dGain is -1.0, the adjoint math resolves to positive
        return {self.name: psi_branch * v_control}

    def stamp_PQ(self, P, Q, start_col_idx):
        """Stamps the VCVS topology for Woodbury updates. 
        
        Injection (P): The auxiliary branch equation row.
        Extraction (Q): The differential control voltage (V3 - V4).
        """
        b = self.branch_idx
        c1, c2 = self.idx_3, self.idx_4 
        
        # P: The gain parameter is stamped into the KVL branch equation row
        if b is not None: P[b, start_col_idx] = 1.0
        
        # Q: The state variable multiplying the gain is (V_ctrl+ - V_ctrl-)
        if c1 is not None: Q[c1, start_col_idx] = 1.0
        if c2 is not None: Q[c2, start_col_idx] = -1.0

