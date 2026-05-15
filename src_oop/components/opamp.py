from .base import Component

class OpAmp(Component):
    """
    Ideal Op-Amp (Type 'E'). 
    Implemented as a Voltage-Controlled Voltage Source (VCVS).
    V(out, gnd) = Gain * (V(n_plus) - V(n_minus))
    """

    REQUIRES_BRANCH_EQ = True
    
    # Base class Woodbury targets:
    SHIFT_KEY = "gain"
    SHIFT_MULTIPLIER = -1.0  # Gain acts inversely in the KVL branch equation

    def bind_nodes(self, node_map):
        # Input terminals
        self.idx_p = node_map.get(self.data.get("n1", 0)) # Non-inverting (+)
        self.idx_m = node_map.get(self.data.get("n2", 0)) # Inverting (-)
        
        # Output terminal
        self.idx_out = node_map.get(self.data.get("n_out", 0))
        
        # Branch index for the VCVS equation
        self.branch_idx = node_map.get(self.name)
        
        # High open-loop gain (default 100k if not specified)
        self.gain = self.data.get("value", 1e5)

    # ==========================================
    # 1. THE PHYSICS EVALUATOR
    # ==========================================
    def evaluate_physics(self, v_k=None, overrides=None, **kwargs):
        """Calculates the effective open-loop gain using absolute overrides."""
        overrides = overrides or {}
        
        # PURE PHYSICS: Use override if it exists, otherwise use nominal
        eff_gain = overrides.get(self.name, self.gain)
        
        return {"gain": eff_gain}

    # ==========================================
    # 2. STATIC STAMPING (Skeleton Matrix)
    # ==========================================
    def stamp_base_matrix(self, Y):
        """
        Phase 1: Stamps the time/voltage-invariant VCVS MNA equations.
        Equation: V_out - Gain*(V_p - V_m) = 0
        """
        b = self.branch_idx
        if b is None: return

        # 1. Output terminal current assignment (KVL / KCL intersection)
        # Current flows OUT of idx_out, through the branch, to ground.
        if self.idx_out is not None:
            Y[self.idx_out, b] += 1.0  # Branch current leaves the output node
            Y[b, self.idx_out] += 1.0  # V_out term in the branch equation

        # Ask evaluate_physics for the nominal gain
        res = self.evaluate_physics()
        gain = res.get("gain", 0.0)

        # 2. Control voltage dependencies in the branch row
        if self.idx_p is not None:
            Y[b, self.idx_p] -= gain
        if self.idx_m is not None:
            Y[b, self.idx_m] += gain

    # ==========================================
    # 3. SENSITIVITY & WOODBURY ENGINES
    # ==========================================
    def get_sensitivities(self, VI, PsiPhi, **kwargs):
        """Calculates exact sensitivity w.r.t Open-Loop Gain (A)."""
        b = self.branch_idx
        if b is None: return {}

        # Forward differential input
        vp = VI[self.idx_p] if self.idx_p is not None else 0.0
        vm = VI[self.idx_m] if self.idx_m is not None else 0.0
        v_diff = vp - vm

        # Adjoint branch variable
        psi_branch = PsiPhi[b]

        # Note: Because dY/dGain is -1.0, the adjoint math resolves to positive
        return {self.name: psi_branch * v_diff}

    def stamp_PQ(self, P, Q, start_col_idx):
        """Stamps the VCVS topology for Woodbury updates.
        
        Injection (P): The auxiliary branch equation row.
        Extraction (Q): The differential input voltage (V_plus - V_minus).
        """
        b = self.branch_idx
        p, m = self.idx_p, self.idx_m
        
        # P: The gain parameter lives entirely inside the KVL branch equation row
        if b is not None: 
            P[b, start_col_idx] = 1.0
            
        # Q: The state variable multiplying the gain is (V_plus - V_minus)
        if p is not None: Q[p, start_col_idx] = 1.0
        if m is not None: Q[m, start_col_idx] = -1.0

