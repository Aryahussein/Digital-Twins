from .base import Component

class Inductor(Component):
    """A linear inductor (Type 'L') using generic MNA branch stamping."""

    IS_DYNAMIC = True
    IS_AC_REACTIVE = True
    REQUIRES_BRANCH_EQ = True
    
    # Base class Woodbury targets:
    SHIFT_KEY = "req"
    SHIFT_MULTIPLIER = -1.0 # Inductor stamps -Z, so matrix shifts are inverted!

    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0))
        self.idx_2 = node_map.get(self.data.get("n2", 0))
        self.branch_idx = node_map.get(self.name) # Inductors get a branch index

    # ==========================================
    # 1. STATIC STAMPING (Skeleton Matrix)
    # ==========================================
    def stamp_base_matrix(self, Y):
        """Phase 1: Stamps the +1/-1 topology for the branch current."""
        self._stamp_branch_equation(Y)

    # ==========================================
    # 2. THE PHYSICS EVALUATOR
    # ==========================================
    def evaluate_physics(self, v_k=None, overrides=None, **kwargs):
        """Pure evaluation of companion models and AC impedance using absolute overrides.
        
        Because the Inductor uses a branch equation: V_L - Z * I_L = 0,
        this function calculates the equivalent Z (req) and history voltage (v_eq).
        """
        overrides = overrides or {}
        
        # Extract simulation state
        domain = kwargs.get("domain", "time")
        dt = kwargs.get("dt", 0.0)
        w = kwargs.get("w", 0.0)
        method = kwargs.get("method", "TR")
        v_prev = kwargs.get("v_prev", None)

        # PURE PHYSICS: Use override if provided, else use nominal value
        eff_l = overrides.get(self.name, self.value)
        res = {"req": 0.0, "v_eq": 0.0}

        if domain == "frequency":
            res["req"] = 1j * w * eff_l
            return res

        if domain == "time" and dt > 0.0:
            b = self.branch_idx
            i_prev = v_prev[b] if v_prev is not None and b is not None else 0.0

            if method == 'TR':
                req = (2.0 * eff_l) / dt
                i = self.idx_1
                j = self.idx_2
                
                v1_prev = v_prev[i] if v_prev is not None and i is not None else 0.0
                v2_prev = v_prev[j] if v_prev is not None and j is not None else 0.0
                v_diff_prev = v1_prev - v2_prev
                
                v_eq = req * i_prev + v_diff_prev
            else:  # 'BE'
                req = eff_l / dt
                v_eq = req * i_prev

            res["req"] = req
            res["v_eq"] = v_eq

        return res

    # ==========================================
    # 3. THE STAMPERS (Custom Branch Interface)
    # ==========================================
    def stamp_matrix(self, Y, res, *args):
        """Stamps equivalent resistance (Transient) or complex impedance (AC) into the Jacobian.
        
        Overrides base class because Inductors stamp exclusively into their branch row/col.
        """
        req = res.get("req", 0.0)
        if req == 0.0 or self.branch_idx is None: return
        
        # Stamps -Z into the [branch, branch] diagonal
        Y[self.branch_idx, self.branch_idx] -= req

    def stamp_rhs(self, J, res, *args):
        """Stamps equivalent history voltage into the Residual vector."""
        v_eq = res.get("v_eq", 0.0)
        if v_eq == 0.0 or self.branch_idx is None: return
        
        # Stamps -V_eq into the branch equation row
        J[self.branch_idx] -= v_eq


    # ==========================================
    # SENSITIVITY ENGINES
    # ==========================================
    def build_adjoint_history(self, J_hist, dt, v_hat_next, method='BE'):
        """Builds the adjoint RHS memory term for Backward Euler."""
        if self.branch_idx is not None:
            i_L_hat = v_hat_next[self.branch_idx]
            V_eq_hat = (self.value / dt) * i_L_hat
            J_hist[self.branch_idx] -= V_eq_hat

    def get_sensitivities(self, VI, PsiPhi, **kwargs):
        """Calculates sensitivity w.r.t Inductance (L)."""
        if self.branch_idx is None: return {}

        # DC Analysis (Inductor is a short, L has no effect on DC bias)
        if kwargs.get('domain', 'time') == 'time' and kwargs.get('dt', None) is None:
            return {self.name: 0.0}

        i_L = VI[self.branch_idx]
        psi_branch = PsiPhi[self.branch_idx]

        # NEW, PURE WAY: Tell physics evaluator to pretend L = 1.0
        res_unity = self.evaluate_physics(overrides={self.name: 1.0}, **kwargs)
        dZ_dL = res_unity.get("req", 0.0)

        domain = kwargs.get('domain', 'time')
        if domain == "frequency":
            dVL_dL = dZ_dL * i_L
            return {self.name: psi_branch * dVL_dL}
        else:
            V_prev = kwargs.get('V_prev', None)
            i_L_prev = V_prev[self.branch_idx] if V_prev is not None else 0.0
            dI = i_L - i_L_prev
            
            dI_dt = dZ_dL * dI
            return {self.name: psi_branch * dI_dt}

    def stamp_PQ(self, P, Q, start_col_idx):
        """Stamps the Injection (P) and Extraction (Q) topology vectors in-place."""
        b = self.branch_idx
        
        # P and Q both target the branch equation row/col, NOT the terminal nodes!
        if b is not None: 
            P[b, start_col_idx] = 1.0
            Q[b, start_col_idx] = 1.0

