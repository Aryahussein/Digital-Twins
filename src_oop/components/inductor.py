from .base import Component

class Inductor(Component):
    """A linear inductor (Type 'L'). Requires an MNA branch equation."""

    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0))
        self.idx_2 = node_map.get(self.data.get("n2", 0))
        self.branch_idx = node_map.get(self.name) # Inductors get a branch index

    # ==========================================
    # THE PHYSICS HELPER
    # ==========================================
    def _get_impedance(self, l_val, **kwargs):
        """Centralized helper for domain and integration impedance logic.
        
        Because the Inductor uses a branch equation: V_L - Z * I_L = 0,
        this function calculates the equivalent Z.
        """
        domain = kwargs.get('domain', 'time')
        
        if domain == "frequency":
            w = kwargs.get('w', 0.0)
            return 1j * w * l_val
        else:
            dt = kwargs.get('dt', 0.0)
            method = kwargs.get('method', 'TR')
            
            if dt == 0: return 0.0
            
            if method == 'TR':
                return (2.0 * l_val) / dt
            else:  # 'BE'
                return l_val / dt

    # ==========================================
    # SOLVER ENGINES (Stamping)
    # ==========================================
    def stamp_mna_connection(self, Y):
        """Stamps the +1/-1 topology for the branch current."""
        self._stamp_branch_equation(Y)

    def stamp_dc(self, Y, sources):
        """In DC, an inductor is a short circuit (V1 - V2 = 0)."""
        sources[self.branch_idx] = 0.0

    def stamp_ac(self, Y, sources, w):
        """Stamps complex impedance: Z = j * w * L."""
        z = self._get_impedance(self.value, domain="frequency", w=w)
        if self.branch_idx is not None:
            Y[self.branch_idx, self.branch_idx] -= z

    def stamp_transient(self, Y, sources, t, dt, v_prev, method='TR'):
        if self.branch_idx is None: return
        i_prev = v_prev[self.branch_idx]

        # Use the helper for the Matrix term!
        req = self._get_impedance(self.value, domain="time", dt=dt, method=method)

        # Calculate RHS History term
        if method == 'TR':
            v1_prev = v_prev[self.idx_1] if self.idx_1 is not None else 0.0
            v2_prev = v_prev[self.idx_2] if self.idx_2 is not None else 0.0
            v_diff_prev = v1_prev - v2_prev
            v_eq = req * i_prev + v_diff_prev
        else:  # BE
            v_eq = req * i_prev
        
        Y[self.branch_idx, self.branch_idx] -= req
        sources[self.branch_idx] -= v_eq

    def build_adjoint_history(self, J_hist, dt, v_hat_next, method='BE'):
        """Builds the adjoint RHS memory term for Backward Euler."""
        if self.branch_idx is not None:
            i_L_hat = v_hat_next[self.branch_idx]
            V_eq_hat = (self.value / dt) * i_L_hat
            J_hist[self.branch_idx] -= V_eq_hat

    # ==========================================
    # SENSITIVITY ENGINES (Adjoint & Woodbury)
    # ==========================================
    def get_sensitivities(self, VI, PsiPhi, **kwargs):
        """Calculates sensitivity w.r.t Inductance (L)."""
        if self.branch_idx is None: return {}

        # DC Analysis (Inductor is a short, L has no effect on DC bias)
        if kwargs.get('domain', 'time') == 'time' and kwargs.get('dt', None) is None:
            return {self.name: 0.0}

        i_L = VI[self.branch_idx]
        psi_branch = PsiPhi[self.branch_idx]

        # dZ/dL evaluated exactly at L = 1.0!
        dZ_dL = self._get_impedance(1.0, **kwargs)

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

    def stamp_PQ(self, P, Q, col_idx):
        """Stamps the Injection (P) and Extraction (Q) topology vectors in-place."""
        b = self.branch_idx
        
        # P and Q both target the branch equation row/col, NOT the terminal nodes!
        if b is not None: 
            P[b, col_idx] = 1.0
            Q[b, col_idx] = 1.0

    def get_delta_y(self, param_name, dp, **kwargs):
        """Transforms a physical parameter change into a scalar Admittance change.
        
        Because the Inductor stamps -Z into the Y-matrix at [branch, branch],
        a change of dp in Inductance results in a change of -dZ in the matrix.
        """
        dz = self._get_impedance(dp, **kwargs)
        return -dz
