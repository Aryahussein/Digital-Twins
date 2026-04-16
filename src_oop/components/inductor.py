from .base import Component

class Inductor(Component):
    """A linear inductor (Type 'L'). Requires an MNA branch equation."""

    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0))
        self.idx_2 = node_map.get(self.data.get("n2", 0))
        self.branch_idx = node_map.get(self.name) # Inductors get a branch index

    def stamp_mna_connection(self, Y):
        """Stamps the +1/-1 topology for the branch current."""
        self._stamp_branch_equation(Y)

    def stamp_dc(self, Y, sources):
        """In DC, an inductor is a short circuit (V1 - V2 = 0)."""
        # The _stamp_branch_equation already handled the Y matrix.
        # We just ensure the RHS is 0.
        sources[self.branch_idx] = 0.0

    def stamp_ac(self, Y, sources, w):
        """Stamps complex impedance: Z = j * w * L."""
        z = 1j * w * self.value
        if self.branch_idx is not None:
            Y[self.branch_idx, self.branch_idx] -= z

    def stamp_transient(self, Y, sources, t, dt, v_prev, method = 'TR'):
        if self.branch_idx is None: return
        i_prev = v_prev[self.branch_idx]

        if method == 'TR':
            req = 2.0 * self.value / dt
            v1_prev = v_prev[self.idx_1] if self.idx_1 is not None else 0.0
            v2_prev = v_prev[self.idx_2] if self.idx_2 is not None else 0.0
            v_diff_prev = v1_prev - v2_prev
            v_eq = req * i_prev + v_diff_prev
        else:  # BE
            req = self.value / dt
            v_eq = req * i_prev
        
        Y[self.branch_idx, self.branch_idx] -= req
        sources[self.branch_idx] -= v_eq

    def build_adjoint_history(self, J_hist, dt, v_hat_next, method='BE'):
        """Builds the adjoint RHS memory term for Backward Euler."""
        if self.branch_idx is not None:
            # The adjoint inductor memory depends on the adjoint branch current 
            # from the 'next' time step (which we already solved in backward time)
            i_L_hat = v_hat_next[self.branch_idx]
            
            # Adjoint Equivalent Voltage Source: V_eq_hat = (L/dt) * I_L_hat
            V_eq_hat = (self.value / dt) * i_L_hat
            
            # Subtract from the branch equation row in the RHS history
            J_hist[self.branch_idx] -= V_eq_hat

    def get_sensitivities(self, VI, PsiPhi, w=0.0, dt=None, V_prev=None, method='TR'):
        """Calculates sensitivity w.r.t Inductance (L)."""
        if self.branch_idx is None: return {}

        # The current flowing through the inductor is stored at the branch index
        i_L = VI[self.branch_idx]
        
        # Adjoint branch variable
        psi_branch = PsiPhi[self.branch_idx]

        # 1. AC Analysis (V_L = j * w * L * I_L)
        if w != 0.0:
            dVL_dL = 1j * w * i_L
            return {self.name: psi_branch * dVL_dL}

        # 2. Transient Analysis
        elif dt is not None and V_prev is not None:
            i_L_prev = V_prev[self.branch_idx]
            
            # Current change over this time step
            dI = i_L - i_L_prev
            
            # dVL/dL varies based on the integration scheme used
            if method == 'TR':
                dI_dt = (2.0 * dI) / dt
            else:  # 'BE' (Backward Euler)
                dI_dt = dI / dt
                
            return {self.name: psi_branch * dI_dt}

        # 3. DC Analysis (Inductor is a short, L has no effect on DC bias)
        else:
            return {self.name: 0.0}
