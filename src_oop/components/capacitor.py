from .base import Component

class Capacitor(Component):
    """A dynamic capacitor component."""
    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0))
        self.idx_2 = node_map.get(self.data.get("n2", 0))
        self.prev_current = 0.0

    def stamp_ac(self, Y, sources, w):
        """EXTENSION ADDED: Stamps AC impedance."""
        g = 1j * w * self.value
        i, j = self.idx_1, self.idx_2
        if i is not None:
            Y[i, i] += g
            if j is not None:
                Y[i, j] -= g
                Y[j, i] -= g
        if j is not None:
            Y[j, j] += g

    def stamp_transient(self, Y, sources, t, dt, v_prev, method='TR'):
        i, j = self.idx_1, self.idx_2

        v1_prev = v_prev[i] if i is not None else 0.0
        v2_prev = v_prev[j] if j is not None else 0.0
        v_diff_prev = v1_prev - v2_prev

        if method == 'TR':
            g_eq = 2.0 * self.value / dt
            I_eq = g_eq * v_diff_prev + self.prev_current
        else:  # BE
            g_eq = self.value / dt
            I_eq = g_eq * v_diff_prev

        if i is not None: Y[i, i] += g_eq
        if j is not None: Y[j, j] += g_eq
        if i is not None and j is not None:
            Y[i, j] -= g_eq
            Y[j, i] -= g_eq

        if i is not None: sources[i] += I_eq
        if j is not None: sources[j] -= I_eq

        # Store for update_transient_state
        self._last_g_eq = g_eq
        self._last_I_eq = I_eq

    def update_transient_state(self, v_now, method='TR'):
        if method != 'TR':
            return
        i, j = self.idx_1, self.idx_2
        v1 = v_now[i] if i is not None else 0.0
        v2 = v_now[j] if j is not None else 0.0
        v_diff = v1 - v2
        self.prev_current = self._last_g_eq * v_diff - self._last_I_eq

    def build_adjoint_history(self, J_hist, dt, v_hat_next, method='BE'):
        i, j = self.idx_1, self.idx_2
        v1_hat = v_hat_next[i] if i is not None else 0.0
        v2_hat = v_hat_next[j] if j is not None else 0.0

        I_eq = (self.value / dt) * (v1_hat - v2_hat)
        if i is not None: J_hist[i] += I_eq
        if j is not None: J_hist[j] -= I_eq

    def get_sensitivities(self, VI, PsiPhi, w=0.0, dt=None, V_prev=None, method='TR'):
        """Calculates sensitivity w.r.t Capacitance (C)."""
        v1 = VI[self.idx_1] if self.idx_1 is not None else 0.0
        v2 = VI[self.idx_2] if self.idx_2 is not None else 0.0
        
        p1 = PsiPhi[self.idx_1] if self.idx_1 is not None else 0.0
        p2 = PsiPhi[self.idx_2] if self.idx_2 is not None else 0.0

        # Adjoint potential difference: (Psi_1 - Psi_2)
        adj_diff = (p1 - p2)

        # 1. AC Analysis
        if w != 0.0:
            # dI/dC = j * w * V_diff
            dI_dC = 1j * w * (v1 - v2)
            return {self.name: -adj_diff * dI_dC}

        # 2. Transient Analysis
        elif dt is not None and V_prev is not None:
            v1_prev = V_prev[self.idx_1] if self.idx_1 is not None else 0.0
            v2_prev = V_prev[self.idx_2] if self.idx_2 is not None else 0.0
            
            # Voltage change over this time step
            dV = (v1 - v2) - (v1_prev - v2_prev)
            
            # dI/dC varies based on the integration scheme used
            if method == 'TR':
                dI_dC = (2.0 * dV) / dt
            else:  # 'BE' (Backward Euler)
                dI_dC = dV / dt
                
            return {self.name: -adj_diff * dI_dC}

        # 3. DC Analysis
        else:
            # Capacitors are open circuits in DC; changing C has no effect
            return {self.name: 0.0}
