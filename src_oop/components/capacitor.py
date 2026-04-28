from .base import Component

class Capacitor(Component):
    """A dynamic capacitor component."""
    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0))
        self.idx_2 = node_map.get(self.data.get("n2", 0))
        self.prev_current = 0.0

    def _get_admittance(self, c_val, **kwargs):
        """Centralized helper for domain and integration logic."""
        domain = kwargs.get('domain', 'time')
        
        if domain == "frequency":
            w = kwargs.get('w', 0.0)
            return 1j * w * c_val
        else:
            dt = kwargs.get('dt', 0.0)
            method = kwargs.get('method', 'TR')
            
            if dt == 0: return 0.0
            
            if method == 'TR':
                return (2.0 * c_val) / dt
            else:  # 'BE'
                return c_val / dt

    def stamp_ac(self, Y, sources, w):
        g = self._get_admittance(self.value, domain="frequency", w=w)
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

        # Use the helper for the Matrix term!
        g_eq = self._get_admittance(self.value, domain="time", dt=dt, method=method)
        
        # Calculate the RHS History term
        if method == 'TR':
            I_eq = g_eq * v_diff_prev + self.prev_current
        else:  # BE
            I_eq = g_eq * v_diff_prev

        if i is not None: Y[i, i] += g_eq
        if j is not None: Y[j, j] += g_eq
        if i is not None and j is not None:
            Y[i, j] -= g_eq
            Y[j, i] -= g_eq

        if i is not None: sources[i] += I_eq
        if j is not None: sources[j] -= I_eq

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

    def get_delta_y(self, param_name, dp, **kwargs):
        """Because Y is linear w.r.t C, Delta Y is simply the admittance of dp."""
        return self._get_admittance(dp, **kwargs)

    def get_sensitivities(self, VI, PsiPhi, **kwargs):
        """Calculates sensitivity w.r.t Capacitance (C)."""
        # DC Check: Capacitors are open circuits in DC
        if kwargs.get('domain', 'time') == 'time' and kwargs.get('dt', None) is None:
            return {self.name: 0.0}

        # Voltages and Adjoints
        v1 = VI[self.idx_1] if self.idx_1 is not None else 0.0
        v2 = VI[self.idx_2] if self.idx_2 is not None else 0.0
        p1 = PsiPhi[self.idx_1] if self.idx_1 is not None else 0.0
        p2 = PsiPhi[self.idx_2] if self.idx_2 is not None else 0.0

        adj_diff = (p1 - p2)

        # For Capacitors, dY/dC is exactly the admittance evaluated at C = 1.0!
        dY_dC = self._get_admittance(1.0, **kwargs)

        # Calculate voltage drop (dV)
        domain = kwargs.get('domain', 'time')
        if domain == "frequency":
            dV = v1 - v2
        else:
            V_prev = kwargs.get('V_prev', None)
            v1_prev = V_prev[self.idx_1] if self.idx_1 is not None else 0.0
            v2_prev = V_prev[self.idx_2] if self.idx_2 is not None else 0.0
            dV = (v1 - v2) - (v1_prev - v2_prev)

        dI_dC = dY_dC * dV

        return {self.name: -adj_diff * dI_dC}

    def stamp_PQ(self, P, Q, col_idx):
        """Stamps the Injection (P) and Extraction (Q) topology vectors in-place."""
        i, j = self.idx_1, self.idx_2
        
        if i is not None: 
            P[i, col_idx] = 1.0
            Q[i, col_idx] = 1.0
        if j is not None: 
            P[j, col_idx] = -1.0
            Q[j, col_idx] = -1.0
