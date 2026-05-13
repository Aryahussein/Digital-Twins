from .base import Component

class Capacitor(Component):
    """A dynamic capacitor component using generic MNA stamping."""
    
    IS_DYNAMIC = True
    IS_AC_REACTIVE = True
    SHIFT_KEY = "g_eq"
    
    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0))
        self.idx_2 = node_map.get(self.data.get("n2", 0))
        self.prev_current = 0.0

    # ==========================================
    # 1. THE PHYSICS EVALUATOR
    # ==========================================
    def evaluate_physics(self, v_k=None, overrides=None, **kwargs):
        """Pure evaluation of companion models and AC admittance using absolute overrides."""
        overrides = overrides or {}
        
        # Extract simulation state from the kwargs bus
        domain = kwargs.get("domain", "time")
        dt = kwargs.get("dt", 0.0)
        w = kwargs.get("w", 0.0)
        method = kwargs.get("method", "TR")
        v_prev = kwargs.get("v_prev", None)

        # PURE PHYSICS: Use the override if provided, else use the nominal value
        eff_c = overrides.get(self.name, self.value)
        res = {"g_eq": 0.0, "I_eq": 0.0}

        if domain == "frequency":
            res["g_eq"] = 1j * w * eff_c
            return res

        if domain == "time" and dt > 0.0:
            i, j = self.idx_1, self.idx_2
            v1_prev = v_prev[i] if i is not None and v_prev is not None else 0.0
            v2_prev = v_prev[j] if j is not None and v_prev is not None else 0.0
            v_diff_prev = v1_prev - v2_prev

            if method == 'TR':
                g_eq = (2.0 * eff_c) / dt
                # FLIPPED SIGN: Maps perfectly to the unified MNA RHS stamper
                I_eq = -(g_eq * v_diff_prev + self.prev_current)
            else:  # 'BE'
                g_eq = eff_c / dt
                # FLIPPED SIGN: Maps perfectly to the unified MNA RHS stamper
                I_eq = -(g_eq * v_diff_prev)

            res["g_eq"] = g_eq
            res["I_eq"] = I_eq

        return res

    # ==========================================
    # TRANSIENT STATE MANAGEMENT
    # ==========================================
    def update_transient_state(self, v_now, v_prev, dt, method='TR'):
        """Calculates physical current by reusing the companion model parameters."""
        if dt == 0.0:
            return
            
        v1_now = v_now[self.idx_1] if self.idx_1 is not None else 0.0
        v2_now = v_now[self.idx_2] if self.idx_2 is not None else 0.0
        v_now_diff = v1_now - v2_now

        res = self.evaluate_physics(domain="time", dt=dt, v_prev=v_prev, method=method)
        g_eq = res["g_eq"]
        I_eq = res["I_eq"]

        # CHANGED TO PLUS: Because I_eq is mathematically negated in evaluate_physics
        self.prev_current = (g_eq * v_now_diff) + I_eq

    # ==========================================
    # SENSITIVITY ENGINES
    # ==========================================
    def build_adjoint_history(self, J_hist, dt, v_hat_next, method='BE'):
        i, j = self.idx_1, self.idx_2
        v1_hat = v_hat_next[i] if i is not None else 0.0
        v2_hat = v_hat_next[j] if j is not None else 0.0

        I_eq = (self.value / dt) * (v1_hat - v2_hat)
        if i is not None: J_hist[i] += I_eq
        if j is not None: J_hist[j] -= I_eq

    def get_sensitivities(self, VI, PsiPhi, **kwargs):
        """Calculates exact sensitivities w.r.t Capacitance (C)."""
        # DC Check
        if kwargs.get('domain', 'time') == 'time' and kwargs.get('dt', None) is None:
            return {self.name: 0.0}

        v1 = VI[self.idx_1] if self.idx_1 is not None else 0.0
        v2 = VI[self.idx_2] if self.idx_2 is not None else 0.0
        p1 = PsiPhi[self.idx_1] if self.idx_1 is not None else 0.0
        p2 = PsiPhi[self.idx_2] if self.idx_2 is not None else 0.0

        adj_diff = (p1 - p2)

        # NEW, PURE WAY: Tell the physics evaluator to pretend C = 1.0!
        res_unity = self.evaluate_physics(overrides={self.name: 1.0}, **kwargs)
        dY_dC = res_unity["g_eq"]

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
