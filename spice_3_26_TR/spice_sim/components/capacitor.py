"""Capacitor Component Module."""

from .base import Component


class Capacitor(Component):
    """A dynamic capacitor component (Type 'C').
    
    In DC: open circuit (no stamp).
    In AC: stamps admittance Y = j*w*C.
    In TRAN (BE): Backward Euler companion — G_eq = C/Δt.
    In TRAN (TR): Trapezoidal companion — G_eq = 2C/Δt, requires stored i_prev.
    
    The TR Norton companion model (Lecture 4 slide 15):
        G_eq = 2C/Δt
        I_eq = i(t) + (2C/Δt)·v(t)
        After solving: i(t+Δt) = (2C/Δt)·(v(t+Δt) − v(t)) − i(t)
    """

    def __init__(self, name, data_dict):
        super().__init__(name, data_dict)
        self._i_cap_prev = 0.0  # TR state: stored capacitor current

    def _validate(self):
        if self.value < 0.0:
            raise ValueError(
                f"Capacitor '{self.name}' has negative capacitance ({self.value}). "
                "Negative capacitance is not physically meaningful."
            )
        if self.value == 0.0:
            import warnings
            warnings.warn(f"Capacitor '{self.name}' has zero capacitance — it will have no effect.")

    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0))
        self.idx_2 = node_map.get(self.data.get("n2", 0))

    def reset_transient_state(self):
        """Clear TR history for a fresh simulation run."""
        self._i_cap_prev = 0.0

    def stamp_ac(self, Y, sources, w):
        """Stamps AC admittance: Y = j*w*C."""
        g = 1j * w * self.value
        i, j = self.idx_1, self.idx_2
        if i is not None:
            Y[i, i] += g
            if j is not None:
                Y[i, j] -= g
                Y[j, i] -= g
        if j is not None:
            Y[j, j] += g

    def stamp_transient(self, Y, sources, t, dt, v_prev, method='BE'):
        """Stamps the capacitor companion model into the MNA system.
        
        BE:  G_eq = C/Δt,    I_eq = (C/Δt)·v_prev
        TR:  G_eq = 2C/Δt,   I_eq = i_prev + (2C/Δt)·v_prev   (slide 15)
        """
        i, j = self.idx_1, self.idx_2
        v1_prev = v_prev[i] if i is not None else 0.0
        v2_prev = v_prev[j] if j is not None else 0.0
        v_diff_prev = v1_prev - v2_prev

        if method == 'TR':
            g_eq = 2.0 * self.value / dt
            I_eq = g_eq * v_diff_prev + self._i_cap_prev
        else:
            g_eq = self.value / dt
            I_eq = g_eq * v_diff_prev

        if i is not None: Y[i, i] += g_eq
        if j is not None: Y[j, j] += g_eq
        if i is not None and j is not None:
            Y[i, j] -= g_eq
            Y[j, i] -= g_eq

        if i is not None: sources[i] += I_eq
        if j is not None: sources[j] -= I_eq

    def post_step_update(self, v_new, v_prev, dt, method='BE'):
        """Updates the stored capacitor current after a successful transient step.
        
        TR: i(t+Δt) = (2C/Δt)·(v(t+Δt) − v(t)) − i(t)
        BE: i(t+Δt) = (C/Δt)·(v(t+Δt) − v(t))    [tracked for consistency]
        """
        i, j = self.idx_1, self.idx_2
        v1_new = v_new[i] if i is not None else 0.0
        v2_new = v_new[j] if j is not None else 0.0
        v1_prev = v_prev[i] if i is not None else 0.0
        v2_prev = v_prev[j] if j is not None else 0.0

        v_diff_new = v1_new - v2_new
        v_diff_prev = v1_prev - v2_prev

        if method == 'TR':
            self._i_cap_prev = (2.0 * self.value / dt) * (v_diff_new - v_diff_prev) - self._i_cap_prev
        else:
            self._i_cap_prev = (self.value / dt) * (v_diff_new - v_diff_prev)

    def build_adjoint_history(self, J_hist, dt, v_hat_next, adjoint_state, method='BE'):
        """Builds adjoint RHS memory term.
        
        BE: I_eq_hat = (C/Δt)·ψ̂_next
        TR: I_eq_hat = (2C/Δt)·ψ̂_next + î_prev
        """
        i, j = self.idx_1, self.idx_2
        v1_hat = v_hat_next[i] if i is not None else 0.0
        v2_hat = v_hat_next[j] if j is not None else 0.0
        psi_diff = v1_hat - v2_hat

        if method == 'TR':
            g_eq = 2.0 * self.value / dt
            i_hat_prev = adjoint_state.get(f"{self.name}_i_hat", 0.0)
            I_eq = g_eq * psi_diff + i_hat_prev
        else:
            g_eq = self.value / dt
            I_eq = g_eq * psi_diff

        if i is not None: J_hist[i] += I_eq
        if j is not None: J_hist[j] -= I_eq

    def update_adjoint_state(self, dt, v_hat_next, v_hat, adjoint_state, method='BE'):
        """Updates adjoint capacitor current for TR.
        
        TR: î_new = (2C/Δt)·(ψ̂_current − ψ̂_next) − î_prev
        """
        if method == 'TR':
            i, j = self.idx_1, self.idx_2
            p1_curr = v_hat[i] if i is not None else 0.0
            p2_curr = v_hat[j] if j is not None else 0.0
            p1_next = v_hat_next[i] if i is not None else 0.0
            p2_next = v_hat_next[j] if j is not None else 0.0

            i_hat_prev = adjoint_state.get(f"{self.name}_i_hat", 0.0)
            adjoint_state[f"{self.name}_i_hat"] = \
                (2.0 * self.value / dt) * ((p1_curr - p2_curr) - (p1_next - p2_next)) - i_hat_prev

    def get_sensitivities(self, VI, PsiPhi, w=0.0, dt=None, V_prev=None, method='BE'):
        """Sensitivity w.r.t. Capacitance (C).
        
        The sensitivity integrand ψ·(dv/dt) uses the finite-difference 
        approximation of dv/dt. The factor of 2 from the TR companion model
        (G_eq = 2C/dt) is already accounted for by the adjoint backward sweep
        through the î_prev chain in build_adjoint_history/update_adjoint_state.
        
        The accuracy improvement from TR comes through:
        1. More accurate forward voltages V (O(dt²) vs O(dt))
        2. More accurate adjoint vectors ψ (O(dt²) vs O(dt))
        3. Trapezoidal quadrature in the integration loop (O(dt²) vs O(dt))
        """
        v1 = VI[self.idx_1] if self.idx_1 is not None else 0.0
        v2 = VI[self.idx_2] if self.idx_2 is not None else 0.0
        p1 = PsiPhi[self.idx_1] if self.idx_1 is not None else 0.0
        p2 = PsiPhi[self.idx_2] if self.idx_2 is not None else 0.0
        adj_diff = (p1 - p2)

        if w != 0.0:
            dI_dC = 1j * w * (v1 - v2)
            return {self.name: -adj_diff * dI_dC}
        elif dt is not None and V_prev is not None:
            v1_prev = V_prev[self.idx_1] if self.idx_1 is not None else 0.0
            v2_prev = V_prev[self.idx_2] if self.idx_2 is not None else 0.0
            dV_dt = ((v1 - v2) - (v1_prev - v2_prev)) / dt
            return {self.name: -adj_diff * dV_dt}
        else:
            return {self.name: 0.0}
