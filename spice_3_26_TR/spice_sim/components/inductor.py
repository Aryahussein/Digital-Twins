"""Inductor Component Module."""

from .base import Component


class Inductor(Component):
    """A linear inductor (Type 'L'). Requires an MNA branch equation.
    
    In DC: short circuit (V1 - V2 = 0).
    In AC: stamps impedance Z = j*w*L in the branch row.
    In TRAN (BE): V_L = (L/Δt)·(I − I_prev).
    In TRAN (TR): V_L = (2L/Δt)·(I − I_prev) − V_L_prev.
    
    The TR companion is derived from L·(i_{n+1}−i_n) = (Δt/2)·(v_L(n)+v_L(n+1)),
    rearranged as: v_L(n+1) = (2L/Δt)·i_{n+1} − [(2L/Δt)·i_n + v_L(n)].
    """

    NODE_KEYS = ("n1", "n2")

    def __init__(self, name, data_dict):
        super().__init__(name, data_dict)
        self._v_L_prev = 0.0  # TR state: previous voltage across inductor

    def _validate(self):
        if self.value == 0.0:
            raise ValueError(
                f"Inductor '{self.name}' has value 0 H. "
                "This creates a division-by-zero in the transient companion model."
            )
        if self.value < 0.0:
            raise ValueError(
                f"Inductor '{self.name}' has negative inductance ({self.value})."
            )

    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0))
        self.idx_2 = node_map.get(self.data.get("n2", 0))
        self.branch_idx = node_map.get(self.name)

    def reset_transient_state(self):
        """Clear TR history for a fresh simulation run."""
        self._v_L_prev = 0.0

    def stamp_mna_connection(self, Y):
        self._stamp_branch_equation(Y)

    def stamp_dc(self, Y, sources):
        """In DC, an inductor is a short circuit (V1 - V2 = 0)."""
        sources[self.branch_idx] = 0.0

    def stamp_ac(self, Y, sources, w):
        """Stamps complex impedance: Z = j*w*L."""
        z = 1j * w * self.value
        if self.branch_idx is not None:
            Y[self.branch_idx, self.branch_idx] -= z

    def stamp_transient(self, Y, sources, t, dt, v_prev, method='BE'):
        """Stamps the inductor companion model into the MNA branch equation.
        
        BE:  R_eq = L/Δt,    V_eq = (L/Δt)·i_prev
        TR:  R_eq = 2L/Δt,   V_eq = (2L/Δt)·i_prev + v_L_prev
        """
        if self.branch_idx is None:
            return

        i_prev = v_prev[self.branch_idx]

        if method == 'TR':
            req = 2.0 * self.value / dt
            v_eq = req * i_prev + self._v_L_prev
        else:
            req = self.value / dt
            v_eq = req * i_prev

        Y[self.branch_idx, self.branch_idx] -= req
        sources[self.branch_idx] -= v_eq

    def post_step_update(self, v_new, v_prev, dt, method='BE'):
        """Updates the stored inductor voltage after a successful transient step.
        
        TR: v_L(n+1) = (2L/Δt)·(i_{n+1} − i_n) − v_L(n)
        """
        if self.branch_idx is None:
            return

        i_new = v_new[self.branch_idx]
        i_prev = v_prev[self.branch_idx]

        if method == 'TR':
            self._v_L_prev = (2.0 * self.value / dt) * (i_new - i_prev) - self._v_L_prev
        else:
            self._v_L_prev = (self.value / dt) * (i_new - i_prev)

    def build_adjoint_history(self, J_hist, dt, v_hat_next, adjoint_state, method='BE'):
        """Builds the adjoint RHS memory term.
        
        BE: V_eq_hat = (L/Δt)·ψ̂_branch_next
        TR: V_eq_hat = (2L/Δt)·ψ̂_branch_next + v̂_L_prev
        """
        if self.branch_idx is None:
            return

        psi_branch = v_hat_next[self.branch_idx]

        if method == 'TR':
            req = 2.0 * self.value / dt
            v_hat_prev = adjoint_state.get(f"{self.name}_v_hat", 0.0)
            V_eq_hat = req * psi_branch + v_hat_prev
        else:
            req = self.value / dt
            V_eq_hat = req * psi_branch

        J_hist[self.branch_idx] -= V_eq_hat

    def update_adjoint_state(self, dt, v_hat_next, v_hat, adjoint_state, method='BE'):
        """Updates adjoint inductor voltage for TR.
        
        TR: v̂_L_new = (2L/Δt)·(ψ̂_branch_curr − ψ̂_branch_next) − v̂_L_prev
        """
        if method == 'TR' and self.branch_idx is not None:
            psi_curr = v_hat[self.branch_idx]
            psi_next = v_hat_next[self.branch_idx]
            v_hat_prev = adjoint_state.get(f"{self.name}_v_hat", 0.0)

            adjoint_state[f"{self.name}_v_hat"] = \
                (2.0 * self.value / dt) * (psi_curr - psi_next) - v_hat_prev

    def get_sensitivities(self, VI, PsiPhi, w=0.0, dt=None, V_prev=None, method='BE'):
        """Sensitivity w.r.t. Inductance (L).
        
        Uses 1/dt for both BE and TR — the TR factor of 2 is handled by 
        the adjoint backward sweep through the v̂_L_prev chain.
        """
        if self.branch_idx is None:
            return {}

        i_L = VI[self.branch_idx]
        psi_branch = PsiPhi[self.branch_idx]

        if w != 0.0:
            return {self.name: psi_branch * 1j * w * i_L}
        elif dt is not None and V_prev is not None:
            i_L_prev = V_prev[self.branch_idx]
            dI_dt = (i_L - i_L_prev) / dt
            return {self.name: psi_branch * dI_dt}
        else:
            return {self.name: 0.0}
