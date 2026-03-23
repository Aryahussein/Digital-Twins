from .base import Component

class OpAmp(Component):
    """
    Ideal Op-Amp (Type 'E'). 
    Implemented as a Voltage-Controlled Voltage Source (VCVS).
    V(out, gnd) = Gain * (V(n_plus) - V(n_minus))
    """

    IS_NONLINEAR = True

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

    def stamp_mna_connection(self, Y):
        """
        Stamps the VCVS MNA equations.
        Equation: V_out - Gain*(V_p - V_m) = 0
        """
        if self.branch_idx is None: return

        # 1. Output current flows into the output node
        if self.idx_out is not None:
            Y[self.idx_out, self.branch_idx] += 1.0
            Y[self.branch_idx, self.idx_out] += 1.0

        # 2. Control voltage dependencies in the branch row
        if self.idx_p is not None:
            Y[self.branch_idx, self.idx_p] -= self.gain
        if self.idx_m is not None:
            Y[self.branch_idx, self.idx_m] += self.gain

    # def stamp_static(self, Y, sources):
    #     # RHS for an ideal Op-Amp is typically 0 (homogeneous equation)
    #     if self.branch_idx is not None:
    #         sources[self.branch_idx] = 0.0

    # # AC and Transient inherit from static
    # def stamp_ac(self, Y, sources, w): self.stamp_static(Y, sources)
    # def stamp_transient(self, Y, sources, t, dt, v_prev): self.stamp_static(Y, sources)

    def get_sensitivities(self, VI, PsiPhi, w=0.0, dt=None, V_prev=None):
        """Calculates sensitivity w.r.t Open-Loop Gain (A)."""
        if self.branch_idx is None: return {}

        # Forward differential input
        vp = VI[self.idx_p] if self.idx_p is not None else 0.0
        vm = VI[self.idx_m] if self.idx_m is not None else 0.0
        v_diff = vp - vm

        # Adjoint branch variable
        psi_branch = PsiPhi[self.branch_idx]

        # Adjoint formula: Psi_branch * (V_plus - V_minus)
        return {self.name: psi_branch * v_diff}
