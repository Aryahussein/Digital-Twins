from .base import Component

class OpAmp(Component):
    """
    Ideal Op-Amp (Type 'E'). 
    Implemented as a Voltage-Controlled Voltage Source (VCVS).
    V(out, gnd) = Gain * (V(n_plus) - V(n_minus))
    """

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
    # SOLVER ENGINES (Stamping)
    # ==========================================
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

    # ==========================================
    # SENSITIVITY ENGINES (Adjoint & Woodbury)
    # ==========================================
    def get_sensitivities(self, VI, PsiPhi, **kwargs):
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

    def stamp_PQ(self, P, Q, col_idx):
        """Stamps the VCVS topology for Woodbury updates.
        
        Injection (P): The auxiliary branch equation row.
        Extraction (Q): The differential input voltage (V_plus - V_minus).
        """
        b = self.branch_idx
        p, m = self.idx_p, self.idx_m
        
        # P: The gain parameter lives entirely inside the KVL branch equation row
        if b is not None: 
            P[b, col_idx] = 1.0
            
        # Q: The state variable multiplying the gain is (V_plus - V_minus)
        if p is not None: Q[p, col_idx] = 1.0
        if m is not None: Q[m, col_idx] = -1.0

    def get_delta_y(self, param_name, dp, **kwargs):
        """Transforms a physical parameter change into a scalar Admittance change.
        
        Because the KVL equation is V_out - A * (V_p - V_m) = 0, the gain A 
        is stamped as a negative multiplier for V_p. Therefore, if the gain 
        shifts by dp, the matrix shifts by -dp.
        """
        return -dp
