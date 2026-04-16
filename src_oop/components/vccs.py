from .base import Component

class VCCS(Component):
    """Voltage-Controlled Current Source (Type 'G'). I = G * (V_pos - V_neg)."""

    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0)) # Output +
        self.idx_2 = node_map.get(self.data.get("n2", 0)) # Output -
        self.idx_3 = node_map.get(self.data.get("n3", 0)) # Control +
        self.idx_4 = node_map.get(self.data.get("n4", 0)) # Control -

    def stamp_static(self, Y):
        g = self.value # The transconductance
        i, j, k, l = self.idx_1, self.idx_2, self.idx_3, self.idx_4
        
        # Current out of node i depends on (Vk - Vl)
        if i is not None:
            if k is not None: Y[i, k] += g
            if l is not None: Y[i, l] -= g
        
        # Current into node j depends on (Vk - Vl)
        if j is not None:
            if k is not None: Y[j, k] -= g
            if l is not None: Y[j, l] += g

    # AC and Transient inherit from static as the gain is frequency independent.
    # def stamp_ac(self, Y, sources, w): self.stamp_static(Y, sources)
    # def stamp_transient(self, Y, sources, t, dt, v_prev): self.stamp_static(Y, sources)

    def get_sensitivities(self, VI, PsiPhi, **kwargs):
        """
        Calculates sensitivity w.r.t the Transconductance (G).
        Formula: -(Psi_out+ - Psi_out-) * (V_ctrl+ - V_ctrl-)
        """
        # 1. Get Forward (Primal) Control Voltage
        v3 = VI[self.idx_3] if self.idx_3 is not None else 0.0
        v4 = VI[self.idx_4] if self.idx_4 is not None else 0.0
        v_control = v3 - v4

        # 2. Get Backward (Adjoint) Output Voltage
        p1 = PsiPhi[self.idx_1] if self.idx_1 is not None else 0.0
        p2 = PsiPhi[self.idx_2] if self.idx_2 is not None else 0.0
        psi_output = p1 - p2

        # 3. Sensitivity is the product of the control voltage and the adjoint output
        # Note the negative sign: it comes from the MNA matrix derivative
        sens_g = -(psi_output * v_control)

        return {self.name: sens_g}
