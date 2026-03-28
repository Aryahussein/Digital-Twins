"""Voltage-Controlled Current Source (VCCS) Component Module."""

from .base import Component


class VCCS(Component):
    """Voltage-Controlled Current Source (Type 'G').
    
    I_out = G * (V_ctrl+ - V_ctrl-)
    
    SPICE format: G<n> N_OUT+ N_OUT- NC+ NC- G_VALUE
    Current flows from N_OUT+ to N_OUT- proportional to the control voltage.
    """

    NODE_KEYS = ("n1", "n2", "n3", "n4")

    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0))  # Output +
        self.idx_2 = node_map.get(self.data.get("n2", 0))  # Output -
        self.idx_3 = node_map.get(self.data.get("n3", 0))  # Control +
        self.idx_4 = node_map.get(self.data.get("n4", 0))  # Control -

    def stamp_static(self, Y):
        """Stamps the transconductance G into the admittance matrix."""
        g = self.value
        i, j, k, l = self.idx_1, self.idx_2, self.idx_3, self.idx_4
        
        if i is not None:
            if k is not None: Y[i, k] += g
            if l is not None: Y[i, l] -= g
        
        if j is not None:
            if k is not None: Y[j, k] -= g
            if l is not None: Y[j, l] += g

    def get_sensitivities(self, VI, PsiPhi, w=0.0, dt=None, V_prev=None, method='BE'):
        """Sensitivity w.r.t. Transconductance (G).
        
        Derivation:
            VCCS stamps G into the off-diagonal positions of the MNA matrix.
            The derivative dY/dG gives the topology matrix entries.
            Adjoint formula: sens = -Psi^T * (dY/dG) * V
            
            Expanding: The current I = G * V_ctrl flows out of node i and 
            into node j. The adjoint projection gives:
                sens = -(Psi_i - Psi_j) * (V_k - V_l)
            
            The negative sign comes from the adjoint formula: the MNA equation
            is Y*V = s, so dV/dp = -Y^{-1} * (dY/dp) * V. The adjoint method
            computes Psi^T * Y^{-1} implicitly, leaving:
                sens = -Psi^T * (dY/dp) * V
        """
        v3 = VI[self.idx_3] if self.idx_3 is not None else 0.0
        v4 = VI[self.idx_4] if self.idx_4 is not None else 0.0
        v_control = v3 - v4

        p1 = PsiPhi[self.idx_1] if self.idx_1 is not None else 0.0
        p2 = PsiPhi[self.idx_2] if self.idx_2 is not None else 0.0
        psi_output = p1 - p2

        return {self.name: -(psi_output * v_control)}
