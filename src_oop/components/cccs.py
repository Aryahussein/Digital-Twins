from .base import Component

class CCCS(Component):
    """
    Current-Controlled Current Source (Type 'F').
    I(n1, n2) = Gain * I(Vcontrol)

    """

    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0))   # Output +
        self.idx_2 = node_map.get(self.data.get("n2", 0))   # Output -
        self.gain = self.value
        
        # Find the branch index of the controlling voltage source
        ctrl_name = self.data.get("controlling_source")
        self.ctrl_branch_idx = node_map.get(ctrl_name)
        
        if self.ctrl_branch_idx is None:
            raise ValueError(
                f"CCCS '{self.name}': controlling source '{ctrl_name}' "
                f"not found in circuit. It must be a voltage source."
            )

    def stamp_static(self, Y):
        b = self.ctrl_branch_idx
        g = self.gain

        # Stamp gain at output node rows, controlling branch column
        if self.idx_1 is not None:
            Y[self.idx_1, b] += g
        if self.idx_2 is not None:
            Y[self.idx_2, b] -= g

    def get_sensitivities(self, VI, PsiPhi, **kwargs):
        """Calculates sensitivity w.r.t Gain (alpha)."""
        if self.ctrl_branch_idx is None: return {}

        # Forward: controlling branch current
        i_ctrl = VI[self.ctrl_branch_idx]

        # Adjoint: output node difference
        p1 = PsiPhi[self.idx_1] if self.idx_1 is not None else 0.0
        p2 = PsiPhi[self.idx_2] if self.idx_2 is not None else 0.0
        psi_output = p1 - p2

        return {self.name: -(psi_output * i_ctrl)}
