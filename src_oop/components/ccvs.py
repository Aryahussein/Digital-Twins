from .base import Component

class CCVS(Component):
    """
    Current-Controlled Voltage Source (Type 'H').
    V(n1, n2) = Transresistance * I(Vcontrol)

    """

    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0))   # Output +
        self.idx_2 = node_map.get(self.data.get("n2", 0))   # Output -
        self.branch_idx = node_map.get(self.name)
        self.transresistance = self.value
        
        # Find the branch index of the controlling voltage source
        ctrl_name = self.data.get("controlling_source")
        self.ctrl_branch_idx = node_map.get(ctrl_name)
        
        if self.ctrl_branch_idx is None:
            raise ValueError(
                f"CCVS '{self.name}': controlling source '{ctrl_name}' "
                f"not found in circuit. It must be a voltage source."
            )

    def stamp_mna_connection(self, Y):
        if self.branch_idx is None: return
        
        b = self.branch_idx

        # Output topology: branch current enters n1, leaves n2
        if self.idx_1 is not None:
            Y[self.idx_1, b] += 1.0
            Y[b, self.idx_1] += 1.0
        if self.idx_2 is not None:
            Y[self.idx_2, b] -= 1.0
            Y[b, self.idx_2] -= 1.0

        # KVL row: -H at controlling branch column
        Y[b, self.ctrl_branch_idx] -= self.transresistance

    def get_sensitivities(self, VI, PsiPhi, **kwargs):
        """Calculates sensitivity w.r.t Transresistance (H)."""
        if self.branch_idx is None: return {}

        # Forward: controlling branch current
        i_ctrl = VI[self.ctrl_branch_idx]

        # Adjoint: branch variable
        psi_branch = PsiPhi[self.branch_idx]

        return {self.name: psi_branch * i_ctrl}
