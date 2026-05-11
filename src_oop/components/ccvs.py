from .base import Component

class CCVS(Component):
    """
    Current-Controlled Voltage Source (Type 'H').
    V(n1, n2) = Transresistance * I(Vcontrol)

    """
    REQUIRES_BRANCH_EQ = True

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

    def stamp_base_matrix(self, Y):
        if self.branch_idx is None: return
        
        b = self.branch_idx

        self._stamp_branch_equation(Y)

        # KVL row: -H at controlling branch column
        Y[b, self.ctrl_branch_idx] -= self.transresistance

    def get_sensitivities(self, VI, PsiPhi, **kwargs):
        """Calculates sensitivity w.r.t Transresistance (H)."""
        if self.branch_idx is None or self.ctrl_branch_idx is None: 
            return {}

        # Forward: controlling branch current
        i_ctrl = VI[self.ctrl_branch_idx]

        # Adjoint: branch variable for this voltage source
        psi_branch = PsiPhi[self.branch_idx]

        return {self.name: psi_branch * i_ctrl}

    def stamp_PQ(self, P, Q, col_idx):
        """Stamps the CCVS topology for Woodbury updates. 
        
        Injection (P): The auxiliary branch equation row (KVL).
        Extraction (Q): The controlling branch current.
        """
        b = self.branch_idx
        b_ctrl = self.ctrl_branch_idx 
        
        # P: Where does the parameter change occur in the rows? 
        # Inside the KVL branch equation row!
        if b is not None: P[b, col_idx] = 1.0
        
        # Q: What state variable multiplies against this parameter?
        # The current of the controlling branch!
        if b_ctrl is not None: Q[b_ctrl, col_idx] = 1.0

    def get_delta_y(self, param_name, dp, **kwargs):
        """Transforms a physical parameter change into a scalar Admittance change.
        
        For a CCVS, the transresistance (H) is stamped as -H in the KVL branch 
        equation. Therefore, if the parameter shifts by dp, the matrix shifts by -dp.
        """
        return -dp
