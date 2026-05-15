from .base import Component
import numpy as np

class CCCS(Component):
    """
    Current-Controlled Current Source (Type 'F').
    I(n1, n2) = Gain * I(Vcontrol)
    """
    SHIFT_KEY = "gain"

    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0))   # Output +
        self.idx_2 = node_map.get(self.data.get("n2", 0))   # Output -
        
        # Find the branch index of the controlling voltage source
        ctrl_name = self.data.get("controlling_source")
        self.ctrl_branch_idx = node_map.get(ctrl_name)
        
        if self.ctrl_branch_idx is None:
            raise ValueError(
                f"CCCS '{self.name}': controlling source '{ctrl_name}' "
                f"not found in circuit. It must be an independent/dependent voltage source."
            )

    # ==========================================
    # 1. THE PHYSICS EVALUATOR
    # ==========================================
    def evaluate_physics(self, v_k=None, overrides=None, **kwargs):
        """Calculates the effective gain of the CCCS using absolute overrides."""
        overrides = overrides or {}
        
        # PURE PHYSICS: Use override if it exists, otherwise use nominal
        eff_gain = overrides.get(self.name, self.value)
        
        return {"gain": eff_gain}

    # ==========================================
    # 2. STATIC STAMPING (Skeleton Matrix)
    # ==========================================
    def stamp_base_matrix(self, Y):
        """Phase 1: Stamps time/voltage-invariant gain into the skeleton matrix."""
        if self.ctrl_branch_idx is None: return
        
        # Ask evaluate_physics for the nominal gain
        res = self.evaluate_physics()
        g = res.get("gain", 0.0)

        b = self.ctrl_branch_idx

        # Stamp gain at output node rows, controlling branch column
        if self.idx_1 is not None:
            Y[self.idx_1, b] += g
        if self.idx_2 is not None:
            Y[self.idx_2, b] -= g

    # ==========================================
    # 3. SENSITIVITY & WOODBURY ENGINES
    # ==========================================
    def get_sensitivities(self, VI, PsiPhi, **kwargs):
        """Calculates exact sensitivity w.r.t Gain (alpha)."""
        if self.ctrl_branch_idx is None: return {}

        # Forward: controlling branch current
        i_ctrl = VI[self.ctrl_branch_idx]

        # Adjoint: output node difference
        p1 = PsiPhi[self.idx_1] if self.idx_1 is not None else 0.0
        p2 = PsiPhi[self.idx_2] if self.idx_2 is not None else 0.0
        psi_output = p1 - p2

        # Because dY/dGain is exactly 1.0, we just multiply the state variables directly
        return {self.name: -(psi_output * i_ctrl)}

    def stamp_PQ(self, P, Q, start_col_idx):
        """Stamps the CCCS Woodbury topology. 
        
        Current Injection (P): Output nodes.
        State Extraction (Q): Controlling branch current.
        """
        out1, out2 = self.idx_1, self.idx_2
        b_ctrl = self.ctrl_branch_idx 
        
        # P: Where does the current get injected?
        if out1 is not None: P[out1, start_col_idx] = 1.0
        if out2 is not None: P[out2, start_col_idx] = -1.0
        
        # Q: What is the state variable controlling the source?
        if b_ctrl is not None: Q[b_ctrl, start_col_idx] = 1.0

