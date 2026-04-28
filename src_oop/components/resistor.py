from .base import Component

class Resistor(Component):
    """A linear resistor component."""
    
    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0))
        self.idx_2 = node_map.get(self.data.get("n2", 0))

    def _get_conductance(self, r_val):
        """Centralized helper for resistance to conductance conversion."""
        if r_val == 0:
            return 0.0 # Prevent division by zero
        return 1.0 / r_val

    def stamp_static(self, Y):
        g = self._get_conductance(self.value)
        i, j = self.idx_1, self.idx_2
        if i is not None:
            Y[i, i] += g
            if j is not None:
                Y[i, j] -= g
                Y[j, i] -= g
        if j is not None:
            Y[j, j] += g

    def get_sensitivities(self, VI, PsiPhi, **kwargs):
        """Calculates Adjoint sensitivity w.r.t Resistance (R)."""
        v1 = VI[self.idx_1] if self.idx_1 is not None else 0.0
        v2 = VI[self.idx_2] if self.idx_2 is not None else 0.0
        p1 = PsiPhi[self.idx_1] if self.idx_1 is not None else 0.0
        p2 = PsiPhi[self.idx_2] if self.idx_2 is not None else 0.0
        
        # Avoid division by zero if a 0-ohm resistor is modeled
        if self.value == 0: return {self.name: 0.0}
        
        # Exact Adjoint mathematical derivative for a resistor
        return {self.name: (1.0 / (self.value**2)) * ((v1 - v2) * (p1 - p2))}

    def stamp_PQ(self, P, Q, col_idx):
        """Stamps the Injection (P) and Extraction (Q) topology vectors in-place."""
        i, j = self.idx_1, self.idx_2
        
        if i is not None: 
            P[i, col_idx] = 1.0
            Q[i, col_idx] = 1.0
        if j is not None: 
            P[j, col_idx] = -1.0
            Q[j, col_idx] = -1.0

    def get_delta_y(self, param_name, dp, **kwargs):
        """Transforms a physical parameter change into a scalar Admittance change.
        
        Calculates the EXACT non-linear finite difference for Large Change analysis.
        Delta Y = (1 / (R + dp)) - (1 / R)
        """
        r_nom = self.value
        r_new = r_nom + dp
        
        # Calculate exactly using our helper to avoid redundant division logic
        g_nom = self._get_conductance(r_nom)
        g_new = self._get_conductance(r_new)
        
        return g_new - g_nom
