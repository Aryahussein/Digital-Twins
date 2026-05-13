from .base import Component

class Resistor(Component):
    """A linear resistor component (Type 'R')."""
    
    # Optional, but great for explicit documentation:
    SHIFT_KEY = "g_eq" 
    
    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0))
        self.idx_2 = node_map.get(self.data.get("n2", 0))

    # ==========================================
    # 1. THE PHYSICS EVALUATOR
    # ==========================================
    def evaluate_physics(self, v_k=None, overrides=None, **kwargs):
        """Pure evaluation of static conductance using absolute overrides."""
        overrides = overrides or {}
        
        eff_r = overrides.get(self.name, self.value)
        
        if eff_r == 0.0:
            return {"g_eq": 1e12} 
            
        return {"g_eq": 1.0 / eff_r}

    # ==========================================
    # 2. SENSITIVITY ENGINES
    # ==========================================
    def get_sensitivities(self, VI, PsiPhi, **kwargs):
        """Calculates exact Adjoint sensitivity w.r.t Resistance (R)."""
        v1 = VI[self.idx_1] if self.idx_1 is not None else 0.0
        v2 = VI[self.idx_2] if self.idx_2 is not None else 0.0
        p1 = PsiPhi[self.idx_1] if self.idx_1 is not None else 0.0
        p2 = PsiPhi[self.idx_2] if self.idx_2 is not None else 0.0
        
        if self.value == 0.0: return {self.name: 0.0}
        
        return {self.name: (1.0 / (self.value**2)) * ((v1 - v2) * (p1 - p2))}
