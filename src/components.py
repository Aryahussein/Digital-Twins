import numpy as np
import models # Your models.py for diode/mosfet equations

class Component:
    """The Base Class. Defines the interface for all components."""
    def __init__(self, name, data_dict):
        self.name = name
        self.type = data_dict.get("type")
        self.value = data_dict.get("value", 0.0)
        self.data = data_dict # Keep raw data just in case
        
        # We will populate matrix indices here for O(1) lookups during simulation
        self.idx_1 = None 
        self.idx_2 = None

    def bind_nodes(self, node_map):
        """Looks up the matrix index for its nodes exactly once."""
        pass 

    # === POLYMORPHIC METHODS (Do nothing by default) ===
    def stamp_mna_connection(self, Y): pass
    def stamp_static(self, Y, sources): pass
    def stamp_source(self, Y, sources): pass
    def stamp_ac(self, Y, sources, w): pass
    def stamp_transient(self, Y, sources, dt, v_prev): pass
    def stamp_nonlinear(self, Y, sources, p_V_guess, V_guess): pass
    
    # === ADJOINT METHODS ===
    def build_adjoint_history(self, J_hist, dt, v_hat_next, adjoint_state, method='BE'): pass
    def update_adjoint_state(self, dt, v_hat_next, v_hat, adjoint_state, method='BE'): pass
    def get_sensitivities(self, VI, PsiPhi, w=0.0, dt=None, V_prev=None): return {}

# =====================================================================
# LINEAR COMPONENTS
# =====================================================================
class Resistor(Component):
    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0))
        self.idx_2 = node_map.get(self.data.get("n2", 0))

    def stamp_static(self, Y, sources):
        g = 1.0 / self.value
        i, j = self.idx_1, self.idx_2
        if i is not None:
            Y[i, i] += g
            if j is not None:
                Y[i, j] -= g
                Y[j, i] -= g
        if j is not None:
            Y[j, j] += g

    def get_sensitivities(self, VI, PsiPhi, w=0.0, dt=None, V_prev=None):
        v1 = VI[self.idx_1] if self.idx_1 is not None else 0.0
        v2 = VI[self.idx_2] if self.idx_2 is not None else 0.0
        p1 = PsiPhi[self.idx_1] if self.idx_1 is not None else 0.0
        p2 = PsiPhi[self.idx_2] if self.idx_2 is not None else 0.0
        
        return {self.name: (1.0 / (self.value**2)) * ((v1 - v2) * (p1 - p2))}

class Capacitor(Component):
    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0))
        self.idx_2 = node_map.get(self.data.get("n2", 0))

    def stamp_transient(self, Y, sources, dt, v_prev):
        g_eq = self.value / dt
        i, j = self.idx_1, self.idx_2
        
        v1_prev = v_prev[i] if i is not None else 0.0
        v2_prev = v_prev[j] if j is not None else 0.0
        I_eq = g_eq * (v1_prev - v2_prev)
        
        if i is not None: Y[i, i] += g_eq
        if j is not None: Y[j, j] += g_eq
        if i is not None and j is not None:
            Y[i, j] -= g_eq
            Y[j, i] -= g_eq
            
        if i is not None: sources[i] += I_eq
        if j is not None: sources[j] -= I_eq

    def build_adjoint_history(self, J_hist, dt, v_hat_next, adjoint_state, method='BE'):
        i, j = self.idx_1, self.idx_2
        v1_hat = v_hat_next[i] if i is not None else 0.0
        v2_hat = v_hat_next[j] if j is not None else 0.0
        
        I_eq = (self.value / dt) * (v1_hat - v2_hat)
        if i is not None: J_hist[i] += I_eq
        if j is not None: J_hist[j] -= I_eq

    def get_sensitivities(self, VI, PsiPhi, w=0.0, dt=None, V_prev=None):
        if dt is None or V_prev is None: return {}
        
        v1 = VI[self.idx_1] if self.idx_1 is not None else 0.0
        v2 = VI[self.idx_2] if self.idx_2 is not None else 0.0
        p1 = PsiPhi[self.idx_1] if self.idx_1 is not None else 0.0
        p2 = PsiPhi[self.idx_2] if self.idx_2 is not None else 0.0
        
        v1_prev = V_prev[self.idx_1] if self.idx_1 is not None else 0.0
        v2_prev = V_prev[self.idx_2] if self.idx_2 is not None else 0.0
        
        dV_dt = ((v1 - v2) - (v1_prev - v2_prev)) / dt
        return {self.name: -(p1 - p2) * dV_dt}

class VoltageSource(Component):
    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0))
        self.idx_2 = node_map.get(self.data.get("n2", 0))
        self.branch_idx = node_map.get(self.name) # MNA branch current index

    def stamp_mna_connection(self, Y):
        i, j, b = self.idx_1, self.idx_2, self.branch_idx
        if i is not None:
            Y[i, b] += 1
            Y[b, i] += 1
        if j is not None:
            Y[j, b] -= 1
            Y[b, j] -= 1

    def stamp_source(self, Y, sources):
        sources[self.branch_idx] = self.value

# =====================================================================
# NONLINEAR COMPONENTS
# =====================================================================
class Mosfet(Component):
    def bind_nodes(self, node_map):
        self.idx_d = node_map.get(self.data.get("n_d", 0))
        self.idx_g = node_map.get(self.data.get("n_g", 0))
        self.idx_s = node_map.get(self.data.get("n_s", 0))
        
        params = self.data.get("model_params", {})
        inst = self.data.get("inst_params", {})
        
        self.VTO = params.get("VTO", 0.7)
        self.W = inst.get("W", 1e-6)
        self.L = inst.get("L", 1e-6)
        mu, Cox = params.get("MU", 0.0), params.get("C_OX", 0.0)
        self.KP = params.get("KP", mu * Cox if mu and Cox else 0.0)
        self.Bn = (self.W / self.L) * self.KP

    def stamp_nonlinear(self, Y, sources, p_V_guess, V_guess):
        # 1. Extract voltages
        vd = V_guess[self.idx_d] if self.idx_d is not None else 0.0
        vg = V_guess[self.idx_g] if self.idx_g is not None else 0.0
        vs = V_guess[self.idx_s] if self.idx_s is not None else 0.0
        
        # 2. Limit and Evaluate (using your existing limits)
        vgs_k, vds_k = vg - vs, vd - vs 
        nmos_data = models.evaluate_nmos(vgs_k, vds_k, self.VTO, self.Bn)
        Id, gm, gds = nmos_data["I_D"], nmos_data["gm"], nmos_data["gds"]

        # 3. Stamp GM
        if self.idx_d is not None:
            if self.idx_g is not None: Y[self.idx_d, self.idx_g] += gm
            if self.idx_s is not None: Y[self.idx_d, self.idx_s] -= gm
        if self.idx_s is not None:
            if self.idx_g is not None: Y[self.idx_s, self.idx_g] -= gm
            if self.idx_s is not None: Y[self.idx_s, self.idx_s] += gm

        # 4. Stamp GDS
        if self.idx_d is not None:
            Y[self.idx_d, self.idx_d] += gds
            if self.idx_s is not None:
                Y[self.idx_d, self.idx_s] -= gds
                Y[self.idx_s, self.idx_d] -= gds
                Y[self.idx_s, self.idx_s] += gds

        # 5. Stamp Ieq
        Ieq = Id - (gm * vgs_k) - (gds * vds_k)
        if self.idx_d is not None: sources[self.idx_d] -= Ieq
        if self.idx_s is not None: sources[self.idx_s] += Ieq
