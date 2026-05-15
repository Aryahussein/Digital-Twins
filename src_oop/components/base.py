"""
Circuit Components Module.

This module defines the base Component interface and implements the specific
polymorphic behaviors (matrix stamping, sensitivity calculations) for various
electrical components (Resistors, Capacitors, Sources, Transistors).
"""

import numpy as np

class Component:
    """The Base Class defining the interface for all circuit components.

    Attributes:
        name (str): The unique netlist name (e.g., 'R1', 'M_INV').
        type (str): The component type character (e.g., 'R', 'C').
        value (float): The default value of the component, if applicable.
        data (dict): The raw parsed dictionary from the netlist.
        idx_1 (int or None): The primary matrix index (often positive terminal).
        idx_2 (int or None): The secondary matrix index (often negative terminal).
    """

    IS_NONLINEAR = False          # Requires Newton-Raphson iteration
    IS_DYNAMIC = False            # Requires time-domain Companion Models (dt)
    IS_AC_REACTIVE = False        # Requires frequency-domain Phasors (w)
    IS_INDEPENDENT_SOURCE = False # Stamped directly into the RHS vector
    REQUIRES_BRANCH_EQ = False
    SHIFT_KEY = "g_eq"            # The key to extract from evaluate_physics
    SHIFT_MULTIPLIER = 1.0        # Multiplier for the shift value
    
    # Woodbury Subspace Rank (1 for 2-terminal passives, 2 for 4-terminal MOSFETs)
    rank = 1 

    def __init__(self, name, data_dict):
        self.name = name
        self.type = data_dict.get("type")
        self.value = data_dict.get("value", 0.0)
        self.data = data_dict 
        
        self.idx_1 = None 
        self.idx_2 = None

    def bind_nodes(self, node_map):
        """Translates string/integer node names into matrix indices."""
        pass 

    @property
    def differentiable_params(self):
        """Returns a list of parameter names this component can calculate sensitivities for."""
        return [self.name]

    def get_nominal_value(self, param_name):
        """Returns the nominal physical value for a given parameter."""
        return self.value

    def set_nominal_value(self, param_name, new_val):
        """Updates the physical value of the component."""
        self.value = new_val

    # === POLYMORPHIC METHODS ===

    def stamp_base_matrix(self, Y):
        """Phase 1 (Skeleton): Stamps time-invariant, voltage-invariant MNA topology."""
        pass

    def stamp_sources(self, J, domain, t=0.0):
        """Phase 2 (Sources): Stamps independent values into the RHS vector."""
        pass

    def stamp_ac(self, Y, sources, w):
        """Stamps frequency-dependent complex impedances and AC phasors."""
        pass

    def evaluate_physics(self, v_k=None, shifts=None, **kwargs):
        """Pure mathematical evaluation of component physics.
        
        Safely accepts a 'shifts' dictionary to calculate effective parameters dynamically.
        Simulation state variables (domain, t, dt, method, v_prev, w) are packed into kwargs.
        
        Returns a dictionary of matrix values (e.g., {'g_eq': val, 'I_eq': val}).
        """
        return {}

    def stamp_matrix(self, Y, res, *args):
        """Universal MNA Stamper for 2-Terminal Admittances."""
        # Use the class attribute we set up earlier (Defaults to "g_eq", Diode overrides to "gd")
        key = getattr(self, "SHIFT_KEY", "g_eq")
        g = res.get(key, 0.0)
        
        if g == 0.0: return
        
        # Safely grab the positive/negative indices
        idx_pos = getattr(self, 'idx_1', getattr(self, 'idx_a', None))
        idx_neg = getattr(self, 'idx_2', getattr(self, 'idx_k', None))
        
        if idx_pos is not None: Y[idx_pos, idx_pos] += g
        if idx_neg is not None: Y[idx_neg, idx_neg] += g
        if idx_pos is not None and idx_neg is not None:
            Y[idx_pos, idx_neg] -= g
            Y[idx_neg, idx_pos] -= g

    def stamp_rhs(self, J, res, *args):
        """Universal MNA Stamper for Equivalent Currents.
        Convention: I_eq flows from positive to negative.
        """
        I_eq = res.get("I_eq", 0.0)
        if I_eq == 0.0: return
        
        idx_pos = getattr(self, 'idx_1', getattr(self, 'idx_a', None))
        idx_neg = getattr(self, 'idx_2', getattr(self, 'idx_k', None))
        
        # Universally subtract from positive, add to negative
        if idx_pos is not None: J[idx_pos] -= I_eq
        if idx_neg is not None: J[idx_neg] += I_eq

    def update_transient_state(self, v_now, v_prev, dt, method='TR'):
        """Updates internal history state after a converged transient step."""
        pass


    # === ADJOINT METHODS ===
    def build_adjoint_history(self, J_hist, dt, v_hat_next, method='BE'):
        """Builds the RHS current history vector for the backward adjoint sweep."""
        pass
        
    def get_sensitivities(self, VI, PsiPhi, **kwargs):
        """Calculates the parameter sensitivities using the Adjoint method."""
        return {}

    def _stamp_branch_equation(self, Y):
        """Shared helper method for components that require MNA branch currents."""
        i, j, b = getattr(self, 'idx_1', None), getattr(self, 'idx_2', None), getattr(self, 'branch_idx', None)
        
        if b is None: return 
            
        if i is not None:
            Y[i, b] += 1
            Y[b, i] += 1
        if j is not None:
            Y[j, b] -= 1
            Y[b, j] -= 1

    def stamp_PQ(self, P, Q, start_col_idx):
        """
        Universal Rank-1 Woodbury Topology for 2-Terminal Components.
        
        P (Injection): Current flows from positive to negative terminal.
        Q (Extraction): State variable is the voltage drop across the terminals.
        """
        # Safely grab the positive/negative indices, regardless of what the subclass calls them
        idx_pos = getattr(self, 'idx_1', getattr(self, 'idx_a', None))
        idx_neg = getattr(self, 'idx_2', getattr(self, 'idx_k', None))
        
        if idx_pos is not None: 
            P[idx_pos, start_col_idx] = 1.0
            Q[idx_pos, start_col_idx] = 1.0
            
        if idx_neg is not None: 
            P[idx_neg, start_col_idx] = -1.0
            Q[idx_neg, start_col_idx] = -1.0


    def get_delta_y(self, shifts=None, V_nom=None, V_k=None, **kwargs):
        """Universal Rank-1 Woodbury shift."""
        shifts = shifts or {}
        
        if not shifts and V_nom is not None and np.allclose(V_nom, V_k):
            return 0.0

        new_overrides = {}
        old_overrides = {}
        
        for param_name, delta in shifts.items():
            nom_val = self.get_nominal_value(param_name)
            
            # If the engine already mutated the component, nom_val is the NEW reality.
            # If the engine hasn't mutated the component, nom_val is the OLD reality.
            # (Assuming the engine ALREADY mutated self.value):
            new_overrides[param_name] = nom_val 
            old_overrides[param_name] = nom_val - delta

        # 1. Evaluate New Reality (with absolute parameters)
        res_new = self.evaluate_physics(v_k=V_k, overrides=new_overrides, **kwargs)
        
        # 2. Evaluate Old Reality (with absolute parameters)
        res_old = self.evaluate_physics(v_k=V_nom, overrides=old_overrides, **kwargs)

        key = self.SHIFT_KEY
        return (res_new.get(key, 0.0) - res_old.get(key, 0.0)) * getattr(self, "SHIFT_MULTIPLIER", 1.0)

