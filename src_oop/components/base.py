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

    IS_NONLINEAR = False

    def __init__(self, name, data_dict):
        self.name = name
        self.type = data_dict.get("type")
        self.value = data_dict.get("value", 0.0)
        self.data = data_dict 
        
        self.idx_1 = None 
        self.idx_2 = None

    def bind_nodes(self, node_map):
        """Translates string/integer node names into matrix indices.

        Args:
            node_map (dict): The global mapping of node names to matrix indices.
        """
        pass 

    @property
    def differentiable_params(self):
        """Returns a list of parameter names this component can calculate sensitivities for."""
        return [self.name]

    def get_nominal_value(self, param_name):
        """Returns the nominal physical value for a given parameter.
        
        By default, standard 2-terminal components return their primary 'value'.
        Complex components (like transistors) should override this method.
        """
        return self.value

    # === POLYMORPHIC METHODS ===
    def stamp_mna_connection(self, Y):
        """Stamps MNA branch topology (+1/-1) into the admittance matrix."""
        pass
        
    def stamp_static(self, Y):
        """Stamps time/frequency-independent linear terms (e.g., Conductance)."""
        pass
        
    def stamp_dc(self, Y, sources):
        """Stamps DC values for Operating Point (.OP) and initial transient steps."""
        pass
        
    def stamp_ac(self, Y, sources, w):
        """Stamps frequency-dependent complex impedances and AC phasors.
        
        Args:
            w (float): Angular frequency in rad/s.
        """
        pass
        
    def stamp_transient(self, Y, sources, t, dt, v_prev,method='TR'): 
        """Stamps dynamic companion models (Backward Euler) and time-varying sources.
        
        Args:
            t (float): Current simulation time in seconds.
            dt (float): Current time step size.
            v_prev (np.ndarray): The solution vector from the previous time step.
        """
        pass

    def update_transient_state(self, v_now, method='TR'):
        "Updates internal state after each transient step (used by TR)."
        pass

    def stamp_nonlinear(self, Y, sources, p_V_guess, V_guess):
        """Stamps linearized conductances (gm, gds) and equivalent currents for NR.
        
        Args:
            p_V_guess (np.ndarray): The voltage solution from the previous iteration.
            V_guess (np.ndarray): The current working voltage guess.
        """
        pass
    

    # === ADJOINT METHODS ===
    def build_adjoint_history(self, J_hist, dt, v_hat_next, method='BE'):
        """Builds the RHS current history vector for the backward adjoint sweep."""
        pass
        
    def get_sensitivities(self, VI, PsiPhi, **kwargs):
        """Calculates the parameter sensitivities using the Adjoint method.
        
        Args:
            VI (np.ndarray): The forward voltage/current solution vector.
            PsiPhi (np.ndarray): The backward adjoint solution vector.
            **kwargs: Contextual simulation state variables (e.g., domain, w, dt, V_prev, method).
            
        Returns:
            dict: A mapping of parameter names (e.g., 'R1', 'M1_W') to scalar sensitivity values.
        """
        return {}

    def _stamp_branch_equation(self, Y):
        """Shared helper method for components that require MNA branch currents.
        
        Requires `self.branch_idx` to be initialized in `bind_nodes()`.
        """
        i, j, b = getattr(self, 'idx_1', None), getattr(self, 'idx_2', None), getattr(self, 'branch_idx', None)
        
        if b is None: return 
            
        if i is not None:
            Y[i, b] += 1
            Y[b, i] += 1
        if j is not None:
            Y[j, b] -= 1
            Y[b, j] -= 1

    def stamp_PQ(self, P, Q, col_idx):
        pass


    def get_delta_y(self, param_name, dp, **kwargs):
        """Transforms a physical parameter change into a scalar Admittance change.
        
        By default, components that do not populate the Y-matrix (e.g., independent 
        sources) return 0.0. Components with impedance/conductance must override this.
        
        Args:
            param_name (str): The parameter being shifted.
            dp (float): The exact physical delta.
            **kwargs: Simulation state variables (domain, w, dt, method, VI, etc.)
            
        Returns:
            complex or float: The exact shift in matrix admittance (Delta Y).
        """
        return 0.0
