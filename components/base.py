"""
Circuit Components Base Module.

This module defines the abstract Component interface that all specific
electrical components inherit from. It standardizes the polymorphic
stamping, sensitivity, and adjoint methods that the engines rely on.
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

    # Subclasses override this to declare which data-dict keys hold node names.
    # circuit.py uses this to discover all nodes — no more hardcoded scan lists.
    NODE_KEYS = ("n1", "n2")

    def __init__(self, name, data_dict):
        self.name = name
        self.type = data_dict.get("type")
        self.value = data_dict.get("value", 0.0)
        self.data = data_dict 
        
        self.idx_1 = None 
        self.idx_2 = None
        
        # Run optional value validation defined by subclasses
        self._validate()

    def _validate(self):
        """Hook for subclasses to check component values at construction time.
        
        Override this to raise ValueError for physically invalid parameters
        (e.g., zero-ohm resistors, negative capacitance).
        """
        pass

    def bind_nodes(self, node_map):
        """Translates string/integer node names into matrix indices.

        Args:
            node_map (dict): The global mapping of node names to matrix indices.
        """
        pass 

    # === POLYMORPHIC STAMPING METHODS ===

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
        
    def stamp_transient(self, Y, sources, t, dt, v_prev, method='BE'): 
        """Stamps dynamic companion models and time-varying sources.
        
        Args:
            t (float): Current simulation time in seconds.
            dt (float): Current time step size.
            v_prev (np.ndarray): The solution vector from the previous time step.
            method (str): Integration method — 'BE' (Backward Euler) or 'TR' (Trapezoidal).
        """
        pass
    
    def post_step_update(self, v_new, v_prev, dt, method='BE'):
        """Updates internal companion model state after a transient step completes.
        
        Called by the transient engine after each successful solve. Required for
        TR integration, which stores the previous capacitor current / inductor 
        voltage to build the next step's history source.
        
        Args:
            v_new (np.ndarray): The just-solved solution vector for this step.
            v_prev (np.ndarray): The solution vector from the previous step.
            dt (float): The time step size.
            method (str): Integration method ('BE' or 'TR').
        """
        pass

    def reset_transient_state(self):
        """Resets any internal state stored for multi-step integration methods.
        
        Called before starting a new transient run to clear stale TR history.
        """
        pass
        
    def stamp_nonlinear(self, Y, sources, p_V_guess, V_guess):
        """Stamps linearized conductances (gm, gds) and equivalent currents for NR.
        
        Args:
            p_V_guess (np.ndarray): The voltage solution from the previous NR iteration.
            V_guess (np.ndarray): The current working voltage guess.
        """
        pass
    
    # === ADJOINT METHODS ===

    def build_adjoint_history(self, J_hist, dt, v_hat_next, adjoint_state, method='BE'):
        """Builds the RHS current history vector for the backward adjoint sweep."""
        pass
        
    def update_adjoint_state(self, dt, v_hat_next, v_hat, adjoint_state, method='BE'):
        """Updates internal memory states required by integration methods like TR."""
        pass
        
    def get_sensitivities(self, VI, PsiPhi, w=0.0, dt=None, V_prev=None, method='BE'):
        """Calculates parameter sensitivities using the Adjoint method.
        
        The sign convention for sensitivity depends on how the component enters 
        the MNA equations. For conductance-stamped elements (R, C, G), the 
        derivative of I = G*V gives dI/dG = V, and the adjoint formula is:
            sens = -(Psi_i - Psi_j) * (V_i - V_j) * dG/dparam
        
        For branch-equation elements (V, L), the derivative appears in the 
        branch row, giving:
            sens = Psi_branch * dV/dparam
        
        Args:
            method (str): Integration method ('BE' or 'TR'). TR uses the 
                companion model derivative (2/dt) instead of (1/dt) for 
                dynamic components.
        
        Returns:
            dict: A mapping of parameter names (e.g., 'R1', 'M1_W') to scalar 
                  sensitivity values.
        """
        return {}

    # === SHARED HELPERS ===

    def _stamp_branch_equation(self, Y):
        """Shared helper for components that require MNA branch currents.
        
        Stamps the +1/-1 topology connecting the branch equation row to the 
        node voltage columns. Requires `self.branch_idx` to be set in bind_nodes().
        """
        i = getattr(self, 'idx_1', None)
        j = getattr(self, 'idx_2', None)
        b = getattr(self, 'branch_idx', None)
        
        if b is None: 
            return 
            
        if i is not None:
            Y[i, b] += 1
            Y[b, i] += 1
        if j is not None:
            Y[j, b] -= 1
            Y[b, j] -= 1

    def get_noise_sources(self, VI, w):
        """Returns noise current sources for this component at operating point VI.

        Noise sources are modelled as uncorrelated current sources in parallel
        with the component. Each entry gives:
            - 'nodes': (node_p, node_n) where current flows from p to n
            - 'S':     single-sided power spectral density in A²/Hz
            - 'label': human-readable source name

        Args:
            VI (np.ndarray): Current operating point (from AC/DC solve).
            w  (float):      Angular frequency in rad/s (for 1/f noise).

        Returns:
            list of dicts, each with keys 'nodes', 'S', 'label'.
            Returns empty list if component generates no noise.
        """
        return []
