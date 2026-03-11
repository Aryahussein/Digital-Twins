"""
DC Analysis Engine Module.

This module is responsible for compiling the static Base MNA matrices and 
solving for the DC Operating Point (bias point) of the circuit. The operating 
point is required before starting AC analysis or Transient integration.
"""

import numpy as np
from scipy.sparse import lil_matrix
from engines.solver import solve_linear_circuit, NonlinearSolver

class DCEngine:
    """Compiles circuit topology and calculates the steady-state DC bias.

    Attributes:
        circuit (Circuit): The main circuit object containing components.
        is_complex (bool): Flag indicating if the matrix must support complex 
            numbers (True if an AC analysis is requested).
        is_nonlinear (bool): Flag indicating the presence of nonlinear devices
            requiring Newton-Raphson iteration.
        ramp (int): Legacy parameter for source-stepping steps.
    """

    def __init__(self, circuit, is_complex, is_nonlinear, ramp=10):
        """Initializes the DC Engine.

        Args:
            circuit (Circuit): The fully populated circuit object.
            is_complex (bool): Whether the base matrix requires complex dtype.
            is_nonlinear (bool): Whether the circuit requires a nonlinear solver.
            ramp (int, optional): Source ramping steps for convergence. Defaults to 10.
        """
        self.circuit = circuit
        self.is_complex = is_complex
        self.is_nonlinear = is_nonlinear
        self.ramp = ramp

    def build_base_matrices(self):
        """Builds the pristine static base matrices.

        This method is called exactly once per simulation. It allocates the 
        matrix memory and stamps time-invariant components (Resistors, MNA topology) 
        so they do not have to be repeatedly restamped in transient/sweep loops.

        Returns:
            tuple: (Y_base, sources_base) where Y_base is a mutable scipy.sparse.lil_matrix
            and sources_base is a 1D numpy array.
        """
        dtype = complex if self.is_complex else float
        
        # Initialize empty List-of-Lists (LIL) matrix for fast structural modifications
        Y_base = lil_matrix((self.circuit.total_dim, self.circuit.total_dim), dtype=dtype)
        sources_base = np.zeros(self.circuit.total_dim, dtype=dtype)

        for comp in self.circuit.components:
            comp.stamp_mna_connection(Y_base)
            comp.stamp_static(Y_base, sources_base)
            
        return Y_base, sources_base

    def compute_dc_bias(self, Y_base_lil, sources_base, v_ini=None, print_stuff=True):
        """Calculates the DC Operating Point of the circuit.

        Treats all capacitors as open circuits and inductors as short circuits.
        Invokes the cascading NonlinearSolver if diodes or transistors are present.

        Args:
            Y_base_lil (scipy.sparse.lil_matrix): The static base admittance matrix.
            sources_base (np.ndarray): The static base RHS current/voltage vector.
            v_ini (np.ndarray, optional): An initial guess vector to speed up NR 
                convergence. Crucial for fast DC sweeps. Defaults to None (0V).
            print_stuff (bool, optional): Toggles console convergence logging. 
                Defaults to True.

        Returns:
            tuple: (lu_factorization, VI_solution_array) representing the steady
            state of the circuit.
        """
        # Create a fresh copy of the base topology
        Y_dc = Y_base_lil.copy()
        sources_dc = sources_base.copy()
        
        # Stamp t=0 / steady-state DC source values
        for comp in self.circuit.components:
            comp.stamp_dc(Y_dc, sources_dc)
        
        if self.is_nonlinear:
            # If no guess is provided, start at 0V
            initial_guess = v_ini if v_ini is not None else np.zeros(self.circuit.total_dim)
            
            # Delegate to the cascading Newton-Raphson solver
            solver = NonlinearSolver(self.circuit, print_stuff=print_stuff)
            return solver.solve(Y_dc, sources_dc, initial_guess)
            
        # For purely linear circuits, convert to CSC and solve immediately
        return solve_linear_circuit(Y_dc.tocsc(), sources_dc)
