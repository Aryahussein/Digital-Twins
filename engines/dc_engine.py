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
    """

    def __init__(self, circuit, is_complex, is_nonlinear):
        """Initializes the DC Engine.

        Args:
            circuit (Circuit): The fully populated circuit object.
            is_complex (bool): Whether the base matrix requires complex dtype.
            is_nonlinear (bool): Whether the circuit requires a nonlinear solver.
        """
        self.circuit = circuit
        self.is_complex = is_complex
        self.is_nonlinear = is_nonlinear

    def build_base_matrices(self):
        """Builds the pristine static base matrices.

        This method is called exactly once per simulation. It allocates the 
        matrix memory and stamps time-invariant components (Resistors, MNA topology) 
        so they do not have to be repeatedly restamped in transient/sweep loops.

        Returns:
            scipy.sparse.lil_matrix: Y_base, the mutable base admittance matrix.
        """
        dtype = complex if self.is_complex else float
        
        Y_base = lil_matrix((self.circuit.total_dim, self.circuit.total_dim), dtype=dtype)

        for comp in self.circuit.components:
            comp.stamp_mna_connection(Y_base)
            comp.stamp_static(Y_base)
            
        return Y_base

    def compute_dc_bias(self, Y_base_lil, v_ini=None, print_stuff=True):
        """Calculates the DC Operating Point of the circuit.

        Generates a clean RHS vector from scratch, treats capacitors as open 
        circuits and inductors as short circuits, and invokes the NR solver if needed.

        Args:
            Y_base_lil (scipy.sparse.lil_matrix): The static base admittance matrix.
            v_ini (np.ndarray, optional): An initial guess vector to speed up NR 
                convergence. Crucial for fast DC sweeps. Defaults to None (0V).
            print_stuff (bool, optional): Toggles console convergence logging.

        Returns:
            tuple: (lu_factorization, VI_solution_array)
        """
        # 1. Fresh copy of the base topology
        Y_dc = Y_base_lil.copy()
        
        # 2. Fresh RHS source vector
        dtype = complex if self.is_complex else float
        sources_dc = np.zeros(self.circuit.total_dim, dtype=dtype)
        
        # 3. Stamp DC source values into the clean vector
        for comp in self.circuit.components:
            comp.stamp_dc(Y_dc, sources_dc)
        
        # 4. Solve
        if self.is_nonlinear:
            initial_guess = v_ini if v_ini is not None else np.zeros(self.circuit.total_dim)
            solver = NonlinearSolver(self.circuit, print_stuff=print_stuff)
            return solver.solve(Y_dc, sources_dc, initial_guess)
            
        return solve_linear_circuit(Y_dc.tocsc(), sources_dc)

    def compute_dc_sweep(self, Y_base_lil, source_name, start, stop, step, keep_lus=False):
        """Executes a large-signal DC sweep (.DC analysis).

        Args:
            Y_base_lil (scipy.sparse.lil_matrix): The static base admittance matrix.
            source_name (str): The netlist name of the component to sweep (e.g., 'V1').
            start (float): The starting value of the sweep.
            stop (float): The stopping value of the sweep.
            step (float): The increment step size.
            keep_lus (bool, optional): If True, stores the LU factorization for each 
                sweep step. Defaults to False.

        Returns:
            tuple: (sweep_axis, VIs, list_of_lus)
        """
        try:
            target_comp = self.circuit.get_component(source_name)
        except KeyError:
            available = [c.name for c in self.circuit.components 
                         if c.data.get("type", "").upper() in ("V", "I")]
            raise ValueError(
                f"DC sweep source '{source_name}' not found in circuit. "
                f"Available sources: {', '.join(available) if available else 'none'}"
            )
        sweep_axis = np.arange(start, stop + (step / 10.0), step)
        VIs, list_of_lus = [], []
        
        print(f"\n--- Starting DC Sweep ({len(sweep_axis)} points) ---")

        original_value = target_comp.value
        current_guess = np.zeros(self.circuit.total_dim)

        for idx, val in enumerate(sweep_axis):
            if idx % max(1, len(sweep_axis)//10) == 0: 
                print(f"Solving DC sweep point: {val:.3f}")

            target_comp.value = val
            
            lu_dc, VI_dc = self.compute_dc_bias(
                Y_base_lil, 
                v_ini=current_guess, 
                print_stuff=False
            )
            
            current_guess = VI_dc.copy() 
            VIs.append(VI_dc)
            if keep_lus: 
                list_of_lus.append(lu_dc)

        # Restore the component to its original state
        target_comp.value = original_value

        return sweep_axis, np.array(VIs), list_of_lus
