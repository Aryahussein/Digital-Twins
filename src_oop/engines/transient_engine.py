"""
Transient Analysis Engine Module.

This module performs time-domain numerical integration (.TRAN). It marches 
forward in time, updating dynamic components (Capacitors, Inductors) using 
methods like Backward Euler, and evaluates time-varying independent sources 
(PULSE, SIN, PWL).
"""

import numpy as np
from engines.solver import solve_linear_circuit, NonlinearSolver

class TransientEngine:
    """Evaluates the time-domain response of the circuit.

    Attributes:
        circuit (Circuit): The main circuit object containing components.
        is_nonlinear (bool): Flag indicating if Newton-Raphson solvers are needed.
        ramp (int): Legacy source-stepping parameter.
    """

    def __init__(self, circuit, is_nonlinear, ramp=1):
        """Initializes the Transient Engine.

        Args:
            circuit (Circuit): The populated circuit object.
            is_nonlinear (bool): Boolean flag denoting presence of nonlinear devices.
            ramp (int, optional): Source ramping steps. Defaults to 1.
        """
        self.circuit = circuit
        self.is_nonlinear = is_nonlinear
        self.ramp = ramp

    def _solve_single_step(self, Y_base_lil, sources_base, t, dt, v_prev, nonlinear_solver=None):
        """Evaluates the circuit equations for a single discrete time step.

        Args:
            Y_base_lil (scipy.sparse.lil_matrix): The pristine static base matrix.
            sources_base (np.ndarray): The static base RHS vector.
            t (float): The current simulation time in seconds.
            dt (float): The time step size in seconds.
            v_prev (np.ndarray): The finalized voltage solution from the previous step.
            nonlinear_solver (NonlinearSolver, optional): A pre-instantiated solver 
                object to prevent reallocation overhead.

        Returns:
            tuple: (lu_factorization, VI_solution_array) for this specific time `t`.
        """
        # Start with a clean slate of the static topology
        Y_step = Y_base_lil.copy()
        sources_step = sources_base.copy()
        
        # Ask polymorphic components to stamp their C/dt terms and time-varying waveforms
        for comp in self.circuit.components:
            comp.stamp_transient(Y_step, sources_step, t, dt, v_prev)
            
        if self.is_nonlinear:
            # We use the previous timestep's result as an almost-perfect initial guess!
            return nonlinear_solver.solve(Y_step, sources_step, v_ini=v_prev)

        # Convert to Compressed Sparse Column format right before the linear solve
        return solve_linear_circuit(Y_step.tocsc(), sources_step)

    def run(self, Y_base_lil, sources_base, v_initial, t_stop, dt, keep_lus=False):
        """Executes the forward transient integration loop.

        Args:
            Y_base_lil (scipy.sparse.lil_matrix): The static base admittance matrix.
            sources_base (np.ndarray): The static base RHS vector.
            v_initial (np.ndarray): The t=0 starting bias point (from DCEngine).
            t_stop (float): The final simulation time in seconds.
            dt (float): The discrete time step in seconds.
            keep_lus (bool, optional): If True, caches every LU factorization for 
                use in backward Adjoint passes. Defaults to False.

        Returns:
            tuple: (time_array, VIs, list_of_lus) containing the 1D time axis, 
            the 2D solution matrix over time, and the cached matrix factorizations.
        """
        # Safely create the time array ensuring the final point is included
        time_array = np.arange(0, t_stop + (dt / 10.0), dt)
        results = np.zeros((len(time_array), self.circuit.total_dim))
        list_of_lus = []
        
        # Instantiate the solver ONCE before the loop to maximize performance
        solver = None
        if self.is_nonlinear:
            solver = NonlinearSolver(self.circuit, print_stuff=False)
        
        v_prev = v_initial
        
        print(f"\n--- Starting Transient Analysis ({len(time_array)} steps) ---")
        
        for step, t in enumerate(time_array):
            if step % max(1, len(time_array)//10) == 0: 
                print(f"Solving forward time {t:.3e} s")
                
            lu, VI = self._solve_single_step(
                Y_base_lil, sources_base, t, dt, v_prev, nonlinear_solver=solver
            )
            
            # Store the results
            results[step, :] = VI
            v_prev = VI
            
            if keep_lus: 
                list_of_lus.append(lu)
                
        return time_array, results, list_of_lus
