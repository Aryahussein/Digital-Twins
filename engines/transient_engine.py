"""
Transient Analysis Engine Module.

This module performs time-domain numerical integration (.TRAN). It supports
Backward Euler (BE) and Trapezoidal (TR) integration methods. BE is first-order
accurate and unconditionally stable. TR is second-order accurate but may exhibit
ringing on stiff circuits.
"""

import numpy as np
from engines.solver import solve_linear_circuit, solve_lu, NonlinearSolver


class TransientEngine:
    """Evaluates the time-domain response of the circuit.

    Attributes:
        circuit (Circuit): The main circuit object containing components.
        is_nonlinear (bool): Flag indicating if Newton-Raphson solvers are needed.
        method (str): Integration method — 'BE' or 'TR'.
    """

    def __init__(self, circuit, is_nonlinear, method='BE'):
        """Initializes the Transient Engine.

        Args:
            circuit (Circuit): The populated circuit object.
            is_nonlinear (bool): Boolean flag denoting presence of nonlinear devices.
            method (str): Integration method ('BE' or 'TR'). Defaults to 'BE'.
        """
        self.circuit = circuit
        self.is_nonlinear = is_nonlinear
        self.method = method.upper()
        
        if self.method not in ('BE', 'TR'):
            raise ValueError(f"Unknown integration method '{method}'. Use 'BE' or 'TR'.")

    def _solve_single_step(self, Y_base_lil, t, dt, v_prev, nonlinear_solver=None):
        """Evaluates the circuit equations for a single discrete time step.

        Args:
            Y_base_lil (scipy.sparse.lil_matrix): The pristine static base matrix.
            t (float): The current simulation time in seconds.
            dt (float): The time step size in seconds.
            v_prev (np.ndarray): The finalized voltage solution from the previous step.
            nonlinear_solver (NonlinearSolver, optional): Pre-instantiated NR solver.

        Returns:
            tuple: (lu_factorization, VI_solution_array) for this time step.
        """
        Y_step = Y_base_lil.copy()
        sources_step = np.zeros(self.circuit.total_dim)
        
        for comp in self.circuit.components:
            comp.stamp_transient(Y_step, sources_step, t, dt, v_prev, method=self.method)
            
        if self.is_nonlinear:
            return nonlinear_solver.solve(Y_step, sources_step, v_ini=v_prev)

        return solve_linear_circuit(Y_step.tocsc(), sources_step)

    def run(self, Y_base_lil, v_initial, t_stop, dt, keep_lus=False):
        """Executes the forward transient integration loop.

        The DC operating point (v_initial) is stored as the t=0 result, then
        time-stepping begins at t=dt.

        Args:
            Y_base_lil (scipy.sparse.lil_matrix): The static base admittance matrix.
            v_initial (np.ndarray): The t=0 starting bias point (from DCEngine).
            t_stop (float): The final simulation time in seconds.
            dt (float): The discrete time step in seconds.
            keep_lus (bool, optional): Cache LU factorizations for Adjoint. Defaults to False.

        Returns:
            tuple: (time_array, VIs, list_of_lus)
        """
        time_array = np.arange(0, t_stop + (dt / 10.0), dt)
        results = np.zeros((len(time_array), self.circuit.total_dim))
        list_of_lus = []
        
        solver = None
        if self.is_nonlinear:
            solver = NonlinearSolver(self.circuit, print_stuff=False)
        
        # Reset any stored TR state from previous runs
        for comp in self.circuit.components:
            comp.reset_transient_state()

        # Store DC operating point as t=0
        results[0, :] = v_initial
        v_prev = v_initial
        
        if keep_lus:
            Y_t0 = Y_base_lil.copy()
            sources_t0 = np.zeros(self.circuit.total_dim)
            for comp in self.circuit.components:
                comp.stamp_transient(Y_t0, sources_t0, 0.0, dt, v_prev, method=self.method)
            if self.is_nonlinear:
                for comp in self.circuit.components:
                    comp.stamp_nonlinear(Y_t0, sources_t0, v_prev, v_prev)
            list_of_lus.append(solve_lu(Y_t0.tocsc()))
        
        print(f"\n--- Starting Transient Analysis ({len(time_array)} steps, method={self.method}) ---")
        
        for step in range(1, len(time_array)):
            t = time_array[step]
            
            if step % max(1, len(time_array)//10) == 0: 
                print(f"Solving forward time {t:.3e} s")
                
            lu, VI = self._solve_single_step(
                Y_base_lil, t, dt, v_prev, nonlinear_solver=solver
            )
            
            # Update TR companion model state BEFORE moving to next step
            for comp in self.circuit.components:
                comp.post_step_update(VI, v_prev, dt, method=self.method)
            
            results[step, :] = VI
            v_prev = VI
            
            if keep_lus: 
                list_of_lus.append(lu)
                
        return time_array, results, list_of_lus
