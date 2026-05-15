"""Transient Analysis Engine Module.

This module performs time-domain numerical integration (.TRAN). It marches 
forward in time, updating dynamic components (Capacitors, Inductors) using 
methods like Backward Euler or the Trapezoidal Rule, and evaluates time-varying 
independent sources (PULSE, SIN, PWL).
"""

import numpy as np
from engines.solver import solve_linear_circuit, NonlinearSolver
from engines.dc_engine import DCEngine

class TransientEngine:
    """Evaluates the time-domain response of the circuit.

    Attributes:
        circuit (Circuit): The main circuit orchestrator.
        is_nonlinear (bool): Flag indicating if Newton-Raphson solvers are needed.
    """

    def __init__(self, circuit):
        """Initializes the Transient Engine.

        Args:
            circuit (Circuit): The populated circuit orchestrator.
        """
        self.circuit = circuit
        self.is_nonlinear = circuit.is_nonlinear
    
    def compute(self, t_stop, dt, method='TR', keep_lus=False, initial_conditions=None):
        """Executes the forward transient integration loop.

        Args:
            t_stop (float): The final simulation time in seconds.
            dt (float): The discrete time step in seconds.
            method (str, optional): The numerical integration method ('TR' for 
                Trapezoidal Rule or 'BE' for Backward Euler). Defaults to 'TR'.
            keep_lus (bool, optional): If True, caches every LU factorization for 
                use in backward Adjoint sensitivity passes. Defaults to False.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray, list]: A tuple containing the 
            1D time axis, the 2D solution matrix over time, and the cached 
            matrix factorizations.
        """
        # Safely create the time array ensuring the final point is included
        time_array = np.arange(0, t_stop + (dt / 10.0), dt)
        results = np.zeros((len(time_array), self.circuit.total_dim))
        list_of_lus = []
        
        print(f"\n--- Starting Transient Analysis ({len(time_array)} steps) ---")

        # 1. Calculate t=0 Initial Conditions (DC Bias Point)
        dc_engine = DCEngine(self.circuit)
        lu_prev, v_prev = dc_engine.compute_dc_bias(print_stuff=False)

        if initial_conditions is not None:
            for node, forced_voltage in initial_conditions.items():
                node_idx = self.circuit.get_idx(node)
                if node_idx is not None:
                    v_prev[node_idx] = forced_voltage
                    print(f"Applying Initial Condition: V({node}) = {forced_voltage}V")
        
        results[0, :] = v_prev
        if keep_lus:
            list_of_lus.append(lu_prev)
        
        solver = None
        if self.is_nonlinear:
            solver = NonlinearSolver(self.circuit, print_stuff=False)
        
        # 2. Time Integration Loop
        for step in range(1, len(time_array)):
            t = time_array[step]
            
            if step % max(1, len(time_array)//10) == 0: 
                print(f"Solving forward time {t:.3e} s")
                
            # Delegate to the nonlinear solver or direct linear solver
            if self.is_nonlinear:
                # Use the previous timestep as an almost-perfect initial guess (v_ini)
                lu, v_t = solver.solve(
                    v_ini=v_prev, 
                    domain="time", 
                    t=t, 
                    dt=dt, 
                    v_prev=v_prev, 
                    method=method
                )
            else:
                Y_step, J_step = self.circuit.build_system(
                    domain="time", t=t, dt=dt, v_prev=v_prev, method=method
                )
                lu, v_t = solve_linear_circuit(Y_step.tocsc(), J_step)
            
            # Store the converged results
            results[step, :] = v_t

            # 3. Update internal history states for dynamic components
            # This is crucial for TR/BE methods to stash I_prev for the companion models!
            for comp in self.circuit._tran_comps:
                if hasattr(comp, 'update_transient_state'):
                    comp.update_transient_state(v_t, v_prev=v_prev, dt=dt, method=method)

            # Advance the state vector
            v_prev = v_t.copy()
            
            if keep_lus: 
                list_of_lus.append(lu)

        # print(list_of_lus)
                
        return time_array, results, list_of_lus
