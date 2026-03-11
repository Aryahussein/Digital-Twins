"""
Adjoint Sensitivity Engine Module.

This module implements the Adjoint Method for calculating exact transient 
sensitivities. It utilizes the factorized LU matrices from the forward transient 
simulation to run a time-reversed simulation, yielding the gradient of the 
objective function with respect to all circuit parameters simultaneously.
"""

import numpy as np

class AdjointEngine:
    """Evaluates transient parameter sensitivities using backward time integration.

    Attributes:
        circuit (Circuit): The main circuit object containing components and topology.
        output_nodes (list): A list of string node names specifying the objective
            measurements (e.g., ['out1', 'out2']).
    """

    def __init__(self, circuit, output_nodes):
        """Initializes the Adjoint Engine.

        Args:
            circuit (Circuit): The fully populated circuit object.
            output_nodes (list): List of node names to evaluate sensitivities for.
                If None, evaluates against all non-ground nodes in the circuit.
        """
        self.circuit = circuit
        self.output_nodes = output_nodes if output_nodes else list(circuit.node_map.keys())

    def compute_transient(self, time_array, V_forward, list_of_lus, dt, method='BE'):
        """Executes the backward adjoint pass and forward sensitivity integration.

        This method performs a separate backward time-traveling pass for each
        specified output node to guarantee mathematically independent sensitivities.

        Args:
            time_array (np.ndarray): The time steps from the forward transient run.
            V_forward (np.ndarray): The 2D solution matrix from the forward run.
            list_of_lus (list): The cached scipy LU factorization objects from 
                each time step of the forward run.
            dt (float): The transient time step size.
            method (str, optional): The integration method used (e.g., 'BE' for 
                Backward Euler). Defaults to 'BE'.

        Returns:
            dict: A nested dictionary containing both the integrated scalars and 
            the time-series arrays. Format:
            {
                "Integrated_Transient": { 'node_name': { 'param_name': scalar_value } },
                "Time_Series": { 'node_name': { 'param_name': np.ndarray } }
            }
        """
        num_steps = len(time_array)
        
        # Extension Added: Dictionaries to hold results for multiple independent nodes
        all_integrated = {}
        all_time_series = {}

        # 1. Loop over every requested output node independently
        for target_node in self.output_nodes:
            v_hat_next = np.zeros(self.circuit.total_dim)
            adjoint_history = []
            adjoint_state = {}

            print(f"\n--- Starting Backward Adjoint Pass for '{target_node}' ({method}) ---")
            
            # ==========================================
            # BACKWARD PASS (Time Travel)
            # ==========================================
            for i in reversed(range(num_steps)):
                
                # Ask components to build the RHS history
                J_adjoint = np.zeros(self.circuit.total_dim)
                for comp in self.circuit.components:
                    comp.build_adjoint_history(J_adjoint, dt, v_hat_next, adjoint_state, method=method)

                # Add objective excitation (Impulse at t=final)
                if i == num_steps - 1:
                    idx = self.circuit.get_idx(target_node)
                    if idx is not None: 
                        J_adjoint[idx] += 1.0

                # Solve transposed using cached LU
                lu = list_of_lus[i]
                v_hat = lu.solve(J_adjoint, trans='T')
                
                # Performance Fix: Append is O(1). We will reverse it later.
                adjoint_history.append(v_hat)

                # Ask components to update their internal memory states
                for comp in self.circuit.components:
                    comp.update_adjoint_state(dt, v_hat_next, v_hat, adjoint_state, method=method)
                
                v_hat_next = v_hat

            # Reverse the history so index 0 corresponds to t=0
            adjoint_history.reverse()

            # ==========================================
            # FORWARD INTEGRATION PASS
            # ==========================================
            print(f"--- Integrating Sensitivities for '{target_node}' ---")
            total_sens = {}
            time_series_sens = {}

            for i in range(num_steps):
                vi_step = V_forward[i]
                v_hat_step = adjoint_history[i]
                
                # Get V_prev for dV/dt sensitivity formulas
                vi_prev = V_forward[i-1] if i > 0 else vi_step 

                # Ask components for their sensitivity math
                for comp in self.circuit.components:
                    step_sens = comp.get_sensitivities(VI=vi_step, PsiPhi=v_hat_step, dt=dt, V_prev=vi_prev)
                    
                    for param, val in step_sens.items():
                        # Accumulate the integral
                        total_sens[param] = total_sens.get(param, 0.0) + (val * dt)
                        
                        # Store the time-series
                        if param not in time_series_sens:
                            time_series_sens[param] = []
                        time_series_sens[param].append(val)

            # Convert time series lists to numpy arrays
            for param in time_series_sens:
                time_series_sens[param] = np.array(time_series_sens[param])

            # Store results for this specific node
            all_integrated[target_node] = total_sens
            all_time_series[target_node] = time_series_sens

        return {"Integrated_Transient": all_integrated, "Time_Series": all_time_series}
