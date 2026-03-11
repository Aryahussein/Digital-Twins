"""
Adjoint Sensitivity Engine Module.

This module implements the Adjoint Method for calculating exact sensitivities 
for Transient, AC, DC Sweep, and Operating Point analyses.
"""

import numpy as np

class AdjointEngine:
    """Evaluates parameter sensitivities using Adjoint network methods.

    Attributes:
        circuit (Circuit): The main circuit object containing components.
        output_nodes (list): A list of string node names specifying the objective
            measurements (e.g., ['out1', 'out2']).
    """

    def __init__(self, circuit, output_nodes):
        self.circuit = circuit
        self.output_nodes = output_nodes if output_nodes else list(circuit.node_map.keys())

    def _solve_adjoint(self, lu, target, base_rhs=None, is_complex=False):
        """Solves the transposed matrix equation for Adjoint sensitivity analysis.

        Injects a mathematical impulse (1.0) at the target output node and solves 
        the system backward using the cached forward LU factorization.

        Args:
            lu (scipy.sparse.linalg.SuperLU): Cached LU factorization from the forward pass.
            target (str or int, optional): The name/ID of the objective output node.
            base_rhs (np.ndarray, optional): Existing RHS history vector (used in TRAN).
            is_complex (bool, optional): Sets vector dtype for AC analysis. Defaults to False.

        Returns:
            np.ndarray: The adjoint state vector (Psi) for the given step.
        """
        # 1. Initialize the Right-Hand Side (RHS) vector
        if base_rhs is not None:
            d = base_rhs  # Use the pre-built transient history vector
        else:
            dtype = complex if is_complex else float
            d = np.zeros(self.circuit.total_dim, dtype=dtype)
        
        # 2. Inject the objective impulse
        if target is not None:
            idx = self.circuit.get_idx(target)
            if idx is not None:
                d[idx] += 1.0
            
        # 3. Solve transposed
        return lu.solve(d, trans='T')

    # ====================================================
    # TRANSIENT SENSITIVITY (Going Back in Time)
    # ====================================================
    def compute_transient(self, time_array, V_forward, list_of_lus, dt, method='BE'):
        num_steps = len(time_array)
        all_integrated = {}
        all_time_series = {}

        for target_node in self.output_nodes:
            v_hat_next = np.zeros(self.circuit.total_dim)
            adjoint_history = []
            adjoint_state = {}

            print(f"\n--- Starting Backward Adjoint Pass for '{target_node}' ---")
            
            # BACKWARD PASS
            for i in reversed(range(num_steps)):
                J_adj = np.zeros(self.circuit.total_dim)
                
                # Ask components to build the RHS history
                for comp in self.circuit.components:
                    comp.build_adjoint_history(J_adj, dt, v_hat_next, adjoint_state, method=method)

                # The impulse is ONLY applied at the very last time step!
                target_impulse = target_node if i == num_steps - 1 else None
                
                # --- DRY Transposed Solve ---
                v_hat = self._solve_adjoint(list_of_lus[i], target_impulse, base_rhs=J_adj)
                
                adjoint_history.append(v_hat)

                # Update component memory states
                for comp in self.circuit.components:
                    comp.update_adjoint_state(dt, v_hat_next, v_hat, adjoint_state, method=method)
                v_hat_next = v_hat

            adjoint_history.reverse()

            # FORWARD INTEGRATION PASS
            print(f"--- Integrating Sensitivities for '{target_node}' ---")
            total_sens = {}
            time_series_sens = {}

            for i in range(num_steps):
                vi_step = V_forward[i]
                v_hat_step = adjoint_history[i]
                vi_prev = V_forward[i-1] if i > 0 else vi_step 

                for comp in self.circuit.components:
                    step_sens = comp.get_sensitivities(VI=vi_step, PsiPhi=v_hat_step, dt=dt, V_prev=vi_prev)
                    for param, val in step_sens.items():
                        total_sens[param] = total_sens.get(param, 0.0) + (val * dt)
                        if param not in time_series_sens: time_series_sens[param] = []
                        time_series_sens[param].append(val)

            for param in time_series_sens:
                time_series_sens[param] = np.array(time_series_sens[param])

            all_integrated[target_node] = total_sens
            all_time_series[target_node] = time_series_sens

        return {"Integrated_Transient": all_integrated, "Time_Series": all_time_series}

    # ====================================================
    # STEADY-STATE SENSITIVITY (.DC, .OP, and .AC)
    # ====================================================
    def compute_sweep(self, V_forward, list_of_lus, freq_array=None):
        """Calculates Adjoint sensitivities for steady-state analyses (DC, OP, AC).

        Args:
            V_forward (np.ndarray): The 2D solution matrix from the forward run.
            list_of_lus (list): Cached LU factorizations from the forward pass.
            freq_array (np.ndarray, optional): Array of frequencies for AC analysis. 
                If None, the engine defaults to a real-valued DC/OP analysis.
        """
        all_series = {}
        num_steps = V_forward.shape[0]
        
        # Toggle switch based on the presence of a frequency array
        is_ac = freq_array is not None

        for target_node in self.output_nodes:
            target_series = {}
            
            for i in range(num_steps):
                # 1. Determine angular frequency and complex mode
                w = 2 * np.pi * freq_array[i] if is_ac else 0.0
                
                # 2. DRY Transposed Solve
                psi = self._solve_adjoint(list_of_lus[i], target_node, is_complex=is_ac)

                # 3. Calculate Gradients
                for comp in self.circuit.components:
                    # w defaults to 0.0 for DC, so physics equations adapt automatically
                    step_sens = comp.get_sensitivities(VI=V_forward[i], PsiPhi=psi, w=w)
                    
                    for param, val in step_sens.items():
                        if param not in target_series: target_series[param] = []
                        target_series[param].append(val)

            # 4. Convert to numpy arrays for fast plotting
            for param in target_series:
                target_series[param] = np.array(target_series[param])
            all_series[target_node] = target_series

        return all_series
