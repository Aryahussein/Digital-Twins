"""
Adjoint Sensitivity Engine Module.

This module implements the Adjoint Method for calculating exact sensitivities 
for Transient, AC, DC Sweep, and Operating Point analyses. The adjoint method
provides all parameter sensitivities simultaneously by solving one additional
transposed linear system per output node.
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
        
        # Create a completely NEW list to prevent infinite loop appending
        resolved_nodes = [] 
        
        # Smart resolution to ensure dictionary keys match
        if output_nodes:
            for n in output_nodes:
                if n is None:
                    continue
                if n in circuit.node_map:
                    resolved_nodes.append(n)
                elif str(n).isdigit() and int(n) in circuit.node_map:
                    resolved_nodes.append(int(n))
                elif str(n) in circuit.node_map:
                    resolved_nodes.append(str(n))
                else:
                    resolved_nodes.append(n)  # Fallback
        
        # If no valid nodes specified, default to all nodes
        if not resolved_nodes:
            resolved_nodes = list(circuit.node_map.keys())
            
        self.output_nodes = resolved_nodes

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
        if base_rhs is not None:
            d = base_rhs
        else:
            dtype = complex if is_complex else float
            d = np.zeros(self.circuit.total_dim, dtype=dtype)
        
        # Inject the objective impulse
        if target is not None:
            idx = self.circuit.get_idx(target)
            if idx is not None:
                d[idx] += 1.0
            
        return lu.solve(d, trans='T')

    # ====================================================
    # TRANSIENT SENSITIVITY (Going Back in Time)
    # ====================================================
    def compute_transient(self, time_array, V_forward, list_of_lus, dt, method='BE'):
        """Computes transient sensitivities using backward-in-time adjoint sweep.
        
        Args:
            time_array (np.ndarray): The forward time axis array.
            V_forward (np.ndarray): The 2D forward solution matrix [steps x nodes].
            list_of_lus (list): Cached LU factorizations from the forward pass.
            dt (float): The time step size.
            method (str): Integration method ('BE' for Backward Euler).
            
        Returns:
            dict: {"Integrated_Transient": {node: {param: scalar}},
                   "Time_Series": {node: {param: array}}}
        """
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
                
                for comp in self.circuit.components:
                    comp.build_adjoint_history(J_adj, dt, v_hat_next, adjoint_state, method=method)

                # The impulse is ONLY applied at the very last time step
                target_impulse = target_node if i == num_steps - 1 else None
                
                v_hat = self._solve_adjoint(list_of_lus[i], target_impulse, base_rhs=J_adj)
                
                adjoint_history.append(v_hat)

                for comp in self.circuit.components:
                    comp.update_adjoint_state(dt, v_hat_next, v_hat, adjoint_state, method=method)
                v_hat_next = v_hat

            adjoint_history.reverse()

            # FORWARD INTEGRATION PASS
            print(f"--- Integrating Sensitivities for '{target_node}' ---")
            time_series_sens = {}

            for i in range(num_steps):
                vi_step = V_forward[i]
                v_hat_step = adjoint_history[i]
                vi_prev = V_forward[i-1] if i > 0 else vi_step 

                for comp in self.circuit.components:
                    step_sens = comp.get_sensitivities(
                        VI=vi_step, PsiPhi=v_hat_step, dt=dt, V_prev=vi_prev, method=method
                    )
                    for param, val in step_sens.items():
                        if param not in time_series_sens: 
                            time_series_sens[param] = []
                        time_series_sens[param].append(val)

            # Integrate the time series: plain sum for both methods.
            # The adjoint sensitivity dJ/dp = sum_k ψ_k · ∂F_k/∂p is EXACT for the 
            # discrete system. It is NOT a numerical integral of a continuous function,
            # so no quadrature weighting is needed. The accuracy improvement from TR 
            # comes through the more accurate forward voltages V and adjoint vectors ψ 
            # (both O(dt²)), not through integration quadrature.
            total_sens = {}
            for param in time_series_sens:
                series = np.array(time_series_sens[param])
                time_series_sens[param] = series
                total_sens[param] = np.sum(series)

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
                
        Returns:
            dict: {node: {param: array_or_scalar}} mapping output nodes to their
                  per-parameter sensitivity values.
        """
        all_series = {}
        num_steps = V_forward.shape[0]
        
        is_ac = freq_array is not None

        for target_node in self.output_nodes:
            target_series = {}
            
            for i in range(num_steps):
                w = 2 * np.pi * freq_array[i] if is_ac else 0.0
                
                psi = self._solve_adjoint(list_of_lus[i], target_node, is_complex=is_ac)

                for comp in self.circuit.components:
                    step_sens = comp.get_sensitivities(VI=V_forward[i], PsiPhi=psi, w=w, method='BE')
                    
                    for param, val in step_sens.items():
                        if param not in target_series: 
                            target_series[param] = []
                        target_series[param].append(val)

            for param in target_series:
                target_series[param] = np.array(target_series[param])
            all_series[target_node] = target_series

        return all_series
