import numpy as np
from core.results import SensitivityData


class AdjointEngine:
    """Evaluates parameter sensitivities using Adjoint network methods.

    Attributes:
        circuit (Circuit): The main circuit object containing components.
        output_nodes (list): A list of string node names specifying the objective
            measurements (e.g., ['out1', 'out2']).
    """

    def __init__(self, circuit, output_nodes):
        self.circuit = circuit

        resolved_nodes = []

        # Smart resolution to ensure dictionary keys match
        if output_nodes:
            for n in output_nodes:
                if n in circuit.node_map:
                    resolved_nodes.append(n)
                elif str(n).isdigit() and int(n) in circuit.node_map:
                    resolved_nodes.append(int(n))
                elif str(n) in circuit.node_map:
                    resolved_nodes.append(str(n))
                else:
                    resolved_nodes.append(n)  # Fallback
        else:
            resolved_nodes = list(circuit.node_map.keys())

        self.output_nodes = resolved_nodes

    def _solve_adjoint(self, lu, target, base_rhs=None, is_complex=False):
        """Solves the transposed matrix equation for Adjoint sensitivity analysis."""
        if base_rhs is not None:
            d = base_rhs
        else:
            dtype = complex if is_complex else float
            d = np.zeros(self.circuit.total_dim, dtype=dtype)

        if target is not None:
            idx = self.circuit.get_idx(target)
            if idx is not None:
                d[idx] += 1.0

        return lu.solve(d, trans="T")

    # ====================================================
    # TRANSIENT SENSITIVITY (Global / Backward Propagation)
    # ====================================================
    def compute_transient(self, time_array, V_forward, list_of_lus, dt, method="BE"):
        """Calculates exact Global Adjoint sensitivities via backward time integration.

        This performs a true reverse-time simulation to calculate the total
        integrated sensitivity of a node across the entire transient window.
        The backward time-series is packed into a standard SensitivityData tensor.
        """
        num_steps = len(time_array)
        param_names = self.circuit.differentiable_params

        # 1. Allocate the unified 3D Tensor for the backward time-series
        tensor = SensitivityData(
            time_array, param_names, self.output_nodes, domain="time"
        )

        # Dictionary to hold the final scalar integrals
        all_integrated = {}

        for target_node in self.output_nodes:
            o_idx = tensor.output_index[target_node]
            v_hat_next = np.zeros(self.circuit.total_dim)
            adjoint_history = []

            print(f"\n--- Starting Backward Adjoint Pass for '{target_node}' ---")

            # 1. BACKWARD PASS (Unchanged)
            for i in reversed(range(num_steps)):
                J_adj = np.zeros(self.circuit.total_dim)

                for comp in self.circuit.components:
                    comp.build_adjoint_history(J_adj, dt, v_hat_next, method=method)

                target_impulse = target_node if i == num_steps - 1 else None

                v_hat = self._solve_adjoint(
                    list_of_lus[i], target_impulse, base_rhs=J_adj
                )
                adjoint_history.append(v_hat)
                v_hat_next = v_hat
                # print(f"J_adj for step {i}: {J_adj}")
                # print(f"v_hat for step {i}: {v_hat}")
            adjoint_history.reverse()

            # 2. FORWARD INTEGRATION PASS
            print(f"--- Integrating Sensitivities for '{target_node}' ---")
            total_sens = {param: 0.0 for param in param_names}
            for i in range(num_steps):
                vi_step = V_forward[i]
                v_hat_step = adjoint_history[i]
                vi_prev = V_forward[i - 1] if i > 0 else vi_step
                for comp in self.circuit.components:
                    step_sens = comp.get_sensitivities(
                        VI=vi_step,
                        PsiPhi=v_hat_step,
                        dt=dt,
                        V_prev=vi_prev,
                        method=method,
                    )
                    for param, val in step_sens.items():

                        p_idx = tensor.param_index[param]

                        # Populate the 3D Tensor directly
                        tensor.data[p_idx, o_idx, i] = val

                        # Accumulate the running scalar integral
                        total_sens[param] += val * dt

            all_integrated[target_node] = total_sens

        # Return the flat dictionary of integrals AND the fully structured Tensor
        return {
            "Integrated_Transient": all_integrated,
            "Time_Series": tensor,
            "Raw_Adjoint_History": adjoint_history,
        }

    # ====================================================
    # UNIFIED SENSITIVITY SOLVER (Continuous/Steady-State)
    # ====================================================
    def compute_sensitivities(
        self, sweep_axis, V_forward, list_of_lus, domain="static", method="TR", dt=0.0
    ):
        """Calculates Adjoint sensitivities for all continuous analysis types.

        Unifies Transient (Local DC), AC, DC, and OP analyses into a single
        execution loop. The underlying components automatically use the appropriate
        physics variables (w vs dt) based on the provided arguments.
        """
        is_ac = domain == "frequency"
        is_tran = domain == "time"
        num_steps = len(sweep_axis)

        # 1. Ask the circuit directly for its parameters! (Clean OOP)
        param_names = self.circuit.differentiable_params

        # 2. Allocate the 3D Tensor
        tensor = SensitivityData(
            sweep_axis, param_names, self.output_nodes, domain=domain
        )

        # 3. Allocate storage for raw adjoint vectors (needed for fault analysis)
        # Shape: (n_outputs, n_sweep_steps, total_dim)
        dtype = complex if is_ac else float
        tensor.adjoint_vectors = np.zeros(
            (len(self.output_nodes), num_steps, self.circuit.total_dim), dtype=dtype
        )

        for target_node in self.output_nodes:
            print(
                f"--- Building Sensitivity Tensor for {domain.upper()} Node '{target_node}' ---"
            )
            o_idx = tensor.output_index[target_node]

            for i in range(num_steps):
                # Physics Context
                w = 2 * np.pi * sweep_axis[i] if is_ac else 0.0
                vi_step = V_forward[i]
                vi_prev = V_forward[i - 1] if (is_tran and i > 0) else vi_step

                # Transposed Solve
                psi = self._solve_adjoint(list_of_lus[i], target_node, is_complex=is_ac)
                
                # Store the raw adjoint vector for fault analysis
                tensor.adjoint_vectors[o_idx, i, :] = psi
                # Calculate Gradients
                for comp in self.circuit.components:
                    step_sens = comp.get_sensitivities(
                        VI=vi_step,
                        PsiPhi=psi,
                        w=w,
                        dt=dt,
                        V_prev=vi_prev,
                        method=method,
                    )

                    # Populate the tensor
                    for param, val in step_sens.items():
                        p_idx = tensor.param_index[param]
                        tensor.data[p_idx, o_idx, i] = val

        return tensor
