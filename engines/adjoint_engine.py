"""
Adjoint Sensitivity Engine Module.

Implements the Adjoint Method for Transient, AC, DC Sweep, and Operating Point
sensitivity analysis.

Transient Sensitivity
---------------------
Computes dV(t_k)/dp at every simulation timestep t_k by running one backward
adjoint sweep per timestep. The result is a true sensitivity waveform aligned
with the simulation time axis - the same length as sweep_axis - so it can be
plotted directly against time with no resampling.

Cost: O(N^2/2) adjoint solves where N is the number of timesteps. Each solve
is a cached triangular solve (very fast). For circuits with many steps (>500),
an automatic stride is applied so at most MAX_OUTPUT_POINTS sweeps are run;
results are then interpolated onto the full time axis.

AC / DC / OP sensitivity uses a single sweep per output node (O(N) total).
"""

import numpy as np

MAX_OUTPUT_POINTS = 500   # cap for very long simulations


class AdjointEngine:
    """Evaluates parameter sensitivities using the Adjoint network method."""

    def __init__(self, circuit, output_nodes):
        self.circuit = circuit

        resolved = []
        if output_nodes:
            for n in output_nodes:
                if n is None:
                    continue
                if n in circuit.node_map:
                    resolved.append(n)
                elif str(n).isdigit() and int(n) in circuit.node_map:
                    resolved.append(int(n))
                elif str(n) in circuit.node_map:
                    resolved.append(str(n))
                else:
                    resolved.append(n)

        if not resolved:
            resolved = list(circuit.node_map.keys())

        self.output_nodes = resolved

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _solve_adjoint(self, lu, target, base_rhs=None, is_complex=False):
        """Solve J^T psi = rhs, injecting a unit impulse at target node."""
        if base_rhs is not None:
            d = base_rhs
        else:
            dtype = complex if is_complex else float
            d = np.zeros(self.circuit.total_dim, dtype=dtype)

        if target is not None:
            idx = self.circuit.get_idx(target)
            if idx is not None:
                d[idx] += 1.0

        return lu.solve(d, trans='T')

    def _backward_sweep(self, obs_idx, target_node, list_of_lus, dt, method):
        """
        Run one backward adjoint sweep with impulse injected at obs_idx.

        Returns adjoint_history[0..obs_idx] in forward time order.
        psi_k is the adjoint state at step k when the objective is V(t_obs).
        """
        v_hat_next  = np.zeros(self.circuit.total_dim)
        adjoint_state = {}
        history = []

        for i in reversed(range(obs_idx + 1)):
            J_adj = np.zeros(self.circuit.total_dim)

            for comp in self.circuit.components:
                comp.build_adjoint_history(
                    J_adj, dt, v_hat_next, adjoint_state, method=method
                )

            impulse_node = target_node if i == obs_idx else None
            v_hat = self._solve_adjoint(list_of_lus[i], impulse_node, base_rhs=J_adj)

            history.append(v_hat)

            for comp in self.circuit.components:
                comp.update_adjoint_state(
                    dt, v_hat_next, v_hat, adjoint_state, method=method
                )
            v_hat_next = v_hat

        history.reverse()   # now in forward order [0..obs_idx]
        return history

    def _accumulate_sens(self, obs_idx, adjoint_history, V_forward, dt, method):
        """
        Sum psi_k * dF_k/dp from k=0 to obs_idx.

        Returns {param: scalar} = dV(t_obs)/dp.
        """
        total = {}
        for i in range(obs_idx + 1):
            vi      = V_forward[i]
            psi     = adjoint_history[i]
            vi_prev = V_forward[i - 1] if i > 0 else vi

            for comp in self.circuit.components:
                s = comp.get_sensitivities(
                    VI=vi, PsiPhi=psi, dt=dt, V_prev=vi_prev, method=method
                )
                for param, val in s.items():
                    total[param] = total.get(param, 0.0) + val

        return total

    # ------------------------------------------------------------------ #
    # Transient sensitivity                                                #
    # ------------------------------------------------------------------ #

    def compute_transient(self, time_array, V_forward, list_of_lus, dt, method='BE'):
        """
        Compute dV(t_k)/dp at every simulation timestep.

        One backward sweep is run per output timestep, giving the true
        sensitivity waveform. The series stored in:

            result.sensitivities["Time_Series"][node][param]

        has the same length as time_array and can be plotted directly against
        result.sweep_axis (the simulation time axis).

        For simulations with more than MAX_OUTPUT_POINTS timesteps, sensitivity
        is computed at a strided subset and linearly interpolated back onto the
        full time axis so the output always has len(time_array) points.

        Returns
        -------
        dict with keys:
            "Integrated_Transient" : {node: {param: scalar}}  dV(T)/dp
            "Time_Series"          : {node: {param: ndarray}} dV(t)/dp
                                     length = len(time_array)
            "Sample_Times"         : None  (kept for API compatibility)
        """
        num_steps = len(time_array)

        # Choose which steps to compute sensitivity at
        if num_steps <= MAX_OUTPUT_POINTS:
            compute_indices = list(range(num_steps))
            stride = 1
        else:
            stride = max(1, num_steps // MAX_OUTPUT_POINTS)
            compute_indices = list(range(0, num_steps, stride))
            if compute_indices[-1] != num_steps - 1:
                compute_indices.append(num_steps - 1)

        n_compute      = len(compute_indices)
        using_stride   = (stride > 1)

        all_integrated = {}
        all_time_series = {}

        for target_node in self.output_nodes:
            print(f"\n--- Per-Timestep Adjoint for '{target_node}' "
                  f"({n_compute} sweeps"
                  + (f", stride={stride}" if using_stride else "")
                  + ") ---")

            sampled_sens = {}   # {param: [val at each compute_index]}

            for sweep_num, obs_idx in enumerate(compute_indices):
                if sweep_num % max(1, n_compute // 10) == 0:
                    print(f"  Sweep {sweep_num + 1}/{n_compute}  "
                          f"(t = {time_array[obs_idx]:.3e} s)")

                adj_hist = self._backward_sweep(
                    obs_idx=obs_idx,
                    target_node=target_node,
                    list_of_lus=list_of_lus,
                    dt=dt,
                    method=method
                )

                sens_at_step = self._accumulate_sens(
                    obs_idx=obs_idx,
                    adjoint_history=adj_hist,
                    V_forward=V_forward,
                    dt=dt,
                    method=method
                )

                for param, val in sens_at_step.items():
                    if param not in sampled_sens:
                        sampled_sens[param] = []
                    sampled_sens[param].append(val)

            # Build full-length arrays aligned with time_array
            compute_times = time_array[compute_indices]
            time_series   = {}

            for param, vals in sampled_sens.items():
                vals_arr = np.array(vals, dtype=float)
                if using_stride:
                    full = np.interp(time_array, compute_times, vals_arr)
                else:
                    full = vals_arr
                time_series[param] = full

            # Final-time scalar = last sample
            integrated = {p: float(v[-1]) for p, v in time_series.items()}

            all_integrated[target_node] = integrated
            all_time_series[target_node] = time_series

        return {
            "Integrated_Transient": all_integrated,
            "Time_Series":          all_time_series,
            "Sample_Times":         None,
        }

    # ------------------------------------------------------------------ #
    # Steady-state (AC, DC, OP)                                           #
    # ------------------------------------------------------------------ #

    def compute_sweep(self, V_forward, list_of_lus, freq_array=None):
        """Adjoint sensitivity for DC sweep, AC, and Operating Point."""
        all_series = {}
        num_steps  = V_forward.shape[0]
        is_ac      = freq_array is not None

        for target_node in self.output_nodes:
            target_series = {}

            for i in range(num_steps):
                w   = 2 * np.pi * freq_array[i] if is_ac else 0.0
                psi = self._solve_adjoint(
                    list_of_lus[i], target_node, is_complex=is_ac
                )

                for comp in self.circuit.components:
                    s = comp.get_sensitivities(
                        VI=V_forward[i], PsiPhi=psi, w=w, method='BE'
                    )
                    for param, val in s.items():
                        if param not in target_series:
                            target_series[param] = []
                        target_series[param].append(val)

            for param in target_series:
                target_series[param] = np.array(target_series[param])
            all_series[target_node] = target_series

        return all_series
