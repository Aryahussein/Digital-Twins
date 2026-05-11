"""Large Change Sensitivity Engine.

This module provides the ultimate unified engine for performing simultaneous,
large-scale parameter sweeps across non-linear circuits. It leverages the
Sherman-Morrison-Woodbury matrix identity injected directly into the core
Newton-Raphson solver to achieve exact mathematical equilibrium without
requiring expensive O(N^1.5) LU refactorizations for every variation step.

Linear and AC circuits converge in exactly 1 iteration (a pure Woodbury step).
Non-Linear circuits cascade Woodbury updates until exact equilibrium is reached.
"""

import numpy as np
from engines.solver import NonlinearSolver, ExplicitWoodburyStrategy
from utils.utils import TemporaryCircuitState

class LargeChangeData:
    """A highly sliceable 3D tensor for large-change sensitivity analysis.

    This class acts as a data vault, storing the absolute node voltages
    resulting from simultaneously scaling multiple circuit parameters. It
    provides a clean, callable interface to extract 1D lines or 2D surfaces
    for plotting without requiring the user to manage raw numpy indices.

    The underlying tensor maintains the shape:
    (Variations × Sweep Steps × Total Nodes)

    Attributes:
        variation_axis (numpy.ndarray): The 1D array of variation labels
            (e.g., scaling multipliers, run indices) applied to the data.
        sweep_axis (numpy.ndarray): The independent variable array from the
            base simulation (e.g., Time in seconds, Frequency in Hertz).
        node_map (dict[str, int]): Maps string node names to matrix column indices.
        data (numpy.ndarray): The complex or real 3D tensor holding the
            calculated absolute voltages.
    """
    
    def __init__(self, variation_axis, sweep_axis, node_map):
        """Initializes the LargeChangeData tensor vault.

        Args:
            variation_axis (numpy.ndarray or list): The variation identifiers.
            sweep_axis (numpy.ndarray or list): The forward simulation's x-axis.
            node_map (dict): The circuit's node coordinate map.
        """
        self.variation_axis = np.atleast_1d(variation_axis)
        self.sweep_axis = np.atleast_1d(sweep_axis)
        self.node_map = node_map
        
        self.data = np.zeros(
            (len(self.variation_axis), len(self.sweep_axis), len(node_map)), 
            dtype=complex
        )
        
    def __call__(self, node, var_idx=None, step_idx=None):
        """Retrieves specific data slices using a flexible Pythonic call.

        Args:
            node (str or int): The target output node to extract data for.
            var_idx (int, optional): The index of the specific variation. 
                Defaults to None.
            step_idx (int, optional): The index of the specific simulation 
                sweep step (time/frequency). Defaults to None.

        Returns:
            numpy.ndarray or float or complex: 
                - If both indices are provided: Returns a single scalar value.
                - If only var_idx is provided: Returns a 1D waveform across the sweep.
                - If only step_idx is provided: Returns a 1D transfer curve across variations.
                - If no indices are provided: Returns a 2D surface matrix (Variations x Sweep).
        """
        n_idx = self.node_map[str(node)]
        
        if var_idx is not None and step_idx is not None: 
            return self.data[var_idx, step_idx, n_idx]
        elif var_idx is not None: 
            return self.data[var_idx, :, n_idx] 
        elif step_idx is not None: 
            return self.data[:, step_idx, n_idx] 
        else: 
            return self.data[:, :, n_idx] 


class LargeChangeEngine:
    """A unified sensitivity engine using Woodbury-accelerated Newton-Raphson.

    This engine executes simultaneous large-change parameter sweeps. By injecting
    the ExplicitWoodburyStrategy into the standard NonlinearSolver, it avoids
    rebuilding and re-factorizing the global MNA matrix for every parameter
    variation, drastically accelerating large change analysis.

    Attributes:
        circuit (Circuit): The initialized main circuit orchestrator.
        param_map (dict): A cached O(1) lookup dictionary mapping parameter
            names to their parent Component objects, preventing string parsing bugs.
    """
    
    def __init__(self, circuit):
        """Initializes the Large Change Engine.

        Args:
            circuit (Circuit): The active circuit orchestrator.
        """
        self.circuit = circuit
        self.param_map = circuit.param_to_component_map

    def compute(self, result, param_names, dp_matrix, variation_axis, method="TR"):
        """Executes simultaneous parameter sweeps using the Woodbury Identity.

        Args:
            result (SimulationResult): The vault containing the baseline simulation's 
                VI state vectors, cached LU factorizations, and sweep axes.
            param_names (list[str]): The k parameter names being varied 
                (e.g., ['R1_value', 'M1_W']).
            dp_matrix (numpy.ndarray): A 2D array of shape (V, k) where V is the 
                number of sweep variations. Each row contains the exact physical 
                shift (Delta P) for each parameter.
            variation_axis (numpy.ndarray): The 1D labels for the V variations 
                (e.g., an alpha array, run indices) for plotting.
            method (str, optional): The numerical integration method ("TR" or "BE"). 
                Defaults to "TR".
                
        Returns:
            LargeChangeData: The calculated circuit voltages for all variations.
        """
        num_variations = dp_matrix.shape[0]
        lc_data = LargeChangeData(variation_axis, result.sweep_axis, result.node_map)
        
        domain = getattr(result, 'domain')
        dt = getattr(result, 'step')

        print(dt)

        # 1. Cache pristine parameters safely
        original_values = {
            p: self.param_map[p].get_nominal_value(p) 
            for p in param_names
        }

        # Initialize the cascading solver with logging turned off for speed
        nl_solver = NonlinearSolver(self.circuit, print_stuff=False)
        
        print(f"\n--- Executing Unified Woodbury-Accelerated Sweeps ({num_variations} variations) ---")
        for v_idx in range(num_variations):
            
            # A. Physically update the component parameters safely
            current_dp = dp_matrix[v_idx, :]
            for i, param in enumerate(param_names):
                comp = self.param_map[param]
                comp.set_nominal_value(param, original_values[param] + current_dp[i])
            
            self.circuit.clear_cache()
            v_prev_variation = np.zeros(self.circuit.total_dim)

            # B. Sweep through the simulation steps (Time or Frequency)
            for step_idx in range(len(result.sweep_axis)):
                V_baseline = result.VI[step_idx]
                lu_base = result.list_of_lus[step_idx]
                
                t = result.sweep_axis[step_idx] if domain == "time" else 0.0
                w = 2 * np.pi * result.sweep_axis[step_idx] if domain == "frequency" else 0.0

                # WOODBURY STRATEGY: Initialize with precomputed PQ topologies
                # We pass the exact dp vector for this variation to the strategy
                woodbury_strategy = ExplicitWoodburyStrategy(
                    self.circuit, param_names, lu_base, V_baseline, dp_array=current_dp
                )

                # Maintain transient history correctly based on the domain
                v_prev = v_prev_variation if domain == "time" else V_baseline

                # THE UNIFIED SOLVE: 
                # Injects the Woodbury Strategy into the core Newton-Raphson loop.
                # Linear circuits converge in exactly 1 iteration.
                _, v_converged = nl_solver.solve(
                    v_ini=V_baseline, domain=domain, t=t, dt=dt, v_prev=v_prev,
                    method=method, strategy=woodbury_strategy 
                )
                
                lc_data.data[v_idx, step_idx, :] = v_converged
                v_prev_variation = v_converged.copy()

        # 2. Restore Circuit to Pristine State to avoid side effects
        for param, val in original_values.items():
            comp = self.param_map[param]
            comp.set_nominal_value(param, val)
            
        self.circuit.clear_cache()
        
        # Real-world time-domain data strips imaginary artifacts
        if domain != "frequency": 
            lc_data.data = np.real(lc_data.data)
            
        return lc_data

