"""
Simulation Results Data Vault Module.

This module provides a secure, object-oriented container for simulation outputs.
It abstracts away the raw matrix math and nested dictionaries, providing users 
and GUI frontends with a clean API to extract waveforms, operating points, 
and Adjoint sensitivity gradients.
"""

import numpy as np

class SimulationResult:
    """A secure data vault that holds the results of a simulation.

    This class provides clean, object-oriented methods to extract waveforms, 
    operating points, and sensitivities without requiring the user to interact 
    with raw matrices or nested dictionary structures.

    Attributes:
        type (str): The analysis type identifier (e.g., '.TRAN', '.AC', '.DC', '.OP').
        sweep_axis (np.ndarray or float): The independent variable array 
            (Time in s, Frequency in Hz, or Voltage in V).
        VI (np.ndarray): The 2D solution matrix of shape [steps, nodes/branches] 
            or 1D array for single-point (.OP) analysis.
        node_map (dict): Dictionary mapping string node/branch names to matrix column indices.
        dt (float, optional): The time step size used in transient analysis.
        sensitivities (dict, optional): The populated adjoint sensitivity data, if calculated.
    """

    def __init__(self, analysis_type, sweep_axis, VI_matrix, node_map, dt=None):
        """Initializes the SimulationResult vault.

        Args:
            analysis_type (str): Type of simulation run.
            sweep_axis (np.ndarray): The x-axis array for sweeps.
            VI_matrix (np.ndarray): The raw matrix output from the math solver.
            node_map (dict): The circuit's node coordinate map.
            dt (float, optional): The transient time step. Defaults to None.
        """
        self.type = analysis_type      
        self.sweep_axis = sweep_axis   
        self.VI = VI_matrix            
        self.node_map = node_map       
        self.dt = dt                   
        
        # This will be populated by the AdjointEngine if sensitivity analysis is run
        self.sensitivities = None

    def _resolve_node_key(self, node):
        """Smart lookup that finds the exact node key regardless of int/str type.
        
        Args:
            node (str or int): The requested node identifier.
            
        Returns:
            str or int: The exact key format stored in the node_map.
            
        Raises:
            ValueError: If the node cannot be matched in the map.
        """
        # 1. Try exact match
        if node in self.node_map:
            return node
            
        # 2. Try converting string to int (User passed "2", map has 2)
        try:
            if int(node) in self.node_map:
                return int(node)
        except ValueError:
            pass
            
        # 3. Try converting int to string (User passed 2, map has "2")
        if str(node) in self.node_map:
            return str(node)
            
        raise ValueError(f"Node '{node}' not found in the circuit's node_map.")

    def get_voltage(self, node):
        """Retrieves the voltage (or branch current) array for a given node.

        Args:
            node (str or int): The name of the node or MNA branch.

        Returns:
            np.ndarray or float or complex: A 1D array for sweeps, or a scalar for OP.
                Returns complex types for AC analysis, and real types for TRAN/DC.

        Raises:
            ValueError: If the requested node does not exist in the node_map.
        """
        exact_key = self._resolve_node_key(node)
        idx = self.node_map[exact_key]
        
        is_complex = (self.type == ".AC")
        
        # Handle both 2D (sweeps) and 1D (.OP) matrix shapes
        raw_data = self.VI[:, idx] if self.VI.ndim == 2 else self.VI[idx]
        
        return raw_data if is_complex else np.real(raw_data)

    def get_sensitivity_parameters(self, node):
        """Returns a list of parameter names that have sensitivity data for a given node.
        
        Args:
            node (str or int): The name of the objective output node.
            
        Returns:
            list: A list of available sensitivity parameters (e.g., ['R1', 'C1', 'M1_W']).
        """
        if not self.sensitivities:
            return []
            
        try:
            exact_key = self._resolve_node_key(node)
        except ValueError:
            return []
        
        if self.type == ".TRAN":
            # Check the new default Continuous Local DC first, then fallback to Integrated
            if "Continuous_Local_DC" in self.sensitivities:
                return list(self.sensitivities["Continuous_Local_DC"].get(exact_key, {}).keys())
            elif "Integrated_Transient" in self.sensitivities:
                return list(self.sensitivities["Integrated_Transient"].get(exact_key, {}).keys())
            return []
            
        elif self.type in [".AC", ".DC", ".OP"]:
            return list(self.sensitivities.get(exact_key, {}).keys())
            
        return []

    def get_sensitivity(self, node, param, output_format="local_dc"):
        """Retrieves the parameter sensitivity for a specified output node.

        Args:
            node (str or int): The name of the objective output node.
            param (str): The component parameter name (e.g., 'C1', 'M1_W').
            output_format (str, optional): Only applies to TRAN analysis. 
                - "local_dc": (DEFAULT) Continuous waveform (delta-v(t)) from Local DC Adjoint.
                - "integrated": The final scalar sensitivity integral (Global Adjoint).
                - "series": The raw backward-traveling time-series array (Global Adjoint).
                - "accumulator": The cumulative sum (running integral) over time (Global Adjoint).

        Returns:
            float or np.ndarray: The requested sensitivity scalar or array.

        Raises:
            ValueError: If sensitivity data wasn't calculated or the format is unknown.
            KeyError: If the node or parameter is missing from the sensitivity dictionary.
        """
        if self.sensitivities is None:
            raise ValueError("Sensitivity data is missing. Did you run the simulation with sensitivity=True?")

        exact_key = self._resolve_node_key(node)

        if self.type == ".TRAN":
            local_dc_dict = self.sensitivities.get("Continuous_Local_DC", {})
            time_series_dict = self.sensitivities.get("Time_Series", {})
            integrated_dict = self.sensitivities.get("Integrated_Transient", {})

            # 1. New Default: Continuous Local DC Adjoint Waveform
            if output_format == "local_dc":
                if exact_key not in local_dc_dict or param not in local_dc_dict[exact_key]:
                    raise KeyError(f"Local DC sensitivity for node '{exact_key}' and param '{param}' not found.")
                return local_dc_dict[exact_key][param]

            # 2. Global Adjoint Formats (Integrated, Series, Accumulator)
            if exact_key not in time_series_dict:
                raise KeyError(f"No global adjoint data calculated for output node '{exact_key}'.")
            if param not in time_series_dict[exact_key]:
                raise KeyError(f"Parameter '{param}' not found. Known parameters: {list(time_series_dict[exact_key].keys())}")

            if output_format == "series":
                return time_series_dict[exact_key][param]
            elif output_format == "accumulator":
                return np.cumsum(time_series_dict[exact_key][param]) * self.dt
            elif output_format == "integrated":
                return integrated_dict[exact_key].get(param, 0.0)
            else:
                raise ValueError(
                    f"Unknown output_format: '{output_format}'. "
                    "Choose 'local_dc', 'integrated', 'series', or 'accumulator'."
                )
                
        # Steady-State Analyses (.AC, .DC, .OP)
        elif self.type in [".AC", ".DC", ".OP"]:
            try:
                return self.sensitivities[exact_key][param]
            except KeyError:
                raise KeyError(f"{self.type} Sensitivity data for node '{exact_key}' and param '{param}' not found.")
