"""
Simulation Results Data Vault Module.

This module provides a secure, object-oriented container for simulation outputs.
It abstracts away the raw matrix math and nested dictionaries, providing users 
and GUI frontends with a clean API to extract waveforms, operating points, 
and Adjoint sensitivity gradients.

Sensitivity Data Structure:
    The internal sensitivity dictionary has different shapes per analysis type
    to efficiently store the data each engine produces:
    
    - TRAN: {"Integrated_Transient": {node: {param: scalar}}, 
             "Time_Series": {node: {param: 1D_array}}}
    - AC/DC/OP: {node: {param: 1D_array_or_scalar}}
    
    The public API (get_sensitivity, get_sensitivity_parameters) hides this
    complexity so callers never need to know the internal layout.
"""

import numpy as np


class SimulationResult:
    """A secure data vault that holds the results of a simulation.

    Attributes:
        type (str): The analysis type identifier (e.g., '.TRAN', '.AC', '.DC', '.OP').
        sweep_axis (np.ndarray or float): The independent variable array 
            (Time in s, Frequency in Hz, or Voltage in V).
        VI (np.ndarray): The 2D solution matrix of shape [steps, nodes/branches] 
            or 1D array for single-point (.OP) analysis.
        node_map (dict): Dictionary mapping string node/branch names to matrix column indices.
        dt (float, optional): The time step size used in transient analysis.
        sensitivities (dict, optional): The populated adjoint sensitivity data, if calculated.
        list_of_lus (list, optional): Cached LU factorizations from the forward pass.
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
        
        # Populated by the AdjointEngine if sensitivity analysis is run
        self.sensitivities = None
        # Populated by the Simulator for adjoint reuse
        self.list_of_lus = []

    def _resolve_node_key(self, node):
        """Smart lookup that finds the exact node key regardless of int/str type.
        
        Args:
            node (str or int): The requested node identifier.
            
        Returns:
            str or int: The exact key format stored in the node_map.
            
        Raises:
            ValueError: If the node cannot be matched in the map.
        """
        if node in self.node_map:
            return node
            
        try:
            if int(node) in self.node_map:
                return int(node)
        except (ValueError, TypeError):
            pass
            
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
        
        raw_data = self.VI[:, idx] if self.VI.ndim == 2 else self.VI[idx]
        
        return raw_data if is_complex else np.real(raw_data)

    def get_sensitivity_parameters(self, node):
        """Returns a list of parameter names that have sensitivity data for a given node.
        
        Args:
            node (str or int): The name of the objective output node.
            
        Returns:
            list: A list of available sensitivity parameters (e.g., ['R1', 'C1', 'M1_W']).
                  Returns empty list if no sensitivity data exists.
        """
        if not self.sensitivities:
            return []
            
        try:
            exact_key = self._resolve_node_key(node)
        except ValueError:
            return []
            
        if self.type == ".TRAN":
            integ_dict = self.sensitivities.get("Integrated_Transient", {})
            return list(integ_dict.get(exact_key, {}).keys())
            
        elif self.type in [".AC", ".DC", ".OP"]:
            return list(self.sensitivities.get(exact_key, {}).keys())
            
        return []

    def get_sensitivity(self, node, param, output_format="integrated"):
        """Retrieves the parameter sensitivity for a specified output node.

        Args:
            node (str or int): The name of the objective output node.
            param (str): The component parameter name (e.g., 'C1', 'M1_W').
            output_format (str, optional): Only applies to TRAN analysis. 
                - "integrated": Returns the final scalar sensitivity integral (default).
                - "series": Returns the raw backward-traveling time-series array.
                - "accumulator": Returns the cumulative sum (running integral) over time.

        Returns:
            float or np.ndarray: The requested sensitivity scalar or array.

        Raises:
            ValueError: If sensitivity data wasn't calculated or the format is unknown.
            KeyError: If the node or parameter is missing from the sensitivity dictionary.
        """
        if self.sensitivities is None:
            raise ValueError(
                "Sensitivity data is missing. "
                "Did you run the simulation with sensitivity=True?"
            )

        exact_key = self._resolve_node_key(node)

        if self.type == ".TRAN":
            time_series_dict = self.sensitivities.get("Time_Series", {})
            integrated_dict = self.sensitivities.get("Integrated_Transient", {})

            if exact_key not in time_series_dict:
                raise KeyError(
                    f"No sensitivity data calculated for output node '{exact_key}'."
                )

            if param not in time_series_dict[exact_key]:
                available = list(time_series_dict[exact_key].keys())
                raise KeyError(
                    f"Parameter '{param}' not found in sensitivities. "
                    f"Available parameters: {available}"
                )

            if output_format == "series":
                return time_series_dict[exact_key][param]
            elif output_format == "accumulator":
                return np.cumsum(time_series_dict[exact_key][param])
            elif output_format == "integrated":
                return integrated_dict[exact_key].get(param, 0.0)
            else:
                raise ValueError(
                    f"Unknown output_format: '{output_format}'. "
                    "Choose 'integrated', 'series', or 'accumulator'."
                )
                
        # Steady-State Analyses (.AC, .DC, .OP)
        elif self.type in [".AC", ".DC", ".OP"]:
            node_sens = self.sensitivities.get(exact_key)
            if node_sens is None:
                raise KeyError(
                    f"{self.type} Sensitivity data for node '{exact_key}' not found."
                )
            if param not in node_sens:
                available = list(node_sens.keys())
                raise KeyError(
                    f"{self.type} Sensitivity for param '{param}' at node "
                    f"'{exact_key}' not found. Available: {available}"
                )
            return node_sens[param]
        
        else:
            raise ValueError(f"Unknown analysis type: '{self.type}'")
