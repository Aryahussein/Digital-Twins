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
        idx = self.node_map.get(node)
        if idx is None:
            raise ValueError(f"Node '{node}' not found in the circuit's node_map.")
        
        # Bug Fix: Do not strip the imaginary part if this is an AC analysis!
        is_complex = (self.type == ".AC")
        
        if self.VI.ndim == 2:
            raw_data = self.VI[:, idx]
            return raw_data if is_complex else np.real(raw_data)
        else:
            raw_data = self.VI[idx]
            return raw_data if is_complex else np.real(raw_data)

    def get_sensitivity_parameters(self, node):
        """Returns a list of parameter names that have sensitivity data for a given node."""
        if not self.sensitivities:
            return []
            
        if self.type == ".TRAN":
            integ_dict = self.sensitivities.get("Integrated_Transient", {})
            return list(integ_dict.get(node, {}).keys())
            
        elif self.type in [".AC", ".DC", ".OP"]:
            return list(self.sensitivities.get(node, {}).keys())
            
        return []

    def get_sensitivity(self, node, param, output_format="integrated"):
        """Retrieves the parameter sensitivity for a specified output node.

        

        Args:
            node (str): The name of the objective output node.
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
            raise ValueError("Sensitivity data is missing. Did you run the simulation with sensitivity=True?")

        if self.type == ".TRAN":
            time_series_dict = self.sensitivities.get("Time_Series", {})
            integrated_dict = self.sensitivities.get("Integrated_Transient", {})

            # Graceful fallback if only the primary output node was tracked
            if node not in time_series_dict:
                available_nodes = list(time_series_dict.keys())
                if available_nodes:
                    node = available_nodes[0]
                else:
                    raise KeyError(f"No sensitivity data calculated for output node '{node}'.")

            if param not in time_series_dict[node]:
                raise KeyError(f"Parameter '{param}' not found in sensitivities. Known parameters: {list(time_series_dict[node].keys())}")

            if output_format == "series":
                return time_series_dict[node][param]
            elif output_format == "accumulator":
                return np.cumsum(time_series_dict[node][param]) * self.dt
            elif output_format == "integrated":
                return integrated_dict.get(param, 0.0)
            else:
                raise ValueError(f"Unknown output_format: '{output_format}'. Choose 'integrated', 'series', or 'accumulator'.")
                
        elif self.type == ".AC":
            try:
                return self.sensitivities[node][param]
            except KeyError:
                raise KeyError(f"AC Sensitivity data for node '{node}' and param '{param}' not found.")
                
        # Extension Added: Safely handle .DC sweeps separately from .OP
        elif self.type == ".DC":
            try:
                return self.sensitivities[node][param]
            except KeyError:
                raise KeyError(f"DC Sweep Sensitivity data for node '{node}' and param '{param}' not found.")
                
        else: # For .OP
            try:
                return self.sensitivities[node][param]
            except KeyError:
                raise KeyError(f"OP Sensitivity data for node '{node}' and param '{param}' not found.")
