import numpy as np

class SensitivityData:
    """A generalized 3D tensor for all simulation sensitivity results.

    This structure handles single-point (.OP), 1D sweeps (.DC, .AC), 
    and time-domain (.TRAN) sensitivities using a single, unified 3D matrix. 
    It hides the raw matrix math from the user, allowing them to query 
    gradients using human-readable component and node names.

    The underlying tensor always maintains the shape: 
    (Parameters × Output Nodes × Sweep Steps)

    Attributes:
        sweep_axis (np.ndarray): The independent variable array 
            (e.g., Time in s, Frequency in Hz, Voltage in V). 
            For .OP analysis, this is a single-element array.
        domain (str): The physical domain of the sweep axis 
            (e.g., "time", "frequency", "voltage", "static").
        param_names (list[str]): Ordered parameter names (Axis 0 labels).
        output_nodes (list[str]): Ordered output node names (Axis 1 labels).
        data (np.ndarray): The 3D sensitivity tensor of shape 
            (n_params, n_outputs, n_sweep_steps).
        param_index (dict[str, int]): Fast-lookup dictionary mapping parameter 
            names to their row index in the tensor.
        output_index (dict[str, int]): Fast-lookup dictionary mapping output 
            node names to their column index in the tensor.
        adjoint_vectors (np.ndarray or None): Optional raw adjoint solution vectors 
            of shape (n_outputs, n_sweep_steps, total_dim). Stored when needed 
            for fault analysis (shorts require raw ψ at every node, not just 
            the sensitivity products).
    """

    def __init__(self, sweep_axis, param_names, output_nodes, domain="static"):
        """Initializes the SensitivityData tensor.

        Args:
            sweep_axis (np.ndarray or list or float): The independent variable 
                array from the solver.
            param_names (list[str]): List of circuit parameter names (e.g., ['R1', 'C1']).
            output_nodes (list[str]): List of objective output nodes requested by the user.
            domain (str, optional): The physical domain of the sweep. Defaults to "static".
        """
        self.sweep_axis = np.atleast_1d(sweep_axis)
        self.domain = domain
        self.param_names = list(param_names)
        self.output_nodes = list(output_nodes)

        self.data = np.zeros((len(param_names), len(output_nodes), len(self.sweep_axis)))

        self.param_index = {name: idx for idx, name in enumerate(param_names)}
        self.output_index = {name: idx for idx, name in enumerate(output_nodes)}
        # Raw adjoint vectors (populated optionally for fault analysis)
        self.adjoint_vectors = None

    def get_sweep_series(self, param, output_node):
        """Retrieves the sensitivity array across the entire sweep axis.

        This grabs a 1D 'core sample' through the depth of the 3D tensor, 
        representing how the sensitivity changes across the sweep variable.

        Args:
            param (str): The component parameter name (e.g., 'R1').
            output_node (str or int): The target output node name.

        Returns:
            np.ndarray: A 1D array representing the partial derivative of the 
            output node with respect to the parameter across the sweep.
        """
        p = self.param_index[param]
        o = self.output_index[output_node]
        return self.data[p, o, :]

    def get_matrix_at_step(self, step_idx):
        """Returns the 2D sensitivity matrix at a specific sweep step.

        Args:
            step_idx (int): The integer index of the sweep step (e.g., a 
                specific time or frequency step).

        Returns:
            np.ndarray: A 2D array of shape (n_params, n_outputs) representing 
            the sensitivities of all nodes to all parameters at the given step.
        """
        return self.data[:, :, step_idx]

    def print_matrix_at_step(self, step_idx):
        """Prints a neatly formatted, labeled 2D sensitivity matrix to the console.

        Args:
            step_idx (int): The integer index of the sweep step to display.
        """
        val = self.sweep_axis[step_idx]
        units = {"time": "s", "frequency": "Hz", "voltage": "V"}.get(self.domain, "")
        
        print(f"\n=== Sensitivity Matrix S({self.domain} = {val:.4e} {units}) ===")
        
        header = f"{'Parameter':<15}"
        for node in self.output_nodes:
            header += f"{'dV('+str(node)+')':<15}"
        print(header)
        print("-" * len(header))
        
        for p_idx, param in enumerate(self.param_names):
            row = f"{param:<15}"
            for o_idx in range(len(self.output_nodes)):
                matrix_val = self.data[p_idx, o_idx, step_idx]
                row += f"{matrix_val:<+15.6e}"
            print(row)


class SimulationResult:
    """A unified data vault holding the results of a circuit simulation.

    This class provides a clean API to extract state variables (voltages/currents) 
    and sensitivity gradients without requiring the user to manage matrix 
    dimensions or analysis-specific logic (.AC vs .TRAN).

    Attributes:
        type (str): The analysis type identifier (e.g., '.TRAN', '.AC', '.DC', '.OP').
        sweep_axis (np.ndarray): The independent variable array.
        VI (np.ndarray): The primary solution matrix generated by the solver.
        node_map (dict[str or int, int]): Maps string node/branch names to 
            matrix column indices.
        sensitivities (SensitivityData or None): The generalized tensor holding 
            all adjoint sensitivity gradients, if the adjoint engine was enabled.
    """

    def __init__(self, analysis_type, sweep_axis, VI_matrix, node_map):
        """Initializes the SimulationResult vault.

        Args:
            analysis_type (str): Type of simulation run (e.g., '.TRAN').
            sweep_axis (np.ndarray): The x-axis array for sweeps.
            VI_matrix (np.ndarray): The raw matrix output from the math solver.
            node_map (dict): The circuit's node coordinate map.
        """
        self.type = analysis_type      
        self.sweep_axis = np.atleast_1d(sweep_axis)   
        self.VI = VI_matrix            
        self.node_map = node_map       
        self.sensitivities = None 
        self.global_sensitivities = None

    def _resolve_node_key(self, node):
        """Resolves the exact node key from the user input.

        Handles casting discrepancies (e.g., user requesting string "2" when 
        the map stores integer 2).

        Args:
            node (str or int): The requested node identifier.

        Returns:
            str or int: The exact matched key from the node_map.

        Raises:
            ValueError: If the requested node does not exist in the node_map.
        """
        if node in self.node_map: 
            return node
            
        try:
            if int(node) in self.node_map: 
                return int(node)
        except ValueError: 
            pass
            
        if str(node) in self.node_map: 
            return str(node)
        
        raise ValueError(f"Node '{node}' not found in the circuit's node_map.")

    def get_voltage(self, node):
        """Retrieves the voltage or branch current array for a given node.

        Args:
            node (str or int): The name of the node or MNA branch.

        Returns:
            np.ndarray or float or complex: A 1D array for sweeps, or a scalar 
            for .OP analyses. Returns complex types for .AC analysis, and real 
            types for .TRAN and .DC analyses.
        """
        exact_key = self._resolve_node_key(node)
        idx = self.node_map[exact_key]
        
        is_complex = (self.type == ".AC")
        raw_data = self.VI[:, idx] if self.VI.ndim == 2 else self.VI[idx]
        
        return raw_data if is_complex else np.real(raw_data)

    def get_sensitivity_parameters(self, node):
        """Retrieves a list of parameters that have sensitivity data for a node.
        
        This is primarily used by plotting tools and GUIs to populate dropdown 
        menus and prevent KeyErrors when requesting data.

        Args:
            node (str or int): The name of the objective output node.
            
        Returns:
            list[str]: A list of available sensitivity parameters (e.g., ['R1', 'C1']).
                Returns an empty list if no sensitivity data was calculated or 
                if the node is invalid.
        """
        if self.sensitivities is None:
            return []
            
        try:
            exact_key = self._resolve_node_key(node)
        except ValueError:
            return []
        
        if exact_key not in self.sensitivities.output_nodes:
            return []

        return self.sensitivities.param_names

    def get_sensitivity(self, node, param):
        """Retrieves the sensitivity gradient for a specified output node.

        Automatically handles the dimensionality of the underlying simulation 
        and routes the request to the underlying 3D tensor.

        Args:
            node (str or int): The name of the objective output node.
            param (str): The component parameter name (e.g., 'C1', 'M1_W').

        Returns:
            float or np.ndarray: A scalar float for single-point (.OP) analyses, 
            otherwise a 1D numpy array representing the gradient.

        Raises:
            ValueError: If the simulation was run without the Adjoint Engine enabled.
            KeyError: If the requested node or parameter is missing from the dataset.
        """
        if self.sensitivities is None:
            raise ValueError(
                "Sensitivity data is missing. Ensure the simulation "
                "was executed with the Adjoint Engine enabled."
            )

        exact_key = self._resolve_node_key(node)
        
        gradient_array = self.sensitivities.get_sweep_series(param, exact_key)

        if len(gradient_array) == 1:
            return gradient_array[0]
            
        return gradient_array
