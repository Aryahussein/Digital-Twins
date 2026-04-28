import numpy as np

class LargeChangeData:
    """A highly sliceable 3D tensor for large-change sensitivity analysis.

    This vault stores the absolute node voltages resulting from simultaneously 
    scaling multiple circuit parameters. It provides a clean, callable interface 
    to extract 1D lines or 2D surfaces for plotting without requiring the user 
    to manage raw numpy indices.

    The underlying tensor maintains the shape:
    (Alpha Multipliers × Sweep Steps × Total Nodes)

    Attributes:
        alpha_axis (np.ndarray): The 1D array of scaling multipliers applied 
            to the sensitivity vectors.
        sweep_axis (np.ndarray): The independent variable array from the base 
            simulation (e.g., Time in s, Frequency in Hz).
        node_map (dict[str or int, int]): Maps string node names to matrix 
            column indices.
        data (np.ndarray): The complex or real 3D tensor holding the calculated 
            absolute voltages.
    """

    def __init__(self, alpha_axis, sweep_axis, node_map):
        """Initializes the LargeChangeData tensor.

        Args:
            alpha_axis (np.ndarray or list): The scaling multipliers.
            sweep_axis (np.ndarray or list): The forward simulation's x-axis.
            node_map (dict): The circuit's node coordinate map.
        """
        self.alpha_axis = np.atleast_1d(alpha_axis)
        self.sweep_axis = np.atleast_1d(sweep_axis)
        self.node_map = node_map
        
        # Shape: (Alphas, Sweep_Steps, Total_Nodes)
        self.data = np.zeros((len(self.alpha_axis), len(self.sweep_axis), len(node_map)), dtype=complex)
        
    def __call__(self, node, alpha_idx=None, step_idx=None):
        """Retrieves specific data slices using a flexible Pythonic call.

        Args:
            node (str or int): The target output node.
            alpha_idx (int, optional): The index of the specific parameter 
                scaling multiplier. Defaults to None.
            step_idx (int, optional): The index of the specific simulation 
                sweep step (time/frequency). Defaults to None.

        Returns:
            np.ndarray or float or complex: 
                - If both indices are provided: Returns a single scalar value.
                - If only alpha_idx is provided: Returns a 1D waveform across the sweep.
                - If only step_idx is provided: Returns a 1D transfer curve across alphas.
                - If no indices are provided: Returns a 2D surface matrix (Alphas x Sweep).
        """
        n_idx = self.node_map[str(node)]
        
        if alpha_idx is not None and step_idx is not None:
            return self.data[alpha_idx, step_idx, n_idx]
        elif alpha_idx is not None:
            return self.data[alpha_idx, :, n_idx] 
        elif step_idx is not None:
            return self.data[:, step_idx, n_idx] 
        else:
            return self.data[:, :, n_idx] 


class LargeChangeEngine:
    """Executes simultaneous large-change sensitivity using Woodbury's Formula.

    This engine identifies the most sensitive components in a circuit and 
    calculates exact voltage variations for large parameter changes. 
    It leverages the Sherman-Morrison-Woodbury matrix identity to mathematically 
    inject parameter changes into the admittance matrix without requiring 
    computationally expensive O(N^1.5) LU refactorizations.

    Attributes:
        circuit (Circuit): The initialized circuit object.
    """
    
    def __init__(self, circuit):
        """Initializes the Large Change Engine.

        Args:
            circuit (Circuit): The main circuit containing the component topologies.
        """
        self.circuit = circuit


    def _build_PQ_topology(self, param_names):
        """Extracts the topological connection matrices P and Q.

        In MNA, a parameter change delta_Y can be factored into P * Delta * Q^T.
        This method maps where each changing component injects current (P) 
        and extracts voltage (Q) within the global matrix.

        Args:
            param_names (list[str]): The names of the k components being varied.

        Returns:
            tuple[np.ndarray, np.ndarray]: Two (N x k) matrices representing 
            the injection (P) and extraction (Q) topologies.
        """
        N = self.circuit.total_dim
        k = len(param_names)
        
        P = np.zeros((N, k))
        Q = np.zeros((N, k))
        
        for col_idx, param in enumerate(param_names):
            comp_name = param.split("_")[0] 
            comp = next(c for c in self.circuit.components if c.name == comp_name)
            
            # Pass the matrices and the target column index by reference
            comp.stamp_PQ(P, Q, col_idx)

        return P, Q

    def _get_delta_y(self, param, dp, domain, w, dt, method):
        """Transforms a physical parameter change into a matrix Admittance change.
        
        Delegates the physics transformation directly to the polymorphic component.
        """
        comp_name = param.split("_")[0]
        comp = next(c for c in self.circuit.components if c.name == comp_name)
        
        # Ask the component to translate its own physical change!
        return comp.get_delta_y(param, dp, domain=domain, w=w, dt=dt, method=method)

    def compute(self, result, param_names, dp_matrix, variation_axis, method="TR"):
        """Executes Woodbury large-change sweeps using an agnostic Delta P matrix.

        Args:
            result (SimulationResult): The vault containing VI and cached LUs.
            param_names (list[str]): The k parameters being simultaneously varied.
            dp_matrix (np.ndarray): A 2D array of shape (V, k) where V is the 
                number of sweep variations and k is the number of parameters.
                Each row contains the exact physical shift for each parameter.
            variation_axis (np.ndarray): The 1D labels for the V variations 
                (e.g., an alpha array, or Monte Carlo run indices) for plotting.
            method (str): Integration scheme.
            
        Returns:
            LargeChangeData: The calculated circuit voltages.
        """
        if not result.list_of_lus:
            raise ValueError("Large-change analysis requires cached LU factorizations.")

        k = len(param_names)
        num_variations = dp_matrix.shape[0]
        
        # 1. Topology Construction
        P, Q = self._build_PQ_topology(param_names)
        
        # 2. Allocate Data Vault (Now using the agnostic variation_axis)
        lc_data = LargeChangeData(variation_axis, result.sweep_axis, result.node_map)
        
        domain = getattr(result, 'domain', 'time') # Fallback if sensitivities aren't present
        dt = getattr(result, 'dt', 0.0) 
        
        print(f"\n--- Executing Agnostic Woodbury Sweeps ({num_variations} variations) ---")
        
        # 3. Time/Freq Sweep Axis Execution
        for step_idx in range(len(result.sweep_axis)):
            V_n = result.VI[step_idx]
            lu = result.list_of_lus[step_idx]
            w = 2 * np.pi * result.sweep_axis[step_idx] if domain == "frequency" else 0.0
            
            # Precompute heavy matrix logic
            X = lu.solve(P)  
            QT_V = Q.T @ V_n 
            QT_X = Q.T @ X   
            
            # 4. Variations Loop (Iterating over the rows of dp_matrix)
            for v_idx in range(num_variations):
                
                # Build the k x k Diagonal Matrix strictly from the input matrix
                delta_matrix = np.zeros((k, k), dtype=complex)
                for i, param in enumerate(param_names):
                    
                    # Extract the exact physical Delta P for this specific variation step
                    dp = dp_matrix[v_idx, i] 
                    
                    delta_matrix[i, i] = self._get_delta_y(param, dp, domain, w, dt, method)
                    
                # Woodbury Core: (I + Delta * Q^T * X)
                I_k = np.eye(k)
                M = I_k + delta_matrix @ QT_X
                
                # Inversion and Update
                inv_M = np.linalg.inv(M)
                dV = -X @ (inv_M @ (delta_matrix @ QT_V))
                
                lc_data.data[v_idx, step_idx, :] = V_n + dV
                
        if domain != "frequency":
            lc_data.data = np.real(lc_data.data)
            
        return lc_data

