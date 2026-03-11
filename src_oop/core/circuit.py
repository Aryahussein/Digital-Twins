from core.component_factory import create_component

class Circuit:
    """Represents the physical circuit, containing components and matrix topology.

    This class acts as the central data structure for the simulator. It converts
    raw parsed netlist dictionaries into polymorphic component objects, identifies
    all unique electrical nodes, and generates the Modified Nodal Analysis (MNA)
    matrix indexing scheme.

    Attributes:
        components (list): A list of instantiated polymorphic component objects.
        components_dict (dict): A name-to-object lookup dictionary for fast access.
        node_map (dict): Maps node names (and MNA branch names) to integer matrix indices.
        total_dim (int): The total dimension of the MNA matrix (N x N).
    """

    def __init__(self, parsed_components):
        """Initializes the Circuit by constructing objects and the matrix topology.

        Args:
            parsed_components (dict): Raw dictionary from the NetlistParser where
                keys are component names and values are parameter dictionaries.
        """
        # 1. Turn dictionaries into Objects!
        self.components = [
            create_component(name, data) 
            for name, data in parsed_components.items()
        ]
        
        # Extension added: Keep a dictionary for fast O(1) lookups by name
        self.components_dict = {comp.name: comp for comp in self.components}
        
        # 2. Build the Global Matrix Index Map
        self.node_map = self._build_node_index()
        self.total_dim = len(self.node_map)
        
        # 3. Tell every component to cache its matrix indices!
        for comp in self.components:
            comp.bind_nodes(self.node_map)

    def get_idx(self, node):
        """Retrieves the matrix row/column index for a given node.

        Ground nodes (0, "0", or "GND") are not included in the MNA matrix and 
        return None.

        Args:
            node (int or str): The name or integer ID of the circuit node.

        Returns:
            int or None: The integer matrix index, or None if the node is ground.
        """
        if node == 0 or node == "0" or str(node).upper() == "GND" or node is None:
            return None
        return self.node_map.get(node)

    def _build_node_index(self):
        """Scans components to build the MNA matrix coordinate map.

        Assigns sequential integer indices to all unique voltage nodes first, 
        followed by additional indices for branch currents required by MNA 
        components (like Voltage Sources and Inductors).

        Returns:
            dict: A mapping of node/branch names to integer matrix indices.
        """
        nodes = set()
        
        # Look through all components for their node connections
        for comp in self.components:
            for key in ["n1", "n2", "n3", "n4", 'n_d', 'n_g', 'n_s', 'n_b']:
                val = comp.data.get(key, 0)
                # Extension added: ensure we aren't adding string grounds
                if val != 0 and val != "0" and str(val).upper() != "GND": 
                    nodes.add(val)

        node_list = sorted(nodes, key=str)
        
        node_map = {}
        current_idx = 0

        # Map voltage nodes
        for node in node_list:
            node_map[node] = current_idx
            current_idx += 1

        # Map MNA current branches (Voltage sources and Inductors)
        for comp in self.components:
            # Future-proofing: Rely on component properties rather than hardcoded letters
            if comp.type in ["V", "L", "H", "F", "E"]: 
                node_map[comp.name] = current_idx
                current_idx += 1
                
        return node_map
    
    def get_component(self, name):
        """Retrieves a component object by its netlist name.
        
        Args:
            name (str): The name of the component (e.g., 'R1', 'M1').
            
        Returns:
            Component: The instantiated component object.
            
        Raises:
            KeyError: If the component does not exist in the circuit.
        """
        if name not in self.components_dict:
            raise KeyError(f"Component '{name}' not found in circuit.")
        return self.components_dict[name]
