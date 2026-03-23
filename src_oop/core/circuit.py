"""
Circuit Representation Module.

This module provides the central `Circuit` class, which acts as the primary data 
structure for the simulator. It includes the factory logic to instantiate 
polymorphic components from parsed netlist dictionaries and generates the 
Modified Nodal Analysis (MNA) matrix indexing scheme.
"""

# Import all component subclasses from the new modular package
from components import (
    Resistor, 
    Capacitor, 
    VoltageSource, 
    NMOS, PMOS,
    Inductor,
    CurrentSource,
    Diode,
    VCCS,
    OpAmp
)

# =====================================================================
# COMPONENT FACTORY REGISTRY
# =====================================================================
_COMPONENT_REGISTRY = {
    'R': Resistor,
    'C': Capacitor,
    'V': VoltageSource,
    'NMOS': NMOS,
    'PMOS': PMOS,
    'L': Inductor,
    'I': CurrentSource,
    'D': Diode,
    'G': VCCS,
    'E': OpAmp
}

def create_component(name, data_dict):
    """Instantiates the correct Component subclass based on the netlist type.

    Args:
        name (str): The unique netlist name of the component (e.g., 'R1', 'M_MAIN').
        data_dict (dict): The parsed parameter dictionary containing at minimum
            a 'type' key (e.g., {'type': 'R', 'n1': '1', 'n2': '0', 'value': 1000}).

    Returns:
        Component: An instantiated component object.

    Raises:
        KeyError: If the 'type' key is missing.
        ValueError: If the component type is not recognized.
    """
    type_char = data_dict.get("type")
    
    if not type_char:
        raise KeyError(f"Component '{name}' is missing a 'type' attribute in the parsed netlist.")

    type_char = str(type_char).upper()
    component_class = _COMPONENT_REGISTRY.get(type_char)
    
    if component_class is None:
        raise ValueError(
            f"Unknown component type '{type_char}' for '{name}'. "
            f"Supported types are: {list(_COMPONENT_REGISTRY.keys())}"
        )
        
    return component_class(name, data_dict)


# =====================================================================
# MAIN CIRCUIT CLASS
# =====================================================================
class Circuit:
    """Represents the physical circuit, containing components and matrix topology.

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
        # 1. Turn dictionaries into Objects using the factory!
        self.components = [
            create_component(name, data) 
            for name, data in parsed_components.items()
        ]
        
        # 2. Keep a dictionary for fast O(1) lookups by name
        self.components_dict = {comp.name: comp for comp in self.components}
        
        # 3. Build the Global Matrix Index Map
        self.node_map = self._build_node_index()
        self.total_dim = len(self.node_map)
        
        # 4. Tell every component to cache its matrix indices!
        for comp in self.components:
            comp.bind_nodes(self.node_map)

    def get_idx(self, node):
        """Retrieves the matrix row/column index for a given node.

        Ground nodes (0, "0", or "GND") return None.
        """
        if node == 0 or node == "0" or str(node).upper() == "GND" or node is None:
            return None
        return self.node_map.get(node)

    def _build_node_index(self):
        """Scans components to build the MNA matrix coordinate map."""
        nodes = set()
        
        # Look through all components for their node connections
        for comp in self.components:
            for key in ["n1", "n2", "n3", "n4", 'n_d', 'n_g', 'n_s', 'n_b']:
                val = comp.data.get(key, 0)
                # Ensure we aren't adding string grounds
                if val != 0 and val != "0" and str(val).upper() != "GND": 
                    nodes.add(val)

        node_list = sorted(nodes, key=str)
        node_map = {}
        current_idx = 0

        # Map voltage nodes
        for node in node_list:
            node_map[node] = current_idx
            current_idx += 1

        # Map MNA current branches (Voltage sources, Inductors, OpAmps, etc.)
        for comp in self.components:
            if comp.type in ["V", "L", "H", "F", "E"]: 
                node_map[comp.name] = current_idx
                current_idx += 1
                
        return node_map
    
    def get_component(self, name):
        """Retrieves a component object by its netlist name."""
        if name not in self.components_dict:
            raise KeyError(f"Component '{name}' not found in circuit.")
        return self.components_dict[name]
