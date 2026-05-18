"""
Circuit Representation Module.

This module provides the central `Circuit` class, which acts as the primary data 
structure for the simulator. It includes the factory logic to instantiate 
polymorphic components from parsed netlist dictionaries and generates the 
Modified Nodal Analysis (MNA) matrix indexing scheme.
"""
import scipy.sparse as sp
import numpy as np

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
    VCVS,
    CCCS,
    CCVS,
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
    'E': VCVS,
    'F': CCCS,
    'H': CCVS
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
        # print(parsed_components)
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

        # 5. Pre-sort components using Capability Flags (Open-Closed Principle!)
        # The Circuit no longer cares what the component *is*, only what it *does*.
        self._src_comps = [c for c in self.components if getattr(c, 'IS_INDEPENDENT_SOURCE', False)]
        self._tran_comps = [c for c in self.components if getattr(c, 'IS_DYNAMIC', False)]
        self._ac_comps = [c for c in self.components if getattr(c, 'IS_AC_REACTIVE', False)]
        self._nl_comps = [c for c in self.components if getattr(c, 'IS_NONLINEAR', False)]
        
        self.is_nonlinear = len(self._nl_comps) > 0

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
                if val != 0 and val != "0" and str(val).upper() != "GND": 
                    nodes.add(val)

        node_list = sorted(nodes, key=str)
        node_map = {}
        current_idx = 0

        # Map voltage nodes
        for node in node_list:
            node_map[node] = current_idx
            current_idx += 1

        # Map MNA current branches using the Open-Closed Principle!
        for comp in self.components:
            if getattr(comp, 'REQUIRES_BRANCH_EQ', False): 
                node_map[comp.name] = current_idx
                current_idx += 1
                
        return node_map
    
    def get_component(self, name):
        """Retrieves a component object by its netlist name."""
        if name not in self.components_dict:
            raise KeyError(f"Component '{name}' not found in circuit.")
        return self.components_dict[name]

    @property
    def differentiable_params(self):
        """A master list of all tunable parameters in the circuit."""
        params = []
        for comp in self.components:
            for p in comp.differentiable_params:
                if p not in params:
                    params.append(p)
        return params

    @property
    def param_to_component_map(self):
        """Returns an O(1) lookup dictionary mapping parameter strings to their Component objects."""
        mapping = {}
        for comp in self.components:
            for p in comp.differentiable_params:
                mapping[p] = comp
        return mapping

    def precompute_base(self):
        """Compiles the static skeleton matrix exactly ONCE.
        
        Always builds as 'float' to maximize Transient/DC performance.
        MNA topology (+1/-1) and Resistors are purely real.
        """
        # No more domain checks! Always blazing fast floats.
        self._Y_base = sp.lil_matrix((self.total_dim, self.total_dim), dtype=float)
        
        for comp in self.components:
            if hasattr(comp, 'stamp_base_matrix'):
                comp.stamp_base_matrix(self._Y_base)

    def clear_cache(self):
        """Frees the base matrix memory cache."""
        if hasattr(self, '_Y_base'):
            del self._Y_base

    def build_system(self, domain="static", t=0.0, dt=0.0, w=0.0, v_prev=None, v_k=None, method="TR", base_matrix=None):
        """The Master Composer. Assembles the global matrix and RHS vector dynamically."""

        if not hasattr(self, '_Y_base'):
            self.precompute_base()

        Y = base_matrix.copy() if base_matrix is not None else self._Y_base.copy()
        if domain == "frequency" and Y.dtype != complex: Y = Y.astype(complex)
        J = np.zeros(self.total_dim, dtype=complex if domain=="frequency" else float)
        
        # 1. Independent Sources (Always stamp RHS)
        for comp in self._src_comps:
            comp.stamp_sources(J, domain, t)
                
        # 2. Dynamic/Non-Linear Components (The Unified Pipeline)
        # We group all components that need dynamic evaluation into one list
        dynamic_comps = []
        if domain == "time": dynamic_comps.extend(self._tran_comps)
        if domain == "frequency": dynamic_comps.extend(self._ac_comps)
        if self.is_nonlinear and v_k is not None: dynamic_comps.extend(self._nl_comps)
        
        # Deduplicate in case a component is both transient AND nonlinear (like a dynamic MOSFET)
        dynamic_comps = list(set(dynamic_comps))

        for comp in dynamic_comps:
            # 1. Evaluate Physics ONCE
            res = comp.evaluate_physics(domain=domain, t=t, dt=dt, w=w, v_prev=v_prev, v_k=v_k, method=method)
            
            # 2. Generic Stamping
            comp.stamp_matrix(Y, res)
            comp.stamp_rhs(J, res)
                
        return Y, J

    def build_rhs_only(self, domain="static", t=0.0, dt=0.0, w=0.0, v_prev=None, v_k=None, method="TR"):
        """
        Blazing-fast extraction of ONLY the Right-Hand Side (J/Sources) vector.
        Completely bypasses matrix allocation (Used by Woodbury Strategy).
        """
        J = np.zeros(self.total_dim, dtype=complex if domain=="frequency" else float)
            
        # 1. Independent Sources
        for comp in self._src_comps:
            comp.stamp_sources(J, domain, t)
                
        # 2. Dynamic/Non-Linear Components (The Unified Pipeline)
        dynamic_comps = []
        if domain == "time": dynamic_comps.extend(self._tran_comps)
        if domain == "frequency": dynamic_comps.extend(self._ac_comps)
        if self.is_nonlinear and v_k is not None: dynamic_comps.extend(self._nl_comps)
        
        # Deduplicate
        dynamic_comps = list(set(dynamic_comps))

        for comp in dynamic_comps:
            # 1. Evaluate Physics ONCE
            res = comp.evaluate_physics(domain=domain, t=t, dt=dt, w=w, v_prev=v_prev, v_k=v_k, method=method)
            
            # 2. Generic Stamping (RHS ONLY)
            comp.stamp_rhs(J, res)
                
        return J
