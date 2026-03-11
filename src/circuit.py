from component_factory import create_component

class Circuit:
    def __init__(self, parsed_components):
        # 1. Turn dictionaries into Objects!
        self.components = [
            create_component(name, data) 
            for name, data in parsed_components.items()
        ]
        
        # 2. Build the Global Matrix Index Map
        self.node_map = self._build_node_index()
        self.total_dim = len(self.node_map)
        
        # 3. Tell every component to cache its matrix indices!
        for comp in self.components:
            comp.bind_nodes(self.node_map)

    def get_idx(self, node):
        if node == 0 or node is None:
            return None
        return self.node_map.get(node)

    def _build_node_index(self):
        nodes = set()
        # Look through all components for their node connections
        for comp in self.components:
            for key in ["n1", "n2", "n3", "n4", 'n_d', 'n_g', 'n_s', 'n_b']:
                val = comp.data.get(key, 0)
                if val != 0: nodes.add(val)

        node_list = sorted(nodes)
        node_map = {}
        current_idx = 0

        # Map voltage nodes
        for node in node_list:
            node_map[node] = current_idx
            current_idx += 1

        # Map MNA current branches (Voltage sources and Inductors)
        for comp in self.components:
            if comp.type in ["V", "L"]:
                node_map[comp.name] = current_idx
                current_idx += 1
                
        return node_map
