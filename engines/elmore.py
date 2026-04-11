"""
Elmore Delay Engine
===================
Computes the Elmore delay from a voltage source to every node in an RC tree.

The Elmore delay to node n is:

    T_D(n) = sum_k [ R_k * C_downstream(k, n) ]

where the sum is over all resistors k in the circuit, and
C_downstream(k, n) is the total capacitance in the subtree that
contains node n when resistor k is removed from the tree.

Algorithm
---------
1. Build adjacency graph from resistors (source node = 0/GND is root).
2. BFS/DFS from the voltage source node to find the tree path to each node.
3. For each target node, walk the path back to the source, accumulating
   R * C_total_subtree at each resistor along the way.

Limitations
-----------
- Requires a tree topology (no loops of resistors). Loops make Elmore
  delay ill-defined.
- Capacitors must be connected to ground (standard RC tree assumption).
- Works only for linear RC circuits with a single input source.
"""

import numpy as np
from collections import defaultdict, deque


class ElmoreEngine:
    """Computes Elmore delays for RC tree circuits."""

    def __init__(self, circuit):
        self.circuit = circuit

    def _build_rc_graph(self):
        """
        Extract resistors and grounded capacitors from the circuit.

        Returns
        -------
        resistors : list of (node_a, node_b, R_value, comp_name)
        cap_at_node : dict {node -> total_capacitance}
        source_node : the node driven by the voltage source (non-ground terminal)
        """
        resistors  = []
        cap_at_node = defaultdict(float)
        source_node = None

        ground = {0, '0', 'GND', 'gnd'}

        for comp in self.circuit.components:
            t = comp.type

            if t == 'R':
                n1 = comp.data.get('n1', 0)
                n2 = comp.data.get('n2', 0)
                # Skip if either end is ground (shunt resistor — rare in RC trees)
                n1_gnd = (n1 in ground or str(n1).upper() == 'GND')
                n2_gnd = (n2 in ground or str(n2).upper() == 'GND')
                if not n1_gnd and not n2_gnd:
                    resistors.append((n1, n2, comp.value, comp.name))
                elif n1_gnd and not n2_gnd:
                    # shunt R — not part of an RC tree, skip
                    pass
                elif n2_gnd and not n1_gnd:
                    pass

            elif t == 'C':
                n1 = comp.data.get('n1', 0)
                n2 = comp.data.get('n2', 0)
                n1_gnd = (n1 in ground or str(n1).upper() == 'GND')
                n2_gnd = (n2 in ground or str(n2).upper() == 'GND')
                if n1_gnd and not n2_gnd:
                    cap_at_node[n2] += comp.value
                elif n2_gnd and not n1_gnd:
                    cap_at_node[n1] += comp.value
                # Floating capacitors ignored for Elmore

            elif t == 'V':
                n1 = comp.data.get('n1', 0)
                n2 = comp.data.get('n2', 0)
                n1_gnd = (n1 in ground or str(n1).upper() == 'GND')
                n2_gnd = (n2 in ground or str(n2).upper() == 'GND')
                if n2_gnd:
                    source_node = n1
                elif n1_gnd:
                    source_node = n2

        return resistors, cap_at_node, source_node

    def _build_adjacency(self, resistors):
        """Build undirected adjacency list: node -> [(neighbour, R, name)]."""
        adj = defaultdict(list)
        for n1, n2, R, name in resistors:
            adj[n1].append((n2, R, name))
            adj[n2].append((n1, R, name))
        return adj

    def _bfs_tree(self, adj, source):
        """
        BFS from source to discover parent relationships in the RC tree.

        Returns
        -------
        parent      : dict {node -> (parent_node, R_to_parent, comp_name)}
        bfs_order   : list of nodes in BFS order (source first)
        """
        parent    = {source: None}
        bfs_order = [source]
        queue     = deque([source])

        while queue:
            node = queue.popleft()
            for neighbour, R, name in adj[node]:
                if neighbour not in parent:
                    parent[neighbour] = (node, R, name)
                    bfs_order.append(neighbour)
                    queue.append(neighbour)

        return parent, bfs_order

    def _subtree_capacitance(self, node, parent, cap_at_node, children):
        """
        Compute total capacitance in the subtree rooted at `node`
        (not including the capacitance at parent side of the edge).
        Uses memoization via a dict passed in.
        """
        memo = {}

        def _rec(n):
            if n in memo:
                return memo[n]
            total = cap_at_node.get(n, 0.0)
            for child in children[n]:
                total += _rec(child)
            memo[n] = total
            return total

        return _rec(node)

    def compute(self):
        """
        Compute the Elmore delay from the voltage source to every reachable node.

        Returns
        -------
        dict : {node_name -> elmore_delay_in_seconds}
                Includes only nodes reachable through series resistors.
        """
        resistors, cap_at_node, source_node = self._build_rc_graph()

        if source_node is None:
            raise ValueError(
                "No voltage source found. Elmore delay requires a V source."
            )
        if not resistors:
            raise ValueError(
                "No series resistors found. Elmore delay requires an RC tree."
            )

        adj        = self._build_adjacency(resistors)
        parent, bfs_order = self._bfs_tree(adj, source_node)

        # Build children list
        children = defaultdict(list)
        for node, info in parent.items():
            if info is not None:
                p_node, _, _ = info
                children[p_node].append(node)

        # Pre-compute subtree capacitance rooted at each node
        subtree_cap = {}
        # Process in reverse BFS order (leaves first)
        for node in reversed(bfs_order):
            c = cap_at_node.get(node, 0.0)
            for child in children[node]:
                c += subtree_cap[child]
            subtree_cap[node] = c

        # Elmore delay to each node = sum of R_k * C_subtree_k
        # along the path from source to that node
        delays = {}
        # delay at source = 0 (it is driven directly)
        path_delay = {source_node: 0.0}

        for node in bfs_order[1:]:  # skip source
            p_node, R, _ = parent[node]
            # Add R * subtree_cap(node) to the delay accumulated at parent
            path_delay[node] = path_delay[p_node] + R * subtree_cap[node]
            delays[node] = path_delay[node]

        return delays

    def report(self):
        """Print a formatted Elmore delay table."""
        delays = self.compute()
        print("\n=== Elmore Delay Report ===")
        print(f"  {'Node':<15} {'Delay (s)':>15} {'Delay (ns)':>15}")
        print(f"  {'-'*45}")
        for node, delay in sorted(delays.items(), key=lambda x: str(x[0])):
            print(f"  {str(node):<15} {delay:>15.6e} {delay*1e9:>15.4f}")
        return delays
