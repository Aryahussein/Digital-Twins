.. _circuit-module:

Circuit Representation Module
=============================
This module provides the central :class:`~components.Circuit` class, which acts as the primary data 
structure for the simulator. It includes the factory logic to instantiate polymorphic components from parsed netlist dictionaries and generates the Modified Nodal Analysis (MNA) matrix indexing scheme.

.. automodule:: components
   :members:  # automatically pulls in everything that follows

------------------------

Factory Helper
--------------

.. py:function:: create_component(name, data_dict)

    Instantiates the correct Component subclass based on the netlist type.

    :param str name: The unique netlist name of the component (e.g., ``'R1'`` or ``'M_MAIN'``).
    :param dict data_dict: The parsed parameter dictionary containing at minimum a ``'type'`` key
        (e.g. ``{'type': 'R', 'n1': '1', 'n2': '0', 'value': 1000}``).

    :returns: An instantiated component object.
    :rtype: Component

    :raises KeyError: If the ``'type'`` key is missing.
    :raises ValueError: If the component type is not recognized.

------------------------

Circuit Class
-------------

.. py:class:: Circuit(parsed_components)

   Represents the physical circuit, containing components and matrix topology.

   .. attribute:: components

      A list of instantiated polymorphic component objects.

   .. attribute:: components_dict

      A name‑to‑object lookup dictionary for fast access.

   .. attribute:: node_map

      Maps node names (and MNA branch names) to integer matrix indices.

   .. attribute:: total_dim

      The total dimension of the MNA matrix (N × N).

   .. method:: get_idx(node)

      Retrieves the matrix row/column index for a given node.

      :param node: Ground nodes (``0``, ``"0"``, or ``"GND"``) return ``None``.
      :returns: Integer index or ``None``.

   .. method:: _build_node_index()

      Scans components to build the MNA matrix coordinate map.

   .. method:: get_component(name)

      Retrieves a component object by its netlist name.

      :param str name: The component’s netlist identifier.
      :raises KeyError: If the component is not found.

   .. attribute:: differentiable_params

      A master list of all tunable parameters in the circuit.  It aggregates
      :attr:`~components.Component.differentiable_params` from each individual
      component, ensuring no duplicates.

