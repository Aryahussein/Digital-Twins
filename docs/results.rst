.. _results-module:

Simulation Result & Sensitivity Data
===================================

This module provides two high‑level containers that are returned by the simulator:
```

* :class:`SensitivityData` – A unified 3‑D tensor for all sensitivity results.
* :class:`SimulationResult` – The “vault” that stores raw simulation outputs and, when available, the associated sensitivities.

-----------------------------------------------------------------------

.. py:class:: SensitivityData(sweep_axis, param_names, output_nodes, domain="static")

    Holds a 3‑D tensor of shape *(Parameters × Output Nodes × Sweep Steps)*.  
    The class offers convenient query helpers that hide the raw NumPy indexing from the user.

    .. attribute:: sweep_axis

       :class:`numpy.ndarray` – Independent variable array (time, frequency, voltage, or single value for .OP).

    .. attribute:: domain

       Physical domain of the sweep (`"time"`, `"frequency"`, `"voltage"`, `"static"`).

    .. attribute:: param_names

       Ordered list of parameter names (axis‑0 labels).

    .. attribute:: output_nodes

       Ordered list of output node names (axis‑1 labels).

    .. attribute:: data

       :class:`numpy.ndarray` – The underlying 3‑D sensitivity tensor.

    .. attribute:: param_index
    .. attribute:: output_index

       Dictionaries mapping names → indices for fast lookup.

    .. method:: get_sweep_series(param, output_node)

       Returns the 1‑D array of partial derivatives *∂output/∂param* across the sweep axis.

       :param str param: Parameter name.
       :param str|int output_node: Target node.
       :returns: ``numpy.ndarray`` – 1‑D sensitivity series.

    .. method:: get_matrix_at_step(step_idx)

       Returns a 2‑D matrix *(n_params × n_outputs)* at the given sweep index.

       :param int step_idx: Sweep index.
       :returns: ``numpy.ndarray``.

    .. method:: print_matrix_at_step(step_idx, output_node, sens_parameter)

       Nicely formatted console output of the sensitivity matrix for a single
       sweep step.  Useful for quick debugging or CLI reporting.

-----------------------------------------------------------------------

.. py:class:: SimulationResult(analysis_type, sweep_axis, VI_matrix, node_map)

    A container that holds all information produced by a simulation run:

    * The raw solution matrix (`VI`).
    * The node/branch mapping.
    * Optional :attr:`sensitivities` (an instance of :class:`SensitivityData`).

    .. attribute:: type

       Analysis type string (`'.TRAN'`, `'.AC'`, `'.DC'`, `'.OP'`).

    .. attribute:: sweep_axis

       Independent variable array.

    .. attribute:: VI

       Raw solver output matrix (2‑D for sweeps, 1‑D for .OP).

    .. attribute:: node_map

       Mapping of node/branch names → column indices in :attr:`VI`.

    .. attribute:: sensitivities

       Optional :class:`SensitivityData` instance.

    .. method:: get_vhats(node)

       Retrieves the raw adjoint vector history (useful for fault analysis).

    .. method:: _resolve_node_key(node)

       Internal helper that normalises user‑supplied node identifiers
       to the exact key stored in :attr:`node_map`.

    .. method:: get_voltage(node)

       Returns the voltage or branch current array for *node*.
       Handles complex values for .AC, real values otherwise.

    .. method:: get_sensitivity_parameters(node)

       List of parameters that have sensitivity data for *node*.
       Useful for populating UI dropdowns.

    .. method:: get_sensitivity(node, param)

       Returns the sensitivity gradient for a given output node and parameter.
       For single‑point analyses it returns a scalar; otherwise a 1‑D array
       across the sweep axis.
