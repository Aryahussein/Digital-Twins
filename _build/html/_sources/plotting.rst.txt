.. _plotting-module:

Simulation Plotting Utilities
============================

This module contains a collection of helper functions that take a
:class:`~core.results.SimulationResult` object and generate industry‑standard plots:
Bode (AC), transient waveforms, DC sweep responses, sensitivity curves,
and fault‑comparison figures.  All plotting is performed with Matplotlib in a
headless mode (`Agg`) so the utilities can be used on servers or CI pipelines.

-----------------------------------------------------------------------

.. py:function:: make_bode_plot(result, output_nodes, folder="./figures/ac", name="bodeplot")

   Generates a two‑panel Bode plot (magnitude & phase) for the specified
   `output_nodes`.  The result’s ``sweep_axis`` is used as frequency.

   Parameters
   ----------
   result : :class:`SimulationResult`
   output_nodes : list | str
       Node(s) to include.
   folder, name : str
       Where to save the PNG.

-----------------------------------------------------------------------

.. py:function:: plot_ac_sensitivity(result, output_node, target_component,
                                   folder="./figures/ac", name="ac_sensitivity")

   Plots the AC magnitude of `output_node` together with the absolute value of its
   adjoint sensitivity to `target_component`.

-----------------------------------------------------------------------

.. py:function:: plot_transient(result, output_nodes=None, folder="./figures/tran",
                               name="transient")

   Produces a standard voltage‑vs‑time trace for all nodes that are not MNA
   branch currents.  If *output_nodes* is omitted the function auto‑selects all
   voltage nodes.

-----------------------------------------------------------------------

.. py:function:: plot_transient_sensitivity(result, output_node,
                                            target_component,
                                            folder="./figures/tran",
                                            name="tran_sensitivity")

   Shows the transient voltage of a single node together with its time‑series
   sensitivity to `target_component`.

-----------------------------------------------------------------------

.. py:function:: plot_transient_sensitivity_to_radiation(result, output_node,
                                                          target_component,
                                                          radiation,
                                                          folder="./figures/tran",
                                                          name="tran_sensitivity")

   Similar to the previous function but plots *radiation‑scaled* sensitivities
   for a list of charge values.

-----------------------------------------------------------------------

.. py:function:: plot_integrated_time_series(result, output_node,
                                               target_component,
                                               folder="./figures/tran",
                                               name="tran_sensitivity")

   Same as :func:`plot_transient_sensitivity` – retained for API compatibility.

-----------------------------------------------------------------------

.. py:function:: plot_dc_sweep(result, output_nodes=None,
                               folder="./figures/dc", name="dc_sweep")

   Plots the DC sweep response (voltage or current) versus the sweep source
   voltage.  Auto‑selects all non‑MNA nodes if *output_nodes* is omitted.

-----------------------------------------------------------------------

.. py:function:: plot_dc_sensitivity(result, output_node,
                                      target_component,
                                      folder="./figures/dc",
                                      name="dc_sensitivity")

   Plots the DC sweep curve of a node together with its sensitivity series to
   `target_component`.

-----------------------------------------------------------------------

.. py:function:: print_solution(result)

   Prints a neatly formatted table of node voltages and branch currents.
   Handles both DC/OP (real) and AC (complex magnitude & phase).

-----------------------------------------------------------------------

.. py:function:: plot_fault_comparison(circuit, result, output_node,
                                       threshold_results, delta_F,
                                       folder="./figures/fault",
                                       name="fault_comparison")

   Creates a fault‑comparison plot showing the nominal response, tolerance band,
   and the first short/open fault responses.  Uses `build_xi` and
   `compute_large_change` from :mod:`applications.large_change_sensitivity`.

-----------------------------------------------------------------------

.. py:function:: plot_all_faults(circuit, result, output_node,
                                 threshold_results, delta_F,
                                 folder="./figures/fault",
                                 name="all_faults")

   Extends the previous function by plotting *every* short and open fault
   response found in `threshold_results`.  Each fault is shown as a separate curve.

-----------------------------------------------------------------------

Example usage

.. code-block:: python

   from plotting import make_bode_plot, plot_transient
   result = run_simulation_core("my_netlist.txt")
   make_bode_plot(result, output_nodes=["out"])
   plot_transient(result)

