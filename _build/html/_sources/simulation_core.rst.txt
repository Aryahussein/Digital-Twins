Main Execution Script
=====================

This file serves as the primary entry point for the EDA simulator. It orchestrates 
the pipeline: parsing the netlist, constructing the OOP circuit, executing the 
numerical engines, and visualizing the results.

Functions
---------

.. py:function:: run_simulation_core(netlist_path, output_nodes=None, sensitivity=False, global_adjoint=False, keep_lus=False)

   Runs the complete SPICE simulation pipeline.

   :param str netlist_path: The file path to the SPICE netlist text file.
   :param list, optional output_nodes: Target nodes for Adjoint sensitivities. Defaults to None.
   :param bool, optional sensitivity: Toggles the unified continuous sensitivity tensor. Defaults to False.
   :param bool, optional global_adjoint: Toggles the backward global integral for TRAN. Defaults to False.
   :param bool, optional keep_lus: Forces LU factorization caching. Defaults to False.
   
   :return: A tuple containing ``(Circuit, SimulationResult)`` consisting of the initialized circuit object and the secure data vault with all simulation outputs.
   :rtype: tuple
