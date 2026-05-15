.. _gui-module:

Graphical User Interface
========================

This module implements a **Tkinter**‑based front end for the Python SPICE simulator.
It provides an interactive editor for netlists, asynchronous simulation execution,
and embedded Matplotlib figures to visualize transient, AC and DC results.

-----------------------------------------------------------------------

.. py:class:: CircuitSimulatorGUI(root, simulation_callback)

   Main application window that coordinates all GUI widgets and
   communicates with the backend simulation engine.

   .. attribute:: root

      :class:`tk.Tk` – The main Tkinter window instance.

   .. attribute:: run_simulation_core

      Callable – Reference to the `run_simulation_core` function defined in
      :mod:`main.py`.  Invoked when the user clicks *Run Simulation*.

   .. attribute:: circuit

      :class:`~core.circuit.Circuit | None` – The parsed circuit object returned
      by the backend after a simulation finishes.

   .. attribute:: result

      :class:`~core.results.SimulationResult | None` – The data vault holding all
      forward and sensitivity results.

-----------------------------------------------------------------------

Key Methods
-----------

.. py:meth:: CircuitSimulatorGUI.load_file()

   Opens a file dialog, reads the selected netlist into the text editor,
   updates the *Run Simulation* button state, and stores the path for later use.

.. py:meth:: CircuitSimulatorGUI.run_simulation()

   Persists any live edits back to disk, disables the run button, and
   starts the heavy numerical engine in a background thread so that
   the UI remains responsive.

.. py:meth:: CircuitSimulatorGUI._simulation_thread()

   Worker function executed in a separate thread.  Calls
   :func:`run_simulation_core`, captures the resulting ``circuit`` and
   ``result``, then schedules ``_simulation_complete`` on the main Tkinter
   event loop using `root.after`.

.. py:meth:: CircuitSimulatorGUI._simulation_error(error)

   Restores the UI state and displays an error dialog if the backend raises
   an exception.

.. py:meth:: CircuitSimulatorGUI._simulation_complete()

   Populates node lists, parameter comboboxes, updates plot or data tabs,
   and re‑enables the *Run Simulation* button once results are available.

.. py:meth:: CircuitSimulatorGUI.on_selection_change(event)

   Triggered when a node is selected in the listbox or a sensitivity
   target is chosen from the combobox.  Calls :func:`update_plot`.

.. py:meth:: CircuitSimulatorGUI.update_data_tab_op()

   Fills the *Nodal Analysis Data* tab with the DC operating point voltages,
   currents and any available sensitivities for each node.

.. py:meth:: CircuitSimulatorGUI.update_plot()

   Generates Matplotlib figures on the fly.  Handles four different analysis
   types:

   * **AC / Bode** – magnitude/phase plots, optionally overlaying sensitivity
     curves.
   * **Transient & DC Sweep** – time or sweep voltage traces with optional
     sensitivity series.

   The method uses :class:`matplotlib.figure.Figure` embedded in a
   `FigureCanvasTkAgg` widget.  It also updates axis labels,
   legends and grid lines according to the selected analysis type.

-----------------------------------------------------------------------

Additional Details
------------------

* **Threading** – Simulation runs on a daemon thread (`threading.Thread`) to keep
  the UI responsive.  All GUI updates are marshalled back to the main thread via
  `root.after`.
* **Plotting** – The widget layout uses a split pane: left side is a
  read‑only text area for the netlist, right side is a notebook with two tabs:
  *Waveform Plot* (Matplotlib) and *Nodal Analysis Data* (plain text).
* **Node Naming** – Nodes are displayed as `V(node)` or `I(V1)` depending on whether
  they represent voltage nodes or MNA branch currents.  Voltage nodes are auto‑selected.
* **Sensitivity UI** – A combobox lists all parameters that the adjoint engine has
  identified for each node; selecting one overlays its sensitivity curve.

-----------------------------------------------------------------------

Example Usage (from a main script)

.. code-block:: python

   import tkinter as tk
   from gui import CircuitSimulatorGUI
   from main import run_simulation_core

   root = tk.Tk()
   app = CircuitSimulatorGUI(root, run_simulation_core)
   root.mainloop()

