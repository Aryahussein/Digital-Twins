.. _dc-engine-module:

DC Analysis Engine
==================

This module implements :class:`DCEngine`, which is responsible for building the static
Modified Nodal Analysis (MNA) topology and solving the DC operating point.
The resulting bias solution is required before any AC or transient analysis can begin.

-----------------------------------------------------------------------

.. py:class:: DCEngine(circuit, is_complex, is_nonlinear)

   Core class that compiles the circuit’s base matrices and performs
   steady‑state solves.

   .. attribute:: circuit

      :class:`~core.circuit.Circuit` – The fully populated circuit object.

   .. attribute:: is_complex

      ``bool`` – True if an AC analysis has been requested; forces complex
      matrix data types.

   .. attribute:: is_nonlinear

      ``bool`` – True if any component in the circuit requires a Newton‑Raphson
      iteration (e.g. MOSFETs, diodes).

-----------------------------------------------------------------------

.. py:meth:: DCEngine.build_base_matrices()

   Builds the pristine static base matrices that will be reused for every
   transient or sweep step.

   Returns
   -------
   tuple
       ``(Y_base, sources_base)`` – where

       * **Y_base** : :class:`scipy.sparse.lil_matrix` – mutable sparse admittance matrix.
       * **sources_base** : 1‑D :class:`numpy.ndarray` – (currently unused but kept for API symmetry).

   Notes
   -----
   * All time‑invariant components (resistors, static conductances, etc.) are stamped once here.
   * The method is called exactly once per simulation run.

-----------------------------------------------------------------------

.. py:meth:: DCEngine.compute_dc_bias(Y_base_lil, v_ini=None, print_stuff=True)

   Computes the DC operating point for a given base matrix.

   Parameters
   ----------
   Y_base_lil : :class:`scipy.sparse.lil_matrix`
       Static base admittance matrix.
   v_ini : :class:`numpy.ndarray`, optional
       Initial guess vector to accelerate Newton‑Raphson convergence.  
       Defaults to a zero vector (0 V).
   print_stuff : bool, optional
       Toggle console convergence logging.

   Returns
   -------
   tuple
       ``(lu_factorization, VI_solution_array)``

       * **lu_factorization** – LU factorization object returned by the solver.
         For linear solves this is simply a SciPy LU decomposition; for nonlinear
         it contains the final Newton‑Raphson state.
       * **VI_solution_array** – 1‑D array of node voltages (and branch currents)
         at DC.

   Notes
   -----
   * Capacitors are treated as open circuits and inductors as short circuits.
   * If ``is_nonlinear`` is True, a :class:`~engines.solver.NonlinearSolver` is used;
     otherwise the linear system is solved with :func:`solve_linear_circuit`.

-----------------------------------------------------------------------

.. py:meth:: DCEngine.compute_dc_sweep(Y_base_lil, source_name, start, stop, step, keep_lus=False)

   Performs a large‑signal DC sweep (``.DC`` analysis).

   Parameters
   ----------
   Y_base_lil : :class:`scipy.sparse.lil_matrix`
       Base admittance matrix.
   source_name : str
       Netlist name of the component to be swept (e.g. ``'V1'``).
   start, stop, step : float
       Sweep bounds and increment.
   keep_lus : bool, optional
       If True, stores the LU factorization for each sweep point.

   Returns
   -------
   tuple
       ``(sweep_axis, VIs, list_of_lus)``

       * **sweep_axis** : 1‑D array of sweep values.
       * **VIs** : 2‑D array where each row is the DC solution at a sweep point.
       * **list_of_lus** : Optional list of LU objects corresponding to each step.

   Notes
   -----
   * The component’s value is temporarily overridden during the sweep and restored afterwards.
   * A previous DC solution (`current_guess`) is reused as an initial guess for subsequent points,
     improving convergence speed.

-----------------------------------------------------------------------

Example usage
-------------

.. code-block:: python

   from core.circuit import Circuit
   from dc_engine import DCEngine

   # Assume `circuit` has been built and we want a DC sweep of V1
   engine = DCEngine(circuit, is_complex=False, is_nonlinear=True)
   Y_base, _ = engine.build_base_matrices()
   axis, Vs, lus = engine.compute_dc_sweep(Y_base, 'V1', 0.0, 5.0, 0.5)

