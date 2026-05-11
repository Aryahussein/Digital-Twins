.. _adjoint-engine-module:

Adjoint Sensitivity Engine
==========================

This module implements :class:`AdjointEngine`, which evaluates parameter sensitivities using the adjoint network method.
The engine supports both *continuous* (steady‑state, AC, DC, OP) and *global transient* sensitivity calculations.

-----------------------------------------------------------------------

.. py:class:: AdjointEngine(circuit, output_nodes)

   Initializes the adjoint solver for a given circuit.

   .. attribute:: circuit

      :class:`~core.circuit.Circuit` – The main circuit object containing components.

   .. attribute:: output_nodes

      ``list[str]`` – Nodes that will be used as objectives in the sensitivity analysis.
      The constructor resolves node names against :attr:`circuit.node_map` so that
      strings, integers and mixed types all work correctly.

-----------------------------------------------------------------------

.. py:method:: AdjointEngine._solve_adjoint(lu, target, base_rhs=None, is_complex=False)

   Internal helper that solves the transposed matrix equation for a single target node.
   The method can be reused by both transient and continuous solvers.

   Parameters
   ----------
   lu : LU factorization object
       From the forward solver (e.g., SciPy or custom).
   target : str | None
       Node name that will receive an impulse of +1.0 in the RHS.
   base_rhs : array_like, optional
       Optional additional RHS vector to be added before solving.
   is_complex : bool, optional
       Flag indicating whether complex arithmetic is required (AC analysis).

   Returns
   -------
   array_like
       Solution vector `ψ` (adjoint variable) with shape `(total_dim,)`.

-----------------------------------------------------------------------

.. py:method:: AdjointEngine.compute_transient(time_array, V_forward, list_of_lus, dt, method="BE")

   Calculates exact *global* adjoint sensitivities via backward time integration.
   The result is a dictionary containing:

   * **Integrated_Transient** – scalar integrals for each parameter and target node.
   * **Time_Series** – a :class:`~core.results.SensitivityData` tensor that stores the full time‑series of sensitivities.
   * **Raw_Adjoint_History** – raw adjoint vectors at every step (useful for fault analysis).

   Parameters
   ----------
   time_array : 1‑D array_like
       Simulation time points.
   V_forward : list/array
       Forward solution vector(s) at each time step.
   list_of_lus : list of LU objects
       LU factorizations from the forward transient solver, one per time step.
   dt : float
       Time step size.
   method : str, optional
       Integration scheme used in the forward pass (`"BE"` = Backward Euler,
       `"TR"` = Trapezoidal).  The same method is used for the backward solve.

   Notes
   -----
   * The algorithm performs a reverse‑time sweep to compute adjoint variables
     and then a forward integration to accumulate sensitivities.
   * All parameters that are differentiable in the circuit (from
     :attr:`circuit.differentiable_params`) are included in the tensor.

-----------------------------------------------------------------------

.. py:method:: AdjointEngine.compute_sensitivities(sweep_axis, V_forward, list_of_lus,
                                                   domain="static", method="TR", dt=0.0)

   Unified continuous sensitivity solver for AC, DC, OP and local‑DC (transient) analyses.
   The routine builds a :class:`~core.results.SensitivityData` tensor that
   contains the partial derivatives of each output node with respect to every
   differentiable parameter.

   Parameters
   ----------
   sweep_axis : 1‑D array_like
       Independent variable array (time, frequency, voltage, or single value for OP).
   V_forward : list/array
       Forward solution vectors at each sweep point.
   list_of_lus : list of LU objects
       LU factorizations from the forward solver, one per step.
   domain : str, optional
       One of ``"time"``, ``"frequency"``, ``"voltage"``, or ``"static"``.
   method : str, optional
       Integration scheme used for transient problems (`"TR"` or `"BE"`).
   dt : float, optional
       Time step size (used only when ``domain == "time"``).

   Returns
   -------
   SensitivityData
       Tensor containing all continuous sensitivities.  The object also stores the
       raw adjoint vectors in its ``adjoint_vectors`` attribute for later fault analysis.

-----------------------------------------------------------------------

Example usage
-------------

.. code-block:: python

   from core.circuit import Circuit
   from adjoint_engine import AdjointEngine

   # Assume `circuit` has been built and a forward solution is available.
   engine = AdjointEngine(circuit, output_nodes=['out'])
   tensor = engine.compute_sensitivities(
       sweep_axis=np.linspace(0, 1e-3, 100),
       V_forward=forward_results,
       list_of_lus=luts,
       domain='time',
       method='TR'
   )

