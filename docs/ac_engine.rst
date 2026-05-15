.. _ac-engine-module:

AC Analysis Engine
==================

This module implements :class:`ACEngine`, which performs small‑signal frequency domain simulations.
The routine linearises all nonlinear components around a supplied DC operating point,
then sweeps the specified frequency range to compute complex phasors at every node.

-----------------------------------------------------------------------

.. py:class:: ACEngine(circuit, is_nonlinear)

   Core class that runs AC sweeps.

   .. attribute:: circuit

      :class:`~core.circuit.Circuit` – The fully populated circuit object.

   .. attribute:: is_nonlinear

      ``bool`` – True if the circuit contains nonlinear devices (diodes,
      MOSFETs, etc.) that must be linearised at each bias point.

-----------------------------------------------------------------------

.. py:method:: ACEngine._solve_single_point(w, Y_small_signal_base)

   Internal helper that solves the AC circuit for a single angular frequency.

   Parameters
   ----------
   w : float
       Angular frequency (rad/s) = 2π·f.
   Y_small_signal_base : :class:`scipy.sparse.lil_matrix`
       Pre‑linearised admittance matrix that already contains static resistors
       and the linearised conductances of any nonlinear devices.

   Returns
   -------
   tuple
       ``(lu_factorization, VI_complex_array)`` – LU factorisation object and the
       complex solution vector at this frequency.

-----------------------------------------------------------------------

.. py:method:: ACEngine.run(Y_base_lil, VI_dc, start_freq, stop_freq, points,
                           sweep_type="DEC", keep_lus=False)

   Executes an AC frequency sweep.

   Parameters
   ----------
   Y_base_lil : :class:`scipy.sparse.lil_matrix`
       Static base admittance matrix (time‑invariant part of the circuit).
   VI_dc : :class:`numpy.ndarray`
       DC operating point vector that was previously computed.
   start_freq, stop_freq : float
       Sweep bounds in hertz.
   points : int
       Number of frequency samples to generate.
   sweep_type : str, optional
       ``"DEC"`` (logarithmic), ``"LIN"``, ``"OCT"`` or ``"LIST"``.  Default is ``"DEC"``.
   keep_lus : bool, optional
       If True, caches the LU factorisation for every frequency point – required by
       the adjoint sensitivity engine.

   Returns
   -------
   tuple
       ``(frequencies, VIs, list_of_lus)``

       * **frequencies** – 1‑D array of sweep frequencies.
       * **VIs** – 2‑D array (n_frequencies × n_nodes) containing complex phasors.
       * **list_of_lus** – Optional list of LU objects, one per frequency.

   Notes
   -----
   * The routine first “bakes” the small‑signal conductances into a base matrix
     (`Y_small_signal_base`) once.  This optimisation avoids re‑stamping at every
     frequency step.
   * For nonlinear circuits it calls each component’s
     :py:meth:`~components.Component.stamp_nonlinear` with the DC bias vector.

-----------------------------------------------------------------------

Example usage
-------------

.. code-block:: python

   from core.circuit import Circuit
   from ac_engine import ACEngine

   engine = ACEngine(circuit, is_nonlinear=True)
   freqs, results, luts = engine.run(
       Y_base_lil=Y_base,
       VI_dc=dc_solution,
       start_freq=1e3,
       stop_freq=1e6,
       points=100
   )

