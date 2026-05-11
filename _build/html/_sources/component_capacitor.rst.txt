.. _capacitor-component:

Capacitor Component
===================

This module implements a dynamic capacitor that participates in AC, transient,
and adjoint sensitivity calculations.  It inherits from :class:`~components.base.Component`.

-----------------------------------------------------------------------

.. py:class:: Capacitor(name, data_dict)

   Dynamic capacitor component.

   Inherits all attributes and abstract methods from :class:`Component`.
   The constructor simply forwards to the base class; the key behaviour
   is implemented in the overridden methods below.

-----------------------------------------------------------------------

Overridden Methods
------------------

.. py:meth:: Capacitor.bind_nodes(node_map)

   Maps ``n1`` and ``n2`` node names to matrix indices.
   Also initializes ``prev_current`` for transient state tracking.

.. py:meth:: Capacitor.stamp_ac(Y, sources, w)

   Stamps the AC admittance **j ωC** into the MNA matrix.

   Parameters
   ----------
   Y : scipy.sparse.lil_matrix
       Admittance matrix to be updated.
   sources : np.ndarray
       RHS vector (unused for passive components).
   w : float
       Angular frequency in rad/s.

.. py:meth:: Capacitor.stamp_transient(Y, sources, t, dt, v_prev, method='TR')

   Implements the companion model:

   * Trapezoidal (`'TR'`) → `g_eq = 2C/dt`, `I_eq = g_eq v_diff_prev + prev_current`
   * Backward Euler (`'BE'`) → `g_eq = C/dt`, `I_eq = g_eq v_diff_prev`

   The equivalent conductance and current are added to the matrix and RHS.
   Internal state variables ``_last_g_eq`` and ``_last_I_eq`` are stored for
   later use by :meth:`update_transient_state`.

.. py:meth:: Capacitor.update_transient_state(v_now, method='TR')

   After a transient step, updates ``prev_current`` using the new voltage
   difference.  Only needed for trapezoidal integration.

.. py:meth:: Capacitor.build_adjoint_history(J_hist, dt, v_hat_next, method='BE')

   Builds the RHS vector for the backward adjoint sweep:

   * `I_eq = (C/dt) (v1_hat – v2_hat)` added to `J_hist`.

.. py:meth:: Capacitor.get_sensitivities(VI, PsiPhi, w=0.0, dt=None, V_prev=None, method='TR')

   Computes the sensitivity of the capacitor’s current with respect to its
   capacitance **C**.

   * **AC (w ≠ 0)** – `dI/dC = j ω (V1 – V2)`.
   * **Transient** – uses the change in voltage difference over the step:
     * Trapezoidal: `dI/dC = (2 ΔV)/dt`
     * Backward Euler: `dI/dC = ΔV/dt`
   * **DC** – capacitors are open circuits; sensitivity is zero.

   The final result returned is `{component_name: -adj_diff * dI_dC}` where
   `adj_diff = (Psi_1 – Psi_2)`.

-----------------------------------------------------------------------

Example Usage

.. code-block:: python

   from components.capacitor import Capacitor

   cap = Capacitor('C1', {'type': 'C', 'n1': 1, 'n2': 0, 'value': 1e-6})
   cap.bind_nodes(node_map)
   cap.stamp_ac(Y, sources, w=2*np.pi*1000)

