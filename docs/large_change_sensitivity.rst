.. _large-change-sensitivity-module:

Large‑Change Sensitivity Utilities
=================================

This module implements the Sherman‑Morrison / Kron’s formula for evaluating the exact effect of adding a resistance between two nodes in a circuit.  
It requires only the LU factorisation of the Y matrix and the nominal solution vector.

-----------------------------------------------------------------------

.. py:function:: build_xi(n, idx_k, idx_l)

   Build the connection vector **xi_kl**.

   Parameters
   ----------
   n : int
       Dimension of the Y matrix (number of nodes/branches).
   idx_k, idx_l : int | None
       Indices of the two nodes between which the fault is added.
       ``None`` may be passed for ground or virtual nodes.

   Returns
   -------
   np.ndarray
       Vector with +1 at `idx_k`, -1 at `idx_l`, zeros elsewhere.

-----------------------------------------------------------------------

.. py:function:: compute_large_change(lu, xi_kl, v, R, output_idx)

   Compute the exact change in an output voltage when a resistance **R**
   is inserted between two nodes.

   The formula used:

   .. math::

      \Delta V_{\text{out}}
        = -\frac{v_{oc}}{\,R + R_{TH}\,}
          \bigl(Y^{-1} \xi_{kl}\bigr)_{\text{output}}

   where

   * :math:`v_{oc} = \xi_{kl}^{T} v` – open‑circuit voltage between the nodes.
   * :math:`R_{TH} = \xi_{kl}^{T} Y^{-1} \xi_{kl}` – Thevenin resistance seen from
     the two nodes.
   * :math:`Y^{-1}\xi_{kl}` is obtained by a single forward solve using the supplied
     LU factorisation.

   Parameters
   ----------
   lu : LU factorisation object
       Result of a prior linear solve (e.g. SciPy or custom).
   xi_kl : np.ndarray
       Connection vector for the fault branch.
   v : np.ndarray
       Nominal solution vector at this time step.
   R : float
       Value of the added resistance.
   output_idx : int
       Matrix index of the node whose voltage change is required.

   Returns
   -------
   float
       The exact change in the output voltage, `ΔV_out`.

-----------------------------------------------------------------------

Example usage

.. code-block:: python

   from large_change_sensitivity import build_xi, compute_large_change

   xi = build_xi(n=10, idx_k=2, idx_l=5)
   delta_v = compute_large_change(lu, xi, v_nominal, R=100.0, output_idx=3)

