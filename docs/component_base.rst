.. _component-base-module:

Circuit Component Base Class
===========================

This module defines the abstract :class:`Component` class that all circuit elements
inherit from.  It specifies the interface for matrix stamping, sensitivity
calculations, transient state updates and adjoint operations.

-----------------------------------------------------------------------

.. py:class:: Component(name, data_dict)

   Abstract base class for all SPICE components.

   Attributes
   ----------
   name : str
       Netlist identifier (e.g. ``'R1'``).
   type : str
       Component type character as parsed from the netlist.
   value : float
       Default component value (if applicable).
   data : dict
       Raw dictionary of parameters produced by :class:`NetlistParser`.
   idx_1, idx_2 : int | None
       Matrix indices for the two terminals.  Set by :meth:`bind_nodes`.

   Class Attribute
   ---------------
   IS_NONLINEAR : bool
       ``True`` if the component requires Newton‑Raphson iteration.

-----------------------------------------------------------------------

.. py:meth:: Component.bind_nodes(node_map)

   Translate string or integer node names into matrix indices.

   Parameters
   ----------
   node_map : dict
       Mapping of node identifiers to MNA matrix indices.

-----------------------------------------------------------------------

.. py:property:: Component.differentiable_params

   Returns a list of parameter names that the component can produce sensitivities for.
   The default implementation returns ``[self.name]``.

-----------------------------------------------------------------------

Polymorphic Matrix‑Stamping Methods
-----------------------------------

All components must implement the following methods.  They are called by the various simulation engines.

.. py:meth:: Component.stamp_mna_connection(Y)

   Stamp the MNA branch topology (+1/-1) into the admittance matrix ``Y``.
   Typically this sets up the branch‑current equations for sources and controlled
   elements.

.. py:meth:: Component.stamp_static(Y)

   Stamp time‑invariant linear terms (e.g., conductance of a resistor).

.. py:meth:: Component.stamp_dc(Y, sources)

   Stamp DC values used during operating‑point or initial transient steps.
   The ``sources`` vector holds the RHS contributions.

.. py:meth:: Component.stamp_ac(Y, sources, w)

   Stamp frequency‑dependent complex impedances and AC phasors.
   ``w`` is the angular frequency in rad/s.

.. py:meth:: Component.stamp_transient(Y, sources, t, dt, v_prev, method='TR')

   Stamp dynamic companion models (Backward Euler or Trapezoidal) and time‑varying
   sources for a transient step.

.. py:meth:: Component.update_transient_state(v_now, method='TR')

   Update any internal state after completing a transient step.
   For example, a MOSFET might store its previous drain current.

.. py:meth:: Component.stamp_nonlinear(Y, sources, p_V_guess, V_guess)

   Stamp linearised conductances (gm, gds) and equivalent currents for Newton‑Raphson
   iterations.  ``p_V_guess`` is the voltage from the previous iteration,
   ``V_guess`` the current working guess.

-----------------------------------------------------------------------

Adjoint Sensitivity Methods
--------------------------

.. py:meth:: Component.build_adjoint_history(J_hist, dt, v_hat_next, method='BE')

   Populate the RHS vector for the backward adjoint sweep.
   `J_hist` is a zero array of size ``total_dim``; it will be modified in‑place.

.. py:meth:: Component.get_sensitivities(VI, PsiPhi, w=0.0, dt=None, V_prev=None, method='TR')

   Return a dictionary mapping parameter names to scalar sensitivities
   for the current simulation step.
   The default implementation returns an empty dict and must be overridden by
   concrete components.

-----------------------------------------------------------------------

Helper Method
-------------

.. py:meth:: Component._stamp_branch_equation(Y)

   Shared helper used by components that require MNA branch currents.  It assumes
   ``self.branch_idx`` has been set during :meth:`bind_nodes`.

   The method automatically checks for ``None`` indices before stamping.

-----------------------------------------------------------------------

Example subclass (Resistor)

.. code-block:: python

   class Resistor(Component):
       IS_NONLINEAR = False
       def stamp_static(self, Y):
           g = 1.0 / self.value
           i, j = self.idx_1, self.idx_2
           if i is not None:
               Y[i,i] += g
               Y[j,j] += g
           if i is not None and j is not None:
               Y[i,j] -= g
               Y[j,i] -= g

