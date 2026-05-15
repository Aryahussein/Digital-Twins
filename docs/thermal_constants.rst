.. _thermal-constants:

Thermal Voltage Helper
======================

This tiny module simply pulls constants from :mod:`scipy.constants` and
computes the thermal voltage at room temperature (298 K).  It is handy for
any SPICE‑style circuit simulation that needs ``Vt``.

.. py:data:: e
   :annotation: float

   Elementary charge in coulombs – imported as ``c.e`` from SciPy.

.. py:data:: kb
   :annotation: float

   Boltzmann constant (J K⁻¹) – imported as ``c.k``.

.. py:data:: T
   :annotation: float

   Reference temperature in kelvin (default 298.15 K).

.. py:data:: Vt
   :annotation: float

   Thermal voltage, computed as ``(kb * T) / e`` (~26 mV at room temperature).

Usage example
-------------

>>> from thermal_constants import Vt
>>> Vt
0.025851999...

The module is intentionally lightweight; if you need a different temperature,
edit :data:`T` or recompute :data:`Vt` manually.
