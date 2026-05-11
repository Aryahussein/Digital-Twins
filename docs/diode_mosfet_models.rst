.. _diode-and-nmos-models:

Diode & NMOS Level‑1 Models
===========================

This module implements two small‑signal models that are useful in SPICE‑style circuit simulation:

* :func:`evaluate_diode` – Shockley diode model with sensitivity to the saturation current.  
* :func:`evaluate_nmos` – Level 1 MOSFET (NMOS) model including gradients for *V<sub>TO</sub>* and *β<sub>n</sub>*.

The functions return dictionaries containing the large‑signal quantities as well as the required small‑signal conductances and sensitivity gradients.

-----------------------------------------------------------------------

.. py:function:: evaluate_diode(vd, Is, Vt)

    Evaluates the Shockley diode model.

    :param float vd: Voltage across the diode (Anode – Cathode).
    :param float Is: Saturation current (e.g., 1e‑14 A).
    :param float Vt: Thermal voltage (e.g., 0.02585 V).

    :returns: ``dict`` with keys:

      * ``"I_D"`` – Large‑signal current in amperes.
      * ``"gd"``   – Small‑signal conductance dI/dv (siemens).
      * ``"dId_dIs"``
        – Sensitivity gradient ∂I/∂Is.

    :rtype: dict

-----------------------------------------------------------------------

.. py:function:: evaluate_nmos(vgs, vds, VTO, Bn)

    Evaluates the Level 1 NMOS model with sensitivity gradients.

    :param float vgs: Gate‑source voltage.
    :param float vds: Drain‑source voltage.
    :param float VTO: Threshold voltage (V<sub>TO</sub>).
    :param float Bn: Transconductance parameter β<sub>n</sub> = μₙC<sub>ox</sub>(W/L).

    :returns: ``dict`` with keys:

      * ``"I_D"``   – Drain current (A).
      * ``"gm"``    – Transconductance dI/dV<sub>GS</sub>.
      * ``"gds"``   – Output conductance dI/dV<sub>DS</sub>.
      * ``"dId_dBn"``
        – Sensitivity ∂I/∂β<sub>n</sub>.
      * ``"dId_dVTO"``
        – Sensitivity ∂I/∂V<sub>TO</sub>.

    :rtype: dict
