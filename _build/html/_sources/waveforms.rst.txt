.. _waveforms-module:

Time‑Domain Waveform Module
===========================

This module implements a small library of transient source generators that are used during
transient analysis.  It supports the common SPICE waveforms **PULSE**, **SIN/SINE/COS** and **PWL**
( Piece‑Wise Linear ).  The :class:`Waveform` class parses the user‑supplied dictionary,
stores the waveform type, and evaluates the instantaneous voltage or current at any time `t`.

-----------------------------------------------------------------------

.. py:class:: Waveform(source_dict)

   Evaluates time‑domain source functions for transient simulations.

   .. attribute:: type

      ``str`` – The waveform type identifier (e.g. ``'PULSE'``, ``'SIN'``, ``'COS'``, ``'PWL'``).

   .. attribute:: params

      ``dict`` – All parameters that define the waveform shape, as parsed from a SPICE netlist.

   .. method:: get_value(t)

      Computes the instantaneous value of the source at simulation time `t`.

      :param float t: Current simulation time in seconds.
      :returns: ``float`` – Instantaneous voltage (V) or current (A).

      The implementation supports the following waveform types:

      * **PULSE**  
        Parameters: ``V1, V2, TD, TR, TF, PW, PER``  
        Returns a piecewise linear pulse with optional rise/fall times.

      * **SIN / SINE / COS**  
        Parameters: ``VOFF, VAMP, FREQ, PHASE`` (phase is supplied in degrees).  
        Returns either a sine or cosine waveform depending on the type string.

      * **PWL**  
        Parameter: ``TIME_VOLTAGE_PAIRS`` – list of ``(time, voltage)`` tuples.  
        Uses :func:`numpy.interp` for linear interpolation between points.

      * **Static / Unknown** – falls back to the value stored in ``params['value']`` (defaults to 0).

-----------------------------------------------------------------------

Example usage
-------------

.. code-block:: python

   from waveforms import Waveform

   # PULSE example
   pulse = Waveform({
       "type": "PULSE",
       "V1": 0.0,
       "V2": 5.0,
       "TD": 1e-6,
       "TR": 10e-9,
       "TF": 10e-9,
       "PW": 100e-9,
       "PER": 200e-9
   })

   voltage_at_t = pulse.get_value(5e-7)  # -> 5.0 V

