.. _netlist-parser-module:

SPICE Netlist Parser
====================

This module reads standard SPICE‑formatted text files, handles line continuations,
extracts scaling suffixes (k, MEG, u, etc.) and compiles the data into dictionaries that
the :class:`~core.circuit.Circuit` factory can consume.

-----------------------------------------------------------------------

.. py:class:: NetlistParser

   Core parser for SPICE netlists.

   .. attribute:: components

      ``dict`` – Parsed component data keyed by component name (e.g. ``'R1'``).

   .. attribute:: models

      ``dict`` – Parsed ``.MODEL`` definitions.

   .. attribute:: analyses

      ``dict`` – Parsed simulation commands (.TRAN, .AC, .DC, .OP).

   .. attribute:: current_line

      ``int`` – Line number of the line currently being parsed (for error reporting).

   .. attribute:: raw_line_text

      ``str`` – Raw text of the current logical line.

   .. attribute:: dispatch_registry

      ``dict`` – Maps SPICE prefix letters to the corresponding mini‑parser
      methods used by :func:`_parse_component`.

-----------------------------------------------------------------------

Key Methods
-----------

.. py:meth:: NetlistParser.parse(file_path)

   Main entry point.  Reads *file_path*, preprocesses line continuations, parses every
   logical line, attaches models and returns a tuple ``(components_dict,
   analyses_dict)`` ready for the simulator.

.. py:meth:: NetlistParser._parse_command(tokens)

   Dispatches dot‑commands (.TRAN, .AC, .MODEL, .DC, .OP, .OPTIONS).

.. py:meth:: NetlistParser._parse_component(tokens)

   Routes component lines to the appropriate mini‑parser based on the first
   character of the component name.

-----------------------------------------------------------------------

Component Mini‑Parsers
---------------------

* ``_parse_passive`` – Resistors, capacitors, inductors.  
  Returns dict with keys: ``type``, ``n1``, ``n2``, ``value``.

* ``_parse_diode`` – Diodes; may include a model name or saturation current.

* ``_parse_mosfet`` – MOSFETs (four‑terminal); parses node list, model and
  instance parameters.

* ``_parse_source`` – Voltage/Current sources; handles DC, AC and transient
  definitions (PULSE, SIN, PWL etc.).

* ``_parse_vccs``, ``_parse_vcvs``, ``_parse_cccs``, ``_parse_ccvs`` – Various
  controlled sources.

-----------------------------------------------------------------------

Utility Functions
-----------------

.. py:staticmethod:: NetlistParser._parse_node(node_str)

   Converts node identifiers to integers or keeps them as strings.  
   Recognises the ground nodes ``0`` and ``GND``.

.. py:meth:: NetlistParser._parse_source_def(tokens)

   Extracts DC, AC magnitude/phase and transient source definitions from
   a source line.

.. py:meth:: NetlistParser._parse_tran_func(func_str)

   Parses SPICE transient functions: **PULSE**, **SIN/SINE/COS** and **PWL**,
   returning a dictionary of parameters.

.. py:staticmethod:: NetlistParser._parse_value(value_str)

   Converts SPICE numbers with scaling suffixes (e.g. ``10k``, ``5u``) into floats.
   Supports multipliers: T, G, MEG, K, MIL, M, U, N, P, F.

.. py:meth:: NetlistParser._extract_params(tokens)

   Turns a list of ``KEY=VALUE`` tokens into a dictionary and returns any
   remaining tokens.

-----------------------------------------------------------------------

Helper Methods
--------------

* ``_preprocess_lines(lines)`` – Handles line continuations starting with ``+``,
  strips comments, and yields tuples ``(line_number, logical_line)``.

* ``_attach_models()`` – Binds parsed ``.MODEL`` definitions to the relevant
  components, upgrading their type and adding ``model_params``.

* ``_throw_error(msg)`` – Raises a :class:`ValueError` with line number and raw text.

-----------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python

   parser = NetlistParser()
   components, analyses = parser.parse("my_circuit.txt")
   circuit = Circuit(components)
   simulator = Simulator(circuit, analyses)
   result = simulator.execute_analysis(sensitivity=True)

