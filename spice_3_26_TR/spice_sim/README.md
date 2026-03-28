# Python SPICE Simulator

A modular, object-oriented SPICE circuit simulator with Modified Nodal Analysis (MNA), Newton-Raphson nonlinear solving, Adjoint sensitivity analysis, and a Tkinter GUI.

## Project Structure

```
spice_sim/
├── main.py                  # CLI entry point
├── run_gui.py               # Tkinter GUI launcher
├── test_all.py              # Comprehensive test suite (28 tests)
├── core/                    # Core data structures
│   ├── circuit.py           # Circuit class + component factory
│   ├── constants.py         # Physical constants (e, kb, Vt)
│   ├── models.py            # Device physics (diode, MOSFET Level 1)
│   ├── results.py           # SimulationResult data vault
│   └── waveforms.py         # PULSE, SIN, COS, PWL evaluation
├── components/              # Polymorphic MNA-stamping components
│   ├── base.py              # Abstract Component interface
│   ├── resistor.py          # R - linear conductance
│   ├── capacitor.py         # C - dynamic (BE companion model)
│   ├── inductor.py          # L - MNA branch + BE companion
│   ├── voltage_source.py    # V - MNA branch + waveforms
│   ├── current_source.py    # I - RHS stamp + waveforms
│   ├── diode.py             # D - nonlinear (Shockley)
│   ├── mosfet.py            # M - NMOS/PMOS Level 1
│   ├── opamp.py             # E - ideal VCVS
│   └── vccs.py              # G - voltage-controlled current source
├── engines/                 # Numerical simulation engines
│   ├── solver.py            # LU factorization + cascading NR solver
│   ├── dc_engine.py         # DC operating point + DC sweep
│   ├── ac_engine.py         # Small-signal frequency sweep
│   ├── transient_engine.py  # Time-domain Backward Euler integration
│   ├── adjoint_engine.py    # Adjoint sensitivity (all analysis types)
│   └── simulator.py         # Master orchestrator
├── utils/                   # User-facing utilities
│   ├── parser.py            # SPICE netlist parser
│   ├── plotting.py          # Matplotlib visualization
│   └── gui.py               # Tkinter GUI frontend
└── testfiles/               # Example netlists
```

## Supported Analysis Types

- **.OP** — DC Operating Point
- **.DC** — DC Sweep (`.DC V1 0 5 0.1`)
- **.AC** — Small-signal frequency sweep (DEC/LIN)
- **.TRAN** — Transient time-domain simulation

## Supported Components

| Prefix | Component | Sensitivity |
|--------|-----------|-------------|
| R | Resistor | dV/dR |
| C | Capacitor | dV/dC |
| L | Inductor | dV/dL |
| V | Voltage Source | dV/dV_dc |
| I | Current Source | dV/dI_dc |
| D | Diode (Shockley) | dV/dI_S |
| M | MOSFET (Level 1) | dV/dW, dV/dL, dV/dVTO |
| G | VCCS | dV/dG |
| E | VCVS / Ideal OpAmp | dV/dGain |

## Running

```bash
# CLI simulation
python main.py

# GUI
python run_gui.py

# Test suite
python test_all.py
```

## Dependencies

- Python 3.8+
- numpy
- scipy
- matplotlib
- tkinter (for GUI only, ships with Python)
