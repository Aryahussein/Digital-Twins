from dc_solver import run_dc
from ac_solver import run_ac
from tran_solver import run_tran

from MODELS.resistor import Resistor
from MODELS.capacitor import Capacitor
from MODELS.inductor import Inductor
from MODELS.voltage_source import VoltageSource
from MODELS.current_source import CurrentSource
from MODELS.opamp import OpAmp
from MODELS.diode import Diode

import numpy as np
import re


# ==========================================================
# Utility
# ==========================================================

def parse_value(val):
    val = val.lower()

    scale = {
        'meg': 1e6,
        't': 1e12,
        'g': 1e9,
        'k': 1e3,
        'm': 1e-3,
        'u': 1e-6,
        'n': 1e-9,
        'p': 1e-12
    }

    for s in sorted(scale.keys(), key=len, reverse=True):
        if val.endswith(s):
            return float(val[:-len(s)]) * scale[s]

    return float(val)


def strip_comments(line):
    for c in ['*', ';']:
        if c in line:
            line = line.split(c, 1)[0]
    return line.strip()


def generate_ac_frequencies(ac_sweep):

    sweep_type, npts, fstart, fstop = ac_sweep

    if sweep_type == 'dec':
        decades = np.log10(fstop / fstart)
        total_pts = int(npts * decades)
        return np.logspace(np.log10(fstart), np.log10(fstop), total_pts)

    elif sweep_type == 'oct':
        octaves = np.log2(fstop / fstart)
        total_pts = int(npts * octaves)
        return np.logspace(np.log10(fstart), np.log10(fstop), total_pts)

    elif sweep_type == 'lin':
        return np.linspace(fstart, fstop, npts)

    else:
        raise RuntimeError("Unsupported AC sweep type")


# ==========================================================
# Detect analysis
# ==========================================================

def detect_analysis(netlist_file):
    with open(netlist_file) as f:
        for line in f:
            line = line.strip().lower()
            if line.startswith('.op'):
                return 'dc'
            if line.startswith('.ac'):
                return 'ac'
            if line.startswith('.tran'):
                return 'tran'
    raise RuntimeError("No analysis directive found")


# ==========================================================
# NETLIST → COMPONENT FACTORY
# ==========================================================

def build_circuit(netlist_file):

    components = []
    nodes = set()

    voltages = []
    opamps = []

    ac_sweep = None
    tran_params = None
    sens_node = None
    print_requests = []

    with open(netlist_file) as f:

        for raw in f:

            line = strip_comments(raw)
            if not line:
                continue

            t = line.split()
            name = t[0].lower()

            # ---------- Passive ----------
            if name.startswith('r'):
                _, n1, n2, val = t
                components.append(Resistor(t[0], n1, n2, parse_value(val)))
                nodes.update([n1, n2])

            elif name.startswith('c'):
                _, n1, n2, val = t
                components.append(Capacitor(t[0], n1, n2, parse_value(val)))
                nodes.update([n1, n2])

            elif name.startswith('l'):
                _, n1, n2, val = t
                components.append(Inductor(t[0], n1, n2, parse_value(val)))
                nodes.update([n1, n2])

            elif name.startswith('i'):
                _, n1, n2, val = t
                components.append(CurrentSource(t[0], n1, n2, parse_value(val)))
                nodes.update([n1, n2])

            # ---------- Voltage sources ----------
            elif name.startswith('v'):

                n1, n2 = t[1], t[2]

                match = re.search(r'(\w+)\((.*?)\)', line)

                if match:
                    source_type = match.group(1).upper()
                    params = [parse_value(x) for x in match.group(2).split()]
                else:
                    source_type = "DC"
                    params = [parse_value(t[3])]

                voltages.append((t[0], n1, n2, source_type, params))
                nodes.update([n1, n2])

            # ---------- Opamp ----------
            elif name.startswith('o'):
                opamps.append(t)
                nodes.update([t[1], t[2], t[3]])

            # ---------- Diode ----------
            elif name.startswith('d'):
                _, n1, n2 = t[:3]
                components.append(Diode(t[0], n1, n2))
                nodes.update([n1, n2])

            # ---------- Analysis ----------
            elif name == '.tran':
                _, dt, tstop = t
                tran_params = (parse_value(dt), parse_value(tstop))

            elif name == '.ac':
                _, sweep_type, npts, fstart, fstop = t
                ac_sweep = (sweep_type.lower(), int(npts), parse_value(fstart), parse_value(fstop))

            elif name == '.sens':
                expr = t[1].lower()
                sens_node = expr[2:-1]

            elif name == '.print':
                print_requests.append((t[1], t[2]))

    # ---------- Node indexing ----------
    nodes.discard("0")
    nodes = sorted(nodes)
    node_index = {n: i for i, n in enumerate(nodes)}

    N = len(nodes)

    # ---------- Voltage sources ----------
    for k, (name, n1, n2, stype, params) in enumerate(voltages):
        components.append(VoltageSource(name, n1, n2, stype, params, k))

    Mv = len(voltages)

    # ---------- Opamps ----------
    for k, t in enumerate(opamps):
        _, nplus, nminus, nout, gain = t
        components.append(OpAmp(t[0], nplus, nminus, nout, parse_value(gain), k))

    Mo = len(opamps)

    return components, node_index, N, Mv, Mo, ac_sweep, tran_params, sens_node, print_requests


# ==========================================================
# MAIN
# ==========================================================

NETLIST_FILE = "test_circuit.sp"

if __name__ == "__main__":

    print("=== RUN START ===")

    analysis = detect_analysis(NETLIST_FILE)
    print("Detected analysis:", analysis)

    components, node_index, N, Mv, Mo, ac_sweep, tran_params, sens_node, print_requests = build_circuit(NETLIST_FILE)

    # ================= DC =================
    if analysis == 'dc':

        run_dc(
            components,
            node_index,
            N,
            Mv,
            Mo,
            sens_node,
            print_requests
        )

    # ================= AC =================
    elif analysis == 'ac':

        if ac_sweep is None:
            raise RuntimeError("No .ac directive found")

        freqs = generate_ac_frequencies(ac_sweep)

        # Run DC for operating point
        x_op = run_dc(
            components,
            node_index,
            N,
            Mv,
            Mo,
            sens_node=None,
            print_requests=None
        )

        run_ac(
            components,
            node_index,
            N,
            Mv,
            Mo,
            freqs,
            sens_node,
            print_requests,
            x_op=x_op
        )

    # ================= TRAN =================
    elif analysis == 'tran':

        if tran_params is None:
            raise RuntimeError("No .tran directive found")

        dt, tstop = tran_params

        run_tran(
            components,
            node_index,
            N,
            Mv,
            Mo,
            dt,
            tstop,
            sens_node,
            print_requests
        )