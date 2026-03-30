import sys
import re
import numpy as np

from dc_solver   import run_dc
from dc_sweep    import run_dc_sweep
from ac_solver   import run_ac
from tran_solver import run_tran

from MODELS.resistor       import Resistor
from MODELS.capacitor      import Capacitor
from MODELS.inductor       import Inductor
from MODELS.voltage_source import VoltageSource
from MODELS.current_source import CurrentSource
from MODELS.opamp          import OpAmp
from MODELS.diode          import Diode


# ==========================================================
# Utility
# ==========================================================

def parse_value(val):
    val = val.lower()
    scale = {
        'meg': 1e6, 't': 1e12, 'g': 1e9, 'k': 1e3,
        'm':   1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 'f': 1e-15
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
        return np.logspace(np.log10(fstart), np.log10(fstop),
                           int(npts * np.log10(fstop / fstart)))
    elif sweep_type == 'oct':
        return np.logspace(np.log10(fstart), np.log10(fstop),
                           int(npts * np.log2(fstop / fstart)))
    elif sweep_type == 'lin':
        return np.linspace(fstart, fstop, npts)
    raise RuntimeError(f"Unsupported AC sweep type: {sweep_type}")


def _is_dc_sweep_line(line):
    """
    Return True if a .dc line is a parameter sweep, False if it is a
    DC operating-point request (.dc op  or  .dc with no source name).

    SPICE .dc sweep syntax:  .dc  SrcName  start  stop  step
    SPICE .dc op   syntax :  .dc  op          (alias for .op)
    """
    t = line.split()
    # t[0] == '.dc'
    if len(t) < 2:
        return False                       # bare .dc → treat as .op
    second = t[1].lower()
    if second == 'op':
        return False                       # .dc op → operating point
    if len(t) < 5:
        return False                       # not enough tokens for a sweep
    # Try to parse t[2..4] as numbers; if they fail it's not a sweep
    try:
        float(t[2]); float(t[3]); float(t[4])
    except ValueError:
        return False
    return True


# ==========================================================
# Detect analysis type
# ==========================================================

def detect_analysis(netlist_file):
    with open(netlist_file) as f:
        for raw in f:
            line = strip_comments(raw).lower()
            if not line:
                continue
            if line.startswith('.dc'):
                return 'dc_sweep' if _is_dc_sweep_line(line) else 'dc'
            if line.startswith('.op'):   return 'dc'
            if line.startswith('.ac'):   return 'ac'
            if line.startswith('.tran'): return 'tran'
    raise RuntimeError("No analysis directive found (.dc / .op / .ac / .tran)")


# ==========================================================
# Netlist parser → component list
# ==========================================================

def build_circuit(netlist_file):

    components     = []
    nodes          = set()
    voltages       = []
    opamps         = []
    dc_sweeps      = []
    ac_sweep       = None
    tran_params    = None
    sens_node      = None
    print_requests = []

    with open(netlist_file) as f:
        for raw in f:
            line = strip_comments(raw)
            if not line:
                continue

            t    = line.split()
            name = t[0].lower()

            # ── Passives ──────────────────────────────────────────────
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

            # ── Voltage sources ───────────────────────────────────────
            elif name.startswith('v'):
                n1, n2 = t[1], t[2]
                match  = re.search(r'(\w+)\((.*?)\)', line)
                if match:
                    src_type = match.group(1).upper()
                    params   = [parse_value(x) for x in match.group(2).split()]
                else:
                    src_type = 'DC'
                    params   = [parse_value(t[3])]
                voltages.append((t[0], n1, n2, src_type, params))
                nodes.update([n1, n2])

            # ── Op-amp ────────────────────────────────────────────────
            elif name.startswith('o'):
                opamps.append(t)
                nodes.update([t[1], t[2], t[3]])

            # ── Diode ─────────────────────────────────────────────────
            elif name.startswith('d'):
                _, n1, n2 = t[:3]
                components.append(Diode(t[0], n1, n2))
                nodes.update([n1, n2])

            # ── MOSFETs ───────────────────────────────────────────────
            elif name.startswith('m'):
                _, d, g, s, b, mos_type = t[:6]
                mos_lo = mos_type.lower()
                if mos_lo == 'nmos':
                    from MODELS.nmos import NMOS
                    components.append(NMOS(t[0], d, g, s, b))
                elif mos_lo == 'pmos':
                    from MODELS.pmos import PMOS
                    components.append(PMOS(t[0], d, g, s, b))
                else:
                    raise RuntimeError(f"Unknown MOSFET type: {mos_type}")
                nodes.update([d, g, s, b])

            # ── Analysis directives ───────────────────────────────────
            elif name == '.tran':
                _, dt, tstop = t
                tran_params = (parse_value(dt), parse_value(tstop))

            elif name == '.ac':
                _, sweep_type, npts, fstart, fstop = t
                ac_sweep = (sweep_type.lower(), int(npts),
                            parse_value(fstart), parse_value(fstop))

            elif name == '.dc':
                if _is_dc_sweep_line(line.lower()):
                    # .dc SrcName start stop step [InnerSrc start stop step]
                    sp = {
                        'src':   t[1],
                        'start': parse_value(t[2]),
                        'stop':  parse_value(t[3]),
                        'step':  parse_value(t[4]),
                    }
                    if len(t) >= 9:
                        i_start = parse_value(t[6])
                        i_stop  = parse_value(t[7])
                        i_step  = parse_value(t[8])
                        i_vals  = list(np.arange(i_start,
                                                  i_stop + i_step * 0.5,
                                                  i_step))
                        sp['inner'] = {'src': t[5], 'values': i_vals}
                    dc_sweeps.append(sp)
                # else: .dc op → just a DC operating-point request, no sweep params needed

            elif name == '.op':
                pass   # DC operating point — no extra params

            elif name == '.sens':
                sens_node = t[1].lower()[2:-1]

            elif name == '.print':
                # .print v 2          → node voltage
                # .print v 1 2        → overlaid voltages on one subplot
                # .print i Vds        → branch current through named source
                print_requests.append((t[1], t[2:]))

            elif name == '.end':
                break

    # ── Node indexing ─────────────────────────────────────────────────
    nodes.discard('0')
    nodes      = sorted(nodes)
    node_index = {n: i for i, n in enumerate(nodes)}
    N          = len(nodes)

    # ── Voltage sources ───────────────────────────────────────────────
    for k, (vname, n1, n2, stype, params) in enumerate(voltages):
        components.append(VoltageSource(vname, n1, n2, stype, params, k))
    Mv = len(voltages)

    # ── Op-amps ───────────────────────────────────────────────────────
    for k, ot in enumerate(opamps):
        _, nplus, nminus, nout, gain = ot
        components.append(OpAmp(ot[0], nplus, nminus, nout,
                                parse_value(gain), k))
    Mo = len(opamps)

    print(f"  Parsed: N={N} nodes, Mv={Mv} V-sources, Mo={Mo} op-amps, "
          f"{len(components)} total components")

    return (components, node_index, N, Mv, Mo,
            ac_sweep, tran_params, sens_node, print_requests, dc_sweeps)


# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":

    if len(sys.argv) >= 2:
        NETLIST_FILE = sys.argv[1]
    else:
        NETLIST_FILE = "test_circuit.sp"
        print("Usage: python main.py <netlist.sp>")
        print(f"No netlist specified — using default: {NETLIST_FILE}\n")

    print("=== RUN START ===")
    print(f"Netlist : {NETLIST_FILE}")

    analysis = detect_analysis(NETLIST_FILE)
    print(f"Analysis: {analysis}")

    (components, node_index, N, Mv, Mo,
     ac_sweep, tran_params, sens_node,
     print_requests, dc_sweeps) = build_circuit(NETLIST_FILE)

    if analysis == 'dc':
        run_dc(components, node_index, N, Mv, Mo, sens_node, print_requests)

    elif analysis == 'dc_sweep':
        if not dc_sweeps:
            raise RuntimeError(
                "Analysis detected as dc_sweep but no valid .dc sweep line found.\n"
                "Sweep syntax: .dc SrcName start stop step\n"
                "Op-point syntax: .dc op  OR  .op"
            )
        run_dc_sweep(components, node_index, N, Mv, Mo,
                     dc_sweeps, print_requests)

    elif analysis == 'ac':
        if ac_sweep is None:
            raise RuntimeError("No .ac directive found")
        freqs = generate_ac_frequencies(ac_sweep)
        x_op  = run_dc(components, node_index, N, Mv, Mo,
                       sens_node=None, print_requests=None)
        run_ac(components, node_index, N, Mv, Mo,
               freqs, sens_node, print_requests, x_op=x_op)

    elif analysis == 'tran':
        if tran_params is None:
            raise RuntimeError("No .tran directive found")
        dt, tstop = tran_params
        run_tran(components, node_index, N, Mv, Mo,
                 dt, tstop, sens_node, print_requests)