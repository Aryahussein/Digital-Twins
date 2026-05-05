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
    t = line.split()
    if len(t) < 2:
        return False
    second = t[1].lower()
    if second == 'op':
        return False
    if len(t) < 5:
        return False
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
    diff_gain_req  = None

    with open(netlist_file) as f:
        for raw in f:
            line = strip_comments(raw)
            if not line:
                continue

            t    = line.split()
            name = t[0].lower()

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

            elif name.startswith('v'):
                n1, n2 = t[1], t[2]

                # Tokens after the two node names
                rest = t[3:]  # e.g. ['DC','0.44','AC','0.5','180']
                               #   or ['AC','0.5','180']
                               #   or ['SINE(0.55','0.01','1g','0)','AC','0.5']
                               #   or ['1.2']

                # ── Defaults ──────────────────────────────────────────
                src_type = 'DC'
                params   = [0.0]
                ac_mag   = None
                ac_phase = 0.0

                # ── Functional waveform: SINE(...) PULSE(...) etc. ────
                match = re.search(r'(\w+)\((.*?)\)', line)
                if match:
                    src_type = match.group(1).upper()
                    params   = [parse_value(x) for x in match.group(2).split()]
                    # AC spec may still follow the closing paren, e.g.
                    #   SINE(0.55 0.01 1g 0) AC 0.5
                    ac_idx = next((i for i, tok in enumerate(rest)
                                   if tok.lower() == 'ac'), None)
                    if ac_idx is not None and ac_idx + 1 < len(rest):
                        ac_mag   = parse_value(rest[ac_idx + 1])
                        ac_phase = parse_value(rest[ac_idx + 2]) \
                                   if ac_idx + 2 < len(rest) else 0.0

                else:
                    # Scan 'rest' for DC and AC keywords in any order
                    i = 0
                    while i < len(rest):
                        tok = rest[i].lower()

                        if tok == 'dc':
                            # DC <value>
                            if i + 1 < len(rest):
                                src_type = 'DC'
                                params   = [parse_value(rest[i + 1])]
                                i += 2
                            else:
                                i += 1

                        elif tok == 'ac':
                            # AC <mag> [phase]
                            if i + 1 < len(rest):
                                ac_mag = parse_value(rest[i + 1])
                                i += 2
                                if i < len(rest):
                                    try:
                                        ac_phase = parse_value(rest[i])
                                        i += 1
                                    except ValueError:
                                        pass  # next token is not a number
                            else:
                                i += 1

                        else:
                            # Bare value with no keyword → treat as DC
                            try:
                                src_type = 'DC'
                                params   = [parse_value(rest[i])]
                            except ValueError:
                                pass
                            i += 1

                    # Pure AC source (no DC keyword seen, but AC was found)
                    if src_type == 'DC' and params == [0.0] and ac_mag is not None:
                        src_type = 'AC'
                        params   = [ac_mag, ac_phase]
                        ac_mag   = None   # VoltageSource.__init__ reads from params

                voltages.append((t[0], n1, n2, src_type, params, ac_mag, ac_phase))
                nodes.update([n1, n2])

            elif name.startswith('o'):
                opamps.append(t)
                nodes.update([t[1], t[2], t[3]])

            elif name.startswith('d'):
                _, n1, n2 = t[:3]
                components.append(Diode(t[0], n1, n2))
                nodes.update([n1, n2])

            elif name.startswith('m'):

                name_tok, d, g, s, b, mos_type = t[:6]

                params = {"W": 1e-6, "L": 1e-6}

                for tok in t[6:]:
                    if '=' in tok:
                        k, v = tok.split('=')
                        params[k.upper()] = parse_value(v)

                mos_lo = mos_type.lower()

                if mos_lo == 'nmos':
                    from MODELS.nmos import NMOS
                    components.append(NMOS(name_tok, d, g, s, b,
                                           W=params["W"], L=params["L"]))
                elif mos_lo == 'pmos':
                    from MODELS.pmos import PMOS
                    components.append(PMOS(name_tok, d, g, s, b,
                                           W=params["W"], L=params["L"]))
                else:
                    raise RuntimeError(f"Unknown MOSFET type: {mos_type}")

                nodes.update([d, g, s, b])

            elif name == '.ac':
                _, sweep_type, npts, fstart, fstop = t
                ac_sweep = (sweep_type.lower(), int(npts),
                            parse_value(fstart), parse_value(fstop))

            elif name == '.tran':
                # Example:
                # .tran 1n 1u
                # .tran tstep tstop

                if len(t) < 3:
                    raise RuntimeError("Invalid .tran syntax")

                tstep = parse_value(t[1])
                tstop = parse_value(t[2])

                tran_params = {
                    'tstep': tstep,
                    'tstop': tstop
                }

            elif name == '.dc':
                # Base format:
                # .dc Vd 0 1.2 0.01
                # Extended:
                # .dc Vd 0 1.2 0.01 SWEEP Vg 0.2 1.0 0.2

                if len(t) < 5:
                    raise RuntimeError("Invalid .dc syntax")

                src   = t[1]
                start = parse_value(t[2])
                stop  = parse_value(t[3])
                step  = parse_value(t[4])

                sweep_dict = {
                    'src': src,
                    'start': start,
                    'stop': stop,
                    'step': step
                }

                # Check for nested sweep
                if len(t) > 5:
                    if t[5].lower() != 'sweep':
                        raise RuntimeError("Expected 'SWEEP' keyword in extended .dc")

                    if len(t) < 10:
                        raise RuntimeError("Invalid nested .dc syntax")

                    inner_src   = t[6]
                    inner_start = parse_value(t[7])
                    inner_stop  = parse_value(t[8])
                    inner_step  = parse_value(t[9])

                    # Build inner sweep values
                    if inner_step == 0:
                        raise RuntimeError("Inner .dc step cannot be zero")

                    if inner_step > 0:
                        values = np.arange(inner_start, inner_stop + inner_step*0.5, inner_step)
                    else:
                        values = np.arange(inner_start, inner_stop + inner_step*0.5, inner_step)

                    sweep_dict['inner'] = {
                        'src': inner_src,
                        'values': values
                    }

                dc_sweeps.append(sweep_dict)

            elif name == '.print':
                print_requests.append((t[1], t[2:]))

            elif name == '.diffgain':
                diff_gain_req = {
                    "in_pos":  t[1],
                    "in_neg":  t[2],
                    "out_pos": t[3],
                    "out_neg": t[4],
                }

            elif name == '.end':
                break

    nodes.discard('0')
    nodes      = sorted(nodes)
    node_index = {n: i for i, n in enumerate(nodes)}
    N          = len(nodes)

    for k, entry in enumerate(voltages):
        vname, n1, n2, stype, params = entry[:5]
        ac_mag   = entry[5] if len(entry) > 5 else None
        ac_phase = entry[6] if len(entry) > 6 else 0.0
        components.append(VoltageSource(vname, n1, n2, stype, params, k,
                                        ac_mag=ac_mag, ac_phase=ac_phase))
    Mv = len(voltages)

    Mo = len(opamps)

    return (components, node_index, N, Mv, Mo,
            ac_sweep, tran_params, sens_node,
            print_requests, dc_sweeps, diff_gain_req)


# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":

    NETLIST_FILE = sys.argv[1] if len(sys.argv) >= 2 else "NETLISTS/diff_amp_ac.sp"

    analysis = detect_analysis(NETLIST_FILE)

    (components, node_index, N, Mv, Mo,
     ac_sweep, tran_params, sens_node,
     print_requests, dc_sweeps, diff_gain_req) = build_circuit(NETLIST_FILE)

    if analysis == 'ac':
        freqs = generate_ac_frequencies(ac_sweep)
        x_op  = run_dc(components, node_index, N, Mv, Mo)

        run_ac(components, node_index, N, Mv, Mo,
               freqs, sens_node, print_requests,
               x_op=x_op,
               diff_gain_request=diff_gain_req)
        
    elif analysis == 'dc_sweep':
        run_dc_sweep(components, node_index, N, Mv, Mo,
                    dc_sweeps, print_requests)
        
    elif analysis == 'tran':
        if tran_params is None:
            raise RuntimeError(".tran specified but no parameters parsed")

        run_tran(components, node_index, N, Mv, Mo,
                tran_params,
                print_requests)