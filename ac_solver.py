import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# Helper functions
# ==========================================================

def strip_comments(line):
    for c in ['*', ';']:
        if c in line:
            line = line.split(c, 1)[0]
    return line.strip()


def parse_value(val):
    val = val.lower()
    scale = {
        't': 1e12, 'g': 1e9, 'meg': 1e6,
        'k': 1e3, 'm': 1e-3,
        'u': 1e-6, 'n': 1e-9, 'p': 1e-12
    }
    for s in scale:
        if val.endswith(s):
            return float(val[:-len(s)]) * scale[s]
    return float(val)


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
        raise ValueError(f"Unsupported AC sweep type: {sweep_type}")


# ==========================================================
# MAIN ENTRY FUNCTION
# ==========================================================

def run_ac(netlist_file):

    resistors = []
    capacitors = []
    inductors = []
    currents = []
    voltages = []
    opamps = []
    nodes = set()
    ac_sweep = None

    # ---------------- Parse Netlist ----------------
    with open(netlist_file) as f:
        for raw in f:
            line = strip_comments(raw)
            if not line:
                continue

            t = line.split()
            name = t[0].lower()

            if name.startswith('r'):
                _, n1, n2, v = t
                resistors.append((n1, n2, parse_value(v)))
                nodes.update([n1, n2])

            elif name.startswith('c'):
                _, n1, n2, v = t
                capacitors.append((n1, n2, parse_value(v)))
                nodes.update([n1, n2])

            elif name.startswith('l'):
                _, n1, n2, v = t
                inductors.append((n1, n2, parse_value(v)))
                nodes.update([n1, n2])

            elif name.startswith('i'):
                _, n_plus, n_minus, v = t
                currents.append((n_plus, n_minus, parse_value(v)))
                nodes.update([n_plus, n_minus])

            elif name.startswith('v'):
                _, n_plus, n_minus, v = t
                voltages.append((t[0], n_plus, n_minus, parse_value(v)))
                nodes.update([n_plus, n_minus])

            elif name.startswith('o'):
                # Oname n+ n- nout gain
                _, nplus, nminus, nout, gain = t
                opamps.append((nplus, nminus, nout, parse_value(gain)))
                nodes.update([nplus, nminus, nout])

            elif name == '.ac':
                _, sweep_type, npts, fstart, fstop = t
                ac_sweep = (
                    sweep_type.lower(),
                    int(npts),
                    parse_value(fstart),
                    parse_value(fstop)
                )

    if ac_sweep is None:
        raise RuntimeError("No .ac statement found in netlist")

    nodes.discard("0")
    nodes = sorted(nodes)

    N = len(nodes)
    Mv = len(voltages)
    Mo = len(opamps)

    node_idx = {n: i for i, n in enumerate(nodes)}

    size = N + Mv + Mo

    # ---------------- AC Solve Function ----------------
    def solve_ac(freq_hz):

        omega = 2 * np.pi * freq_hz
        j = 1j

        G = sp.zeros(size, size)
        Z = sp.zeros(size, 1)

        # -------- Resistors --------
        for n1, n2, R in resistors:
            g = 1 / R
            if n1 != "0":
                G[node_idx[n1], node_idx[n1]] += g
            if n2 != "0":
                G[node_idx[n2], node_idx[n2]] += g
            if n1 != "0" and n2 != "0":
                i, jdx = node_idx[n1], node_idx[n2]
                G[i, jdx] -= g
                G[jdx, i] -= g

        # -------- Capacitors --------
        for n1, n2, Cval in capacitors:
            yc = j * omega * Cval
            if n1 != "0":
                G[node_idx[n1], node_idx[n1]] += yc
            if n2 != "0":
                G[node_idx[n2], node_idx[n2]] += yc
            if n1 != "0" and n2 != "0":
                i, jdx = node_idx[n1], node_idx[n2]
                G[i, jdx] -= yc
                G[jdx, i] -= yc

        # -------- Inductors --------
        for n1, n2, Lval in inductors:
            yl = 1 / (j * omega * Lval)
            if n1 != "0":
                G[node_idx[n1], node_idx[n1]] += yl
            if n2 != "0":
                G[node_idx[n2], node_idx[n2]] += yl
            if n1 != "0" and n2 != "0":
                i, jdx = node_idx[n1], node_idx[n2]
                G[i, jdx] -= yl
                G[jdx, i] -= yl

        # -------- Current Sources --------
        for n_plus, n_minus, val in currents:
            if n_plus != "0":
                Z[node_idx[n_plus]] -= val
            if n_minus != "0":
                Z[node_idx[n_minus]] += val

        # -------- Voltage Sources --------
        for k, (name, n_plus, n_minus, val) in enumerate(voltages):
            row = N + k

            if n_plus != "0":
                G[row, node_idx[n_plus]] = 1
                G[node_idx[n_plus], row] = 1

            if n_minus != "0":
                G[row, node_idx[n_minus]] = -1
                G[node_idx[n_minus], row] = -1

            Z[row] = val

        # -------- Ideal Op Amps --------
        for k, (nplus, nminus, nout, gain) in enumerate(opamps):

            row = N + Mv + k

            # Vout - A(V+ - V-) = 0

            if nout != "0":
                G[row, node_idx[nout]] = 1
                G[node_idx[nout], row] = 1

            if nplus != "0":
                G[row, node_idx[nplus]] -= gain

            if nminus != "0":
                G[row, node_idx[nminus]] += gain

        X = G.LUsolve(Z)
        return np.array(X[:N], dtype=complex).flatten()

    # ---------------- Sweep ----------------
    frequencies = generate_ac_frequencies(ac_sweep)
    out_node = nodes[-1]
    out_idx = node_idx[out_node]

    mag = []
    phase = []

    for f in frequencies:
        V = solve_ac(f)
        vout = V[out_idx]
        mag.append(abs(vout))
        phase.append(np.angle(vout, deg=True))

    # ---------------- Plot ----------------
    fig, (ax_mag, ax_phase) = plt.subplots(
        2, 1, sharex=True, figsize=(7, 6)
    )

    ax_mag.semilogx(frequencies, 20 * np.log10(mag))
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.set_title(f"Bode Plot (Node {out_node})")
    ax_mag.grid(True, which="both")

    ax_phase.semilogx(frequencies, phase)
    ax_phase.set_xlabel("Frequency (Hz)")
    ax_phase.set_ylabel("Phase (deg)")
    ax_phase.grid(True, which="both")

    plt.tight_layout()
    plt.show()
